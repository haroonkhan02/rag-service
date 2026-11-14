"""Answer generation: numbered passages in, a cited answer streamed out.

Three backends behind one async interface (`stream(question, hits)` yields text pieces and
ends with a `GenResult`):

    anthropic    Claude through the Anthropic SDK, streamed. Requires an Anthropic API key,
                 which the SDK reads from the environment; it is never part of the settings.
    mlx          an open-weight model running locally on Apple silicon (no API cost)
    extractive   no LLM: quotes the top passage. Keeps the service usable without a model.

The prompt and the citation format are the ones from robotics-docs-rag.
"""

from __future__ import annotations

import asyncio
import re
import threading
import time
from collections.abc import AsyncIterator
from dataclasses import asdict, dataclass
from typing import Protocol

from ragsvc.config import GenerationSettings
from ragsvc.search import Hit

ABSTAIN = "I can't answer that from the ROS 2 documentation I have."
SYSTEM_PROMPT = f"""You answer questions about ROS 2 (Robot Operating System 2) for robotics \
engineers, using only the numbered documentation passages supplied with each question.

Rules:
- Every sentence that states a fact must end with the number(s) of the passage(s) that \
support it, in square brackets, e.g. "Use transient local durability [2]." or "[1][3]".
- Use only information from the passages. Do not add facts from memory, even if you know them.
- If the passages do not contain the answer, reply with exactly: "{ABSTAIN}"
- Be concise: answer first, then any needed detail. Put commands, code and names in \
backticks. Copy commands exactly as they appear in the passages."""

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# USD per million tokens (input, output), Anthropic list prices
PRICES = {"claude-opus-5-5": (4.0, 20.0), "claude-sonnet-5-5": (2.0, 10.0),
          "claude-haiku-5-5": (0.10, 0.50)}
_CITATION = re.compile(r"\[(\d{1,2})\]")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


def format_passages(hits: list[Hit]) -> str:
    blocks = []
    for i, h in enumerate(hits, start=1):
        c = h.chunk
        where = c.section + (f" (page {c.page})" if c.page and "page" not in c.section else "")
        blocks.append(f"[{i}] {where}\nSource: {c.source}\n{c.text}")
    return "\n\n".join(blocks)


def build_user_message(question: str, hits: list[Hit]) -> str:
    return f"<passages>\n{format_passages(hits)}\n</passages>\n\nQuestion: {question}"


def parse_citations(text: str) -> list[int]:
    """Passage numbers cited in the answer, in order of first use."""
    seen: list[int] = []
    for m in _CITATION.finditer(text):
        n = int(m.group(1))
        if n not in seen:
            seen.append(n)
    return seen


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    price_in, price_out = PRICES.get(model, (0.0, 0.0))
    return (input_tokens * price_in + output_tokens * price_out) / 1e6


@dataclass
class GenResult:
    text: str
    model: str
    stop_reason: str = "end_turn"
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: float | None = None  # time to the first piece of text
    latency_ms: float = 0.0
    cost_usd: float = 0.0

    @property
    def refused(self) -> bool:
        return self.stop_reason == "refusal"

    @property
    def abstained(self) -> bool:
        return ABSTAIN.lower().rstrip(".") in self.text.lower()

    def as_dict(self) -> dict:
        return {**asdict(self), "abstained": self.abstained}


class Generator(Protocol):
    name: str

    def stream(self, question: str, hits: list[Hit]) -> AsyncIterator[str | GenResult]: ...


class _Timer:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.first: float | None = None

    def mark(self) -> None:
        if self.first is None:
            self.first = (time.perf_counter() - self.t0) * 1e3

    @property
    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self.t0) * 1e3


class ExtractiveGenerator:
    """No model: the first sentences of the best passage, cited as [1]."""

    name = "extractive"

    def __init__(self, max_sentences: int = 2):
        self.max_sentences = max_sentences

    async def stream(self, question: str, hits: list[Hit]) -> AsyncIterator[str | GenResult]:
        timer = _Timer()
        if hits:
            body = hits[0].chunk.text.split("\n\n", 1)[-1]  # drop the section header line
            sentences = _SENTENCE.split(" ".join(body.split()))[: self.max_sentences]
            text = " ".join(sentences).rstrip() + " [1]"
        else:
            text = ABSTAIN
        for word in text.split(" "):
            timer.mark()
            yield word + " "
            await asyncio.sleep(0)
        yield GenResult(text, self.name, ttft_ms=timer.first, latency_ms=timer.elapsed_ms)


class AnthropicGenerator:
    """Claude, streamed. Thinking is adaptive (always on for Opus 5.5); depth via `effort`.

    Server-side refusal fallbacks are on: if a safety classifier declines, the API retries
    on a fallback model inside the same stream. A refusal of the whole chain ends the
    stream with `stop_reason == "refusal"`; the caller must then discard any partial text.
    """

    def __init__(self, model: str = "claude-opus-5-5", effort: str = "low",
                 max_tokens: int = 4000, client=None):
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic(max_retries=3)
        self.client, self.model, self.effort, self.max_tokens = client, model, effort, max_tokens
        self.name = f"anthropic:{model}:{effort}"

    async def stream(self, question: str, hits: list[Hit]) -> AsyncIterator[str | GenResult]:
        timer = _Timer()
        if not hits:
            yield ABSTAIN
            yield GenResult(ABSTAIN, self.model, latency_ms=timer.elapsed_ms)
            return
        parts: list[str] = []
        async with self.client.beta.messages.stream(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": build_user_message(question, hits)}],
            output_config={"effort": self.effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        ) as stream:
            async for text in stream.text_stream:
                timer.mark()
                parts.append(text)
                yield text
            final = await stream.get_final_message()
        served_by = getattr(final, "model", None) or self.model
        usage = final.usage
        yield GenResult(
            text="".join(parts).strip(), model=served_by,
            stop_reason=final.stop_reason or "end_turn",
            input_tokens=usage.input_tokens, output_tokens=usage.output_tokens,
            ttft_ms=timer.first, latency_ms=timer.elapsed_ms,
            cost_usd=cost_usd(served_by, usage.input_tokens, usage.output_tokens),
        )


class MLXGenerator:
    """A local open-weight model via mlx-lm (Apple silicon). Greedy decoding.

    mlx-lm generates synchronously, so the loop runs in a thread and hands pieces to the
    event loop through a queue. One generation at a time: the GPU is the bottleneck, and
    queueing here keeps time-to-first-token honest under concurrent requests.
    """

    def __init__(self, model: str, max_tokens: int = 400, loader=None, generator=None):
        if loader is None or generator is None:
            import mlx_lm

            loader = loader or mlx_lm.load
            generator = generator or mlx_lm.stream_generate
        self.model, self.tokenizer = loader(model)
        self._generate = generator
        self.model_id, self.max_tokens = model, max_tokens
        self.name = f"mlx:{model.split('/')[-1]}"
        self._gpu = asyncio.Lock()

    def _prompt(self, question: str, hits: list[Hit]) -> str:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(question, hits)}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False,
                                                  add_generation_prompt=True)

    async def stream(self, question: str, hits: list[Hit]) -> AsyncIterator[str | GenResult]:
        timer = _Timer()
        if not hits:
            yield ABSTAIN
            yield GenResult(ABSTAIN, self.model_id, latency_ms=timer.elapsed_ms)
            return
        prompt = self._prompt(question, hits)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        done = object()

        def worker() -> None:
            last = None
            try:
                for last in self._generate(self.model, self.tokenizer, prompt,
                                           max_tokens=self.max_tokens):
                    if stop.is_set():  # the client went away
                        break
                    loop.call_soon_threadsafe(queue.put_nowait, last.text)
                loop.call_soon_threadsafe(queue.put_nowait, (done, last))
            except BaseException as exc:  # surface model errors in the request
                loop.call_soon_threadsafe(queue.put_nowait, exc)

        async with self._gpu:
            timer = _Timer()  # time in the GPU queue is not model latency
            task = loop.run_in_executor(None, worker)
            parts: list[str] = []
            last = None
            try:
                while True:
                    item = await queue.get()
                    if isinstance(item, BaseException):
                        raise item
                    if isinstance(item, tuple) and item[0] is done:
                        last = item[1]
                        break
                    if item:
                        timer.mark()
                        parts.append(item)
                        yield item
            finally:
                stop.set()
                await task
        finish = getattr(last, "finish_reason", None)
        yield GenResult(
            text="".join(parts).strip(), model=self.model_id,
            stop_reason="max_tokens" if finish == "length" else "end_turn",
            input_tokens=int(getattr(last, "prompt_tokens", 0) or 0),
            output_tokens=int(getattr(last, "generation_tokens", 0) or 0),
            ttft_ms=timer.first, latency_ms=timer.elapsed_ms,
        )


def make_generator(cfg: GenerationSettings) -> Generator:
    if cfg.backend == "anthropic":
        return AnthropicGenerator(cfg.model, cfg.effort, cfg.max_tokens)
    if cfg.backend == "mlx":
        return MLXGenerator(cfg.mlx_model, cfg.mlx_max_tokens)
    return ExtractiveGenerator()
