"""Generators with fake clients / models: no network, no key, no model download."""

import asyncio
from types import SimpleNamespace

import pytest

from ragsvc.chunking import Chunk
from ragsvc.generate import (
    ABSTAIN,
    FALLBACK_BETA,
    AnthropicGenerator,
    ExtractiveGenerator,
    GenResult,
    MLXGenerator,
    build_user_message,
    cost_usd,
    parse_citations,
)
from ragsvc.search import Hit


def hit(text="Topics\n\nTopics carry data streams. They are typed. More detail here.", page=None):
    return Hit(Chunk("Concepts/Topics::0", "Concepts/Topics", "Topics", "Topics > Intro",
                     text, "https://x/topics", page), 1.0)


async def collect(gen, question, hits):
    pieces, result = [], None
    async for item in gen.stream(question, hits):
        if isinstance(item, GenResult):
            result = item
        else:
            pieces.append(item)
    return pieces, result


class FakeStream:
    def __init__(self, texts, final):
        self._texts, self._final = texts, final

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def text_stream(self):
        async def gen():
            for t in self._texts:
                yield t
        return gen()

    async def get_final_message(self):
        return self._final


class FakeAnthropic:
    def __init__(self, texts, stop_reason="end_turn", model="claude-opus-5-5"):
        self.calls = []
        final = SimpleNamespace(stop_reason=stop_reason, model=model,
                                usage=SimpleNamespace(input_tokens=1000, output_tokens=50))
        outer = self

        class Messages:
            def stream(self, **kwargs):
                outer.calls.append(kwargs)
                return FakeStream(texts, final)

        self.beta = SimpleNamespace(messages=Messages())


def test_prompt_numbers_passages_and_keeps_the_question_last():
    msg = build_user_message("What is a topic?", [hit(), hit("Paper\n\nNav2 text.", page=3)])
    assert msg.startswith("<passages>\n[1] Topics > Intro\nSource: https://x/topics\n")
    assert "[2] Topics > Intro (page 3)" in msg
    assert msg.endswith("Question: What is a topic?")


def test_parse_citations_orders_by_first_use():
    assert parse_citations("Use it [2]. Also [1][2] and [10].") == [2, 1, 10]
    assert parse_citations("no citations, list[int] or [abc]") == []


def test_cost_uses_list_prices_and_is_zero_for_unknown_models():
    assert cost_usd("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(6.0)
    assert cost_usd("mlx-community/whatever", 10, 10) == 0.0


def test_extractive_quotes_the_top_passage():
    pieces, res = asyncio.run(collect(ExtractiveGenerator(), "q", [hit()]))
    assert res.text == "Topics carry data streams. They are typed. [1]"
    assert "".join(pieces).strip() == res.text
    assert res.model == "extractive" and not res.abstained
    _, res = asyncio.run(collect(ExtractiveGenerator(), "q", []))
    assert res.text == ABSTAIN and res.abstained


def test_anthropic_generator_streams_and_reports_usage():
    client = FakeAnthropic(["Topics carry ", "data [1]."])
    gen = AnthropicGenerator(client=client)
    pieces, res = asyncio.run(collect(gen, "What is a topic?", [hit()]))
    assert pieces == ["Topics carry ", "data [1]."]
    assert res.text == "Topics carry data [1]." and res.stop_reason == "end_turn"
    assert (res.input_tokens, res.output_tokens) == (1000, 50)
    assert res.cost_usd == pytest.approx(1000 * 4e-6 + 50 * 20e-6)
    assert res.ttft_ms is not None
    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["output_config"] == {"effort": "low"}
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    assert "temperature" not in call and "thinking" not in call
    assert call["messages"][0]["content"].endswith("Question: What is a topic?")


def test_anthropic_refusal_and_fallback_model_are_reported():
    client = FakeAnthropic(["partial"], stop_reason="refusal")
    _, res = asyncio.run(collect(AnthropicGenerator(client=client), "q", [hit()]))
    assert res.refused
    client = FakeAnthropic(["ok [1]"], model="claude-opus-4-8")
    _, res = asyncio.run(collect(AnthropicGenerator(client=client), "q", [hit()]))
    assert res.model == "claude-opus-4-8" and not res.refused


def test_anthropic_without_passages_abstains_without_calling_the_api():
    client = FakeAnthropic(["x"])
    pieces, res = asyncio.run(collect(AnthropicGenerator(client=client), "q", []))
    assert pieces == [ABSTAIN] and res.abstained and client.calls == []


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        return "|".join(m["role"] for m in messages)


def fake_mlx(pieces, finish="stop"):
    def generate(model, tokenizer, prompt, max_tokens):
        assert prompt == "system|user"
        for i, p in enumerate(pieces):
            yield SimpleNamespace(text=p, prompt_tokens=900, generation_tokens=i + 1,
                                  finish_reason=finish if i == len(pieces) - 1 else None)
    return generate


def test_mlx_generator_bridges_the_sync_model_loop():
    gen = MLXGenerator("org/model-4bit", loader=lambda name: ("M", FakeTokenizer()),
                       generator=fake_mlx(["Use ", "", "`ros2 bag` [1]."]))
    pieces, res = asyncio.run(collect(gen, "q", [hit()]))
    assert pieces == ["Use ", "`ros2 bag` [1]."]
    assert res.text == "Use `ros2 bag` [1]." and res.model == "org/model-4bit"
    assert (res.input_tokens, res.output_tokens) == (900, 3)
    assert res.stop_reason == "end_turn" and res.cost_usd == 0.0
    assert gen.name == "mlx:model-4bit"


def test_mlx_generator_reports_truncation_and_model_errors():
    gen = MLXGenerator("m", loader=lambda name: ("M", FakeTokenizer()),
                       generator=fake_mlx(["a"], finish="length"))
    _, res = asyncio.run(collect(gen, "q", [hit()]))
    assert res.stop_reason == "max_tokens"

    def broken(model, tokenizer, prompt, max_tokens):
        raise RuntimeError("out of memory")
        yield  # pragma: no cover

    gen = MLXGenerator("m", loader=lambda name: ("M", FakeTokenizer()), generator=broken)
    with pytest.raises(RuntimeError, match="out of memory"):
        asyncio.run(collect(gen, "q", [hit()]))
