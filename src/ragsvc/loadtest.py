"""Load-test client: workloads, a closed-loop and a fixed-rate driver, latency summaries.

    closed loop   `concurrency` workers, each sends its next request when the previous one
                  returns. Measures throughput and latency at a given number of clients.
    fixed rate    requests leave on a schedule whether or not earlier ones are back (open
                  loop). Used for the rate limiter, where the arrival rate is the point.

A closed loop under-reports tail latency when the server stalls, because stalled workers
stop sending (coordinated omission); read its p95/p99 as "per client", not "per arrival".
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field

import httpx


@dataclass
class Sample:
    latency_ms: float
    status: int
    cache: str | None = None
    ttft_ms: float | None = None  # /ask streams only
    extra: dict = field(default_factory=dict)


def percentile(values: Sequence[float], p: float) -> float:
    """Nearest-rank percentile (p in 0..100) of a non-empty sample."""
    v = sorted(values)
    return v[min(len(v) - 1, max(0, round(p / 100 * (len(v) - 1))))]


def summarize(samples: Sequence[Sample], wall_s: float) -> dict:
    ok = [s for s in samples if s.status == 200]
    lat = [s.latency_ms for s in ok]
    out = {
        "requests": len(samples), "ok": len(ok), "wall_s": round(wall_s, 2),
        "status": dict(sorted(Counter(str(s.status) for s in samples).items())),
        "throughput_rps": round(len(ok) / wall_s, 2) if wall_s else 0.0,
        "cache": dict(sorted(Counter(s.cache for s in ok if s.cache).items())),
    }
    if lat:
        out["latency_ms"] = {"mean": round(sum(lat) / len(lat), 2),
                             **{f"p{p}": round(percentile(lat, p), 2) for p in (50, 95, 99)},
                             "max": round(max(lat), 2)}
    ttft = [s.ttft_ms for s in ok if s.ttft_ms is not None]
    if ttft:
        out["ttft_ms"] = {f"p{p}": round(percentile(ttft, p), 1) for p in (50, 95)}
    return out


def zipf_workload(questions: Sequence[str], n: int, s: float = 1.1, seed: int = 0,
                  paraphrases: dict[int, str] | None = None,
                  paraphrase_share: float = 0.0) -> list[str]:
    """n requests where the i-th most popular question is asked with weight 1 / i**s.

    Popularity rank is a seeded shuffle of the questions. With `paraphrases` (question
    index -> other wording), that share of the requests uses the other wording.
    """
    rng = random.Random(seed)
    order = list(range(len(questions)))
    rng.shuffle(order)
    weights = [1.0 / (rank ** s) for rank in range(1, len(order) + 1)]
    out = []
    for i in rng.choices(order, weights=weights, k=n):
        if paraphrases and i in paraphrases and rng.random() < paraphrase_share:
            out.append(paraphrases[i])
        else:
            out.append(questions[i])
    return out


async def post_search(client: httpx.AsyncClient, query: str, mode: str | None = None,
                      headers: dict | None = None) -> Sample:
    body = {"query": query} | ({"mode": mode} if mode else {})
    t0 = time.perf_counter()
    try:
        r = await client.post("/search", json=body, headers=headers)
        latency = (time.perf_counter() - t0) * 1e3
        cache = r.json().get("cache") if r.status_code == 200 else None
        return Sample(latency, r.status_code, cache)
    except httpx.HTTPError:
        return Sample((time.perf_counter() - t0) * 1e3, 0)


async def post_ask(client: httpx.AsyncClient, query: str, mode: str | None = None,
                   headers: dict | None = None) -> Sample:
    """Streamed /ask: time to the first `token` event and to the end of the stream."""
    body = {"query": query} | ({"mode": mode} if mode else {})
    t0 = time.perf_counter()
    ttft, done, event = None, {}, None
    try:
        async with client.stream("POST", "/ask", json=body, headers=headers) as r:
            if r.status_code != 200:
                await r.aread()
                return Sample((time.perf_counter() - t0) * 1e3, r.status_code)
            async for line in r.aiter_lines():
                if line.startswith("event: "):
                    event = line[7:]
                    if event == "token" and ttft is None:
                        ttft = (time.perf_counter() - t0) * 1e3
                elif line.startswith("data: ") and event in ("done", "refusal", "error"):
                    done = json.loads(line[6:])
    except httpx.HTTPError:
        return Sample((time.perf_counter() - t0) * 1e3, 0)
    latency = (time.perf_counter() - t0) * 1e3
    if done.get("type") != "done":
        return Sample(latency, 599, extra=done)  # stream ended without an answer
    return Sample(latency, 200, done.get("cache"), ttft, {
        "output_tokens": done["usage"]["output_tokens"],
        "input_tokens": done["usage"]["input_tokens"],
        "generation_ms": done["timings_ms"]["generation"],
        "abstained": done["abstained"], "citations": len(done["citations"]),
        "invalid_citations": len(done["invalid_citations"]),
        "stop_reason": done["stop_reason"]})


async def closed_loop(client: httpx.AsyncClient, queries: Sequence[str], concurrency: int,
                      send=post_search, **kw) -> tuple[list[Sample], float]:
    it = iter(queries)
    samples: list[Sample] = []

    async def worker() -> None:
        for q in it:
            samples.append(await send(client, q, **kw))

    t0 = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return samples, time.perf_counter() - t0


async def fixed_rate(client: httpx.AsyncClient, queries: Sequence[str], rate_per_s: float,
                     send=post_search, **kw) -> tuple[list[Sample], float]:
    t0 = time.perf_counter()
    tasks = []
    for i, q in enumerate(queries):
        delay = t0 + i / rate_per_s - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        tasks.append(asyncio.create_task(send(client, q, **kw)))
    samples = list(await asyncio.gather(*tasks))
    return samples, time.perf_counter() - t0


class ServerProcess:
    """`python -m ragsvc.serve` with settings overrides, stopped on exit."""

    def __init__(self, port: int, overrides: dict[str, str] | None = None,
                 ready_timeout_s: float = 300.0):
        self.port, self.overrides, self.ready_timeout_s = port, overrides or {}, ready_timeout_s
        self.url = f"http://127.0.0.1:{port}"
        self.proc: subprocess.Popen | None = None

    def __enter__(self) -> ServerProcess:
        env = {**os.environ, **{f"RAGSVC__{k}": str(v) for k, v in self.overrides.items()}}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "ragsvc.serve", "--port", str(self.port)], env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + self.ready_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"server exited with code {self.proc.returncode}")
            try:
                if httpx.get(f"{self.url}/readyz", timeout=1.0).status_code == 200:
                    return self
            except httpx.HTTPError:
                time.sleep(0.5)
        self.__exit__()
        raise RuntimeError("server did not become ready")

    def __exit__(self, *exc) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    def metrics(self) -> dict:
        return httpx.get(f"{self.url}/metrics", timeout=5.0).json()
