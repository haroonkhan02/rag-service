"""The load-test client against the in-process app (ASGI transport): no sockets."""

import asyncio
import dataclasses

import httpx
import pytest

from ragsvc.app import create_app
from ragsvc.config import RateLimitSettings, Settings
from ragsvc.generate import ExtractiveGenerator
from ragsvc.loadtest import (
    Sample,
    closed_loop,
    fixed_rate,
    percentile,
    post_ask,
    summarize,
    zipf_workload,
)
from ragsvc.service import RagService


def client_for(searcher, cfg=None):
    service = RagService(searcher, ExtractiveGenerator(), cfg or Settings())
    transport = httpx.ASGITransport(app=create_app(service))
    return httpx.AsyncClient(transport=transport, base_url="http://test")


def test_percentile_is_nearest_rank():
    v = list(range(1, 101))
    assert percentile(v, 50) == 51 and percentile(v, 95) == 95 and percentile(v, 100) == 100
    assert percentile([7.0], 99) == 7.0


def test_summarize_separates_failures_and_counts_cache_labels():
    samples = [Sample(10, 200, "miss"), Sample(2, 200, "exact"), Sample(4, 200, "exact"),
               Sample(1, 429), Sample(1, 503)]
    s = summarize(samples, wall_s=2.0)
    assert (s["requests"], s["ok"], s["throughput_rps"]) == (5, 3, 1.5)
    assert s["status"] == {"200": 3, "429": 1, "503": 1}
    assert s["cache"] == {"exact": 2, "miss": 1}
    assert s["latency_ms"]["p50"] == 4 and s["latency_ms"]["max"] == 10
    assert "latency_ms" not in summarize([Sample(1, 500)], 1.0)


def test_zipf_workload_is_seeded_skewed_and_mixes_paraphrases():
    qs = [f"q{i}" for i in range(20)]
    a = zipf_workload(qs, 500, seed=3)
    assert a == zipf_workload(qs, 500, seed=3) and a != zipf_workload(qs, 500, seed=4)
    counts = sorted((a.count(q) for q in qs), reverse=True)
    assert counts[0] > 5 * counts[-1] and set(a) <= set(qs)
    b = zipf_workload(qs, 500, seed=3, paraphrases={i: f"p{i}" for i in range(20)},
                      paraphrase_share=0.5)
    share = sum(q.startswith("p") for q in b) / len(b)
    assert 0.4 < share < 0.6


def test_closed_loop_reports_cache_hits(searcher):
    async def run():
        cfg = dataclasses.replace(Settings(), rate_limit=RateLimitSettings(enabled=False))
        async with client_for(searcher, cfg) as client:
            queries = ["ros2 bag record", "what is a topic"] * 5
            return await closed_loop(client, queries, concurrency=1, mode="hybrid")

    samples, wall = asyncio.run(run())
    s = summarize(samples, wall)
    assert s["ok"] == 10 and s["cache"] == {"exact": 8, "miss": 2}


def test_fixed_rate_shows_the_rate_limiter(searcher):
    async def run():
        cfg = dataclasses.replace(Settings(),
                                  rate_limit=RateLimitSettings(rate_per_s=1.0, burst=5))
        async with client_for(searcher, cfg) as client:
            return await fixed_rate(client, ["ros2 bag record"] * 20, rate_per_s=400,
                                    mode="bm25")

    samples, _ = asyncio.run(run())
    codes = [s.status for s in samples]
    assert codes.count(200) == 5 and codes.count(429) == 15


def test_post_ask_measures_time_to_first_token(searcher):
    async def run():
        async with client_for(searcher) as client:
            first = await post_ask(client, "How do I record topics with ros2 bag?", "hybrid")
            again = await post_ask(client, "How do I record topics with ros2 bag?", "hybrid")
            bad = await post_ask(client, "q", "fuzzy")
            return first, again, bad

    first, again, bad = asyncio.run(run())
    assert first.status == 200 and first.cache == "miss"
    assert first.ttft_ms is not None and first.ttft_ms <= first.latency_ms
    assert first.extra["citations"] == 1 and first.extra["stop_reason"] == "end_turn"
    assert again.cache == "exact" and bad.status == 422


@pytest.mark.parametrize("n", [0, 3])
def test_closed_loop_with_more_workers_than_requests(searcher, n):
    async def run():
        async with client_for(searcher) as client:
            return await closed_loop(client, ["ros2 bag record"] * n, concurrency=8,
                                     mode="bm25")

    samples, _ = asyncio.run(run())
    assert len(samples) == n
