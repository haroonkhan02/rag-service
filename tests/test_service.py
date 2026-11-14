"""The request pipeline with a hashing embedder and a counting searcher: no models."""

import asyncio
import dataclasses
import threading

import pytest

from ragsvc.config import CacheSettings, ServerSettings, Settings
from ragsvc.generate import ExtractiveGenerator, GenResult
from ragsvc.service import Overloaded, RagService


def settings(**cache):
    return dataclasses.replace(Settings(), cache=CacheSettings(**cache))


class Counting:
    """Wraps a searcher: counts real searches and can hold them until released."""

    def __init__(self, searcher, gate: threading.Event | None = None):
        self._s, self.calls, self.gate = searcher, 0, gate
        self.embedder, self.bundle, self.chunks = searcher.embedder, searcher.bundle, \
            searcher.chunks

    def search(self, query, mode=None, k=None, query_vec=None):
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        return self._s.search(query, mode, k, query_vec)


def make(searcher, generator=None, cfg=None, gate=None):
    counting = Counting(searcher, gate)
    return RagService(counting, generator or ExtractiveGenerator(), cfg or settings()), counting


async def events(service, query, **kw):
    return [ev async for ev in service.ask(query, **kw)]


def test_second_identical_search_is_an_exact_cache_hit(searcher):
    async def run():
        svc, counting = make(searcher)
        first = await svc.search("How do I record a bag?", "hybrid", 3)
        again = await svc.search("how do i record a bag", "hybrid", 3)
        other_k = await svc.search("How do I record a bag?", "hybrid", 2)
        return first, again, other_k, counting.calls

    first, again, other_k, calls = asyncio.run(run())
    assert first.cache == "miss" and again.cache == "exact" and other_k.cache == "miss"
    assert [h.chunk.chunk_id for h in again.hits] == [h.chunk.chunk_id for h in first.hits]
    assert calls == 2
    assert "bm25" in first.timings_ms and "embed" in first.timings_ms
    assert set(again.timings_ms) == {"service_total"}


def test_paraphrase_hits_the_semantic_cache_only_above_the_threshold(searcher):
    async def run(threshold):
        svc, counting = make(searcher, cfg=settings(semantic_threshold=threshold))
        await svc.search("record topics with ros2 bag record", "hybrid", 3)
        out = await svc.search("record topics with ros2 bag record please", "hybrid", 3)
        far = await svc.search("what do services return", "hybrid", 3)
        return out, far, counting.calls

    out, far, calls = asyncio.run(run(0.8))
    assert out.cache == "semantic" and out.similarity > 0.8
    assert out.matched_query == "record topics with ros2 bag record"
    assert far.cache == "miss" and calls == 2
    out, _, calls = asyncio.run(run(0.999))
    assert out.cache == "miss" and calls == 3


def test_caches_can_be_switched_off(searcher):
    async def run():
        svc, counting = make(searcher, cfg=settings(exact=False, semantic=False))
        a = await svc.search("ros2 bag record", "bm25", 2)
        b = await svc.search("ros2 bag record", "bm25", 2)
        return a.cache, b.cache, counting.calls, len(svc.search_cache)

    assert asyncio.run(run()) == ("miss", "miss", 2, 0)


def test_concurrent_identical_searches_share_one_computation(searcher):
    async def run():
        gate = threading.Event()
        svc, counting = make(searcher, gate=gate)
        tasks = [asyncio.create_task(svc.search("ros2 bag record", "bm25", 2))
                 for _ in range(5)]
        await asyncio.sleep(0.05)
        gate.set()
        outs = await asyncio.gather(*tasks)
        return sorted(o.cache for o in outs), counting.calls

    caches, calls = asyncio.run(run())
    assert caches == ["miss", "shared", "shared", "shared", "shared"] and calls == 1


def test_full_queue_sheds_load_instead_of_waiting(searcher):
    async def run():
        gate = threading.Event()
        cfg = dataclasses.replace(settings(exact=False, semantic=False),
                                  server=ServerSettings(model_workers=1, max_queue=1))
        svc, _ = make(searcher, cfg=cfg, gate=gate)
        tasks = [asyncio.create_task(svc.search(f"ros2 bag record {i}", "bm25", 2))
                 for i in range(4)]
        await asyncio.sleep(0.05)
        gate.set()
        outs = await asyncio.gather(*tasks, return_exceptions=True)
        return outs, svc.metrics.counters

    outs, counters = asyncio.run(run())
    assert sum(isinstance(o, Overloaded) for o in outs) == 2
    assert counters["shed"] == 2


def test_failed_search_is_not_cached_and_reaches_every_waiter(searcher):
    class Broken(Counting):
        def search(self, *a, **kw):
            self.calls += 1
            raise RuntimeError("index gone")

    async def run():
        svc = RagService(Broken(searcher), ExtractiveGenerator(), settings())
        outs = await asyncio.gather(*(svc.search("q", "bm25", 2) for _ in range(3)),
                                    return_exceptions=True)
        return outs, len(svc.search_cache)

    outs, entries = asyncio.run(run())
    assert all(isinstance(o, RuntimeError) for o in outs) and entries == 0


def test_resolve_rejects_bad_mode_and_k(searcher):
    svc, _ = make(searcher)
    assert svc.resolve(None, None) == ("hybrid_rerank", 5)
    with pytest.raises(ValueError, match="unknown mode"):
        svc.resolve("fuzzy", 3)
    with pytest.raises(ValueError, match="k must be"):
        svc.resolve("bm25", 31)


def test_ask_streams_passages_tokens_and_a_final_summary(searcher):
    svc, _ = make(searcher)
    evs = asyncio.run(events(svc, "How do I record topics with ros2 bag?", mode="hybrid", k=2))
    assert [e["type"] for e in evs][:2] == ["passages", "token"]
    done = evs[-1]
    assert done["type"] == "done" and done["cache"] == "miss"
    assert "".join(e["text"] for e in evs if e["type"] == "token").strip() == done["answer"]
    assert done["citations"] == [1] and done["invalid_citations"] == []
    assert len(evs[0]["passages"]) == 2 and evs[0]["passages"][0]["n"] == 1


class Scripted:
    def __init__(self, text="Answer [1] and [7].", stop_reason="end_turn", cost=0.01):
        self.name, self.text, self.stop_reason, self.cost, self.calls = \
            "scripted", text, stop_reason, cost, 0

    async def stream(self, question, hits):
        self.calls += 1
        yield self.text
        yield GenResult(self.text, "m", self.stop_reason, 100, 10, 5.0, 20.0, self.cost)


def test_repeated_ask_is_served_from_the_answer_cache_at_no_cost(searcher):
    async def run():
        gen = Scripted()
        svc, counting = make(searcher, gen)
        first = await events(svc, "What is a topic?", mode="hybrid", k=2)
        again = await events(svc, "what is a topic", mode="hybrid", k=2)
        return first[-1], again, gen.calls, counting.calls

    first, again, gen_calls, searches = asyncio.run(run())
    assert first["usage"]["cost_usd"] == 0.01 and first["invalid_citations"] == [7]
    assert [e["type"] for e in again] == ["passages", "token", "done"]
    done = again[-1]
    assert done["cache"] == "exact" and done["answer"] == first["answer"]
    assert done["usage"] == {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                             "saved_cost_usd": 0.01}
    assert gen_calls == 1 and searches == 1


def test_paraphrase_reuses_the_cached_answer(searcher):
    async def run():
        gen = Scripted()
        svc, _ = make(searcher, gen, settings(semantic_threshold=0.8))
        await events(svc, "record topics with ros2 bag record", mode="hybrid", k=2)
        again = await events(svc, "record topics with ros2 bag record please",
                             mode="hybrid", k=2)
        return again[-1], gen.calls

    done, gen_calls = asyncio.run(run())
    assert done["cache"] == "semantic" and done["similarity"] > 0.8 and gen_calls == 1


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_refused_or_truncated_answers_are_not_cached(searcher, stop_reason):
    async def run():
        gen = Scripted(stop_reason=stop_reason)
        svc, _ = make(searcher, gen)
        first = await events(svc, "What is a topic?", mode="hybrid", k=2)
        await events(svc, "What is a topic?", mode="hybrid", k=2)
        return first[-1]["type"], gen.calls, len(svc.answer_cache)

    last, gen_calls, entries = asyncio.run(run())
    assert last == ("refusal" if stop_reason == "refusal" else "done")
    assert gen_calls == 2 and entries == 0
