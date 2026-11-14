"""The request pipeline behind the HTTP layer: cache -> search -> generate, with limits.

    search   exact cache -> (embed) -> semantic cache -> BM25 + dense + RRF + rerank
    ask      answer cache -> search -> generator stream -> answer cache

Model calls (embedding, cross-encoder) block, so they run on a small worker pool and the
event loop stays free. Three things keep the pool from being swamped:

- identical questions that arrive while the first is still running share its result
  (single flight) instead of each paying for the reranker;
- at most `max_queue` searches wait for a worker; beyond that the service answers
  "overloaded" straight away instead of letting latency grow without bound;
- cached results never touch the pool at all.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from ragsvc.cache import QueryCache
from ragsvc.config import RETRIEVAL_MODES, Settings
from ragsvc.generate import Generator, GenResult, parse_citations
from ragsvc.search import Hit, Searcher


class Overloaded(RuntimeError):
    """Too many searches are already waiting for a model worker."""


@dataclass
class SearchOutcome:
    hits: list[Hit]
    timings_ms: dict[str, float]
    cache: str = "miss"  # miss | exact | semantic | shared
    similarity: float | None = None
    matched_query: str | None = None


@dataclass
class Metrics:
    counters: dict[str, int] = field(default_factory=dict)

    def inc(self, name: str, by: int = 1) -> None:
        self.counters[name] = self.counters.get(name, 0) + by


def passages_payload(hits: list[Hit]) -> list[dict]:
    return [{"n": i, "chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id,
             "section": h.chunk.section, "source": h.chunk.source, "page": h.chunk.page,
             "score": round(h.score, 4), "text": h.chunk.text}
            for i, h in enumerate(hits, start=1)]


class RagService:
    def __init__(self, searcher: Searcher, generator: Generator, settings: Settings):
        self.searcher, self.generator, self.settings = searcher, generator, settings
        c = settings.cache
        threshold = c.semantic_threshold if c.semantic else None
        self.search_cache = QueryCache(c.max_entries, c.ttl_s, threshold)
        self.answer_cache = QueryCache(c.max_entries, c.ttl_s, threshold)
        self.metrics = Metrics()
        self._pool = ThreadPoolExecutor(settings.server.model_workers,
                                        thread_name_prefix="ragsvc-model")
        self._waiting = 0
        self._inflight: dict[tuple, asyncio.Future] = {}

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)

    # -- helpers -------------------------------------------------------------------------
    def resolve(self, mode: str | None, k: int | None) -> tuple[str, int]:
        r = self.settings.retrieval
        mode = r.mode if mode is None else mode
        k = r.top_k if k is None else k
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {RETRIEVAL_MODES}")
        if not 1 <= k <= r.candidates:
            raise ValueError(f"k must be between 1 and {r.candidates}")
        return mode, k

    @property
    def queue_depth(self) -> int:
        return self._waiting

    async def _in_pool(self, fn, *args):
        if self._waiting >= self.settings.server.model_workers + self.settings.server.max_queue:
            self.metrics.inc("shed")
            raise Overloaded("search queue is full")
        self._waiting += 1
        try:
            return await asyncio.get_running_loop().run_in_executor(self._pool, fn, *args)
        finally:
            self._waiting -= 1

    def _embed(self, query: str):
        return self.searcher.embedder.encode_queries([query])[0]

    # -- search --------------------------------------------------------------------------
    async def search(self, query: str, mode: str | None = None,
                     k: int | None = None) -> SearchOutcome:
        mode, k = self.resolve(mode, k)
        scope = (mode, k)
        t0 = time.perf_counter()
        cfg = self.settings.cache
        self.metrics.inc("search_requests")

        def done(hits, timings, cache, hit=None) -> SearchOutcome:
            timings = {**timings, "service_total": (time.perf_counter() - t0) * 1e3}
            self.metrics.inc(f"search_cache_{cache}")
            return SearchOutcome(hits, timings, cache,
                                 hit.similarity if hit else None,
                                 hit.matched_query if hit else None)

        use_vec = mode != "bm25" and self.searcher.embedder is not None
        if cfg.exact:
            # a miss is counted once: here, or by the semantic lookup that follows
            hit = self.search_cache.get_exact(scope, query,
                                              count_miss=not (cfg.semantic and use_vec))
            if hit:
                return done(hit.value, {}, "exact", hit)
        key = (scope, query)
        shared = self._inflight.get(key)
        if shared is not None:
            hits, timings = await asyncio.shield(shared)
            return done(hits, timings, "shared")
        future: asyncio.Future = asyncio.get_running_loop().create_future()
        self._inflight[key] = future
        try:
            vec, timings = None, {}
            if cfg.semantic and use_vec:
                te = time.perf_counter()
                vec = await self._in_pool(self._embed, query)
                timings["embed"] = (time.perf_counter() - te) * 1e3
                hit = self.search_cache.get_semantic(scope, vec)
                if hit:
                    future.set_result((hit.value, timings))
                    return done(hit.value, timings, "semantic", hit)
            result = await self._in_pool(self.searcher.search, query, mode, k, vec)
            timings = {**timings, **{n: v for n, v in result.timings_ms.items() if n != "total"}}
            if cfg.exact or cfg.semantic:
                self.search_cache.put(scope, query, result.hits, vec)
            future.set_result((result.hits, timings))
            return done(result.hits, timings, "miss")
        except BaseException as exc:
            if not future.done():
                future.set_exception(exc)
                future.exception()  # mark as retrieved: waiters re-raise it themselves
            raise
        finally:
            self._inflight.pop(key, None)

    # -- ask -----------------------------------------------------------------------------
    async def ask(self, query: str, mode: str | None = None,
                  k: int | None = None) -> AsyncIterator[dict]:
        """Events: passages, then token*, then done (or refusal). Dicts with a `type`."""
        mode, k = self.resolve(mode, k)
        scope = (mode, k, self.generator.name)
        t0 = time.perf_counter()
        cfg = self.settings.cache
        self.metrics.inc("ask_requests")
        cached = self.answer_cache.get_exact(scope, query) if cfg.exact else None
        found: SearchOutcome | None = None
        if cached is None:
            found = await self.search(query, mode, k)
            # a paraphrase that reused another question's passages may reuse its answer too
            if found.cache == "semantic" and found.matched_query:
                cached = self.answer_cache.get_exact(scope, found.matched_query, count_miss=False)
                if cached:
                    cached.kind, cached.similarity = "semantic", found.similarity or 0.0
        if cached is not None:
            hits, gen = cached.value
            self.metrics.inc(f"answer_cache_{cached.kind}")
            yield {"type": "passages", "passages": passages_payload(hits), "cache": cached.kind}
            yield {"type": "token", "text": gen.text}
            yield self._done(gen, hits, cached.kind, t0, {}, cached.similarity)
            return
        self.metrics.inc("answer_cache_miss")
        yield {"type": "passages", "passages": passages_payload(found.hits),
               "cache": found.cache}
        gen: GenResult | None = None
        async for item in self.generator.stream(query, found.hits):
            if isinstance(item, GenResult):
                gen = item
            else:
                yield {"type": "token", "text": item}
        if gen is None:  # pragma: no cover - a generator must end with a GenResult
            raise RuntimeError("generator ended without a result")
        if gen.refused:
            self.metrics.inc("refusals")
            yield {"type": "refusal", "message": "The model declined to answer this question. "
                   "Discard any partial text."}
            return
        if gen.stop_reason == "end_turn" and (cfg.exact or cfg.semantic):
            self.answer_cache.put(scope, query, (found.hits, gen))  # never cache a cut-off
        yield self._done(gen, found.hits, "miss", t0, found.timings_ms, None)

    def _done(self, gen: GenResult, hits: list[Hit], cache: str, t0: float,
              retrieval_ms: dict, similarity: float | None) -> dict:
        citations = parse_citations(gen.text)
        fresh = cache == "miss"
        return {
            "type": "done", "answer": gen.text, "citations": citations,
            "invalid_citations": [n for n in citations if not 1 <= n <= len(hits)],
            "abstained": gen.abstained, "model": gen.model, "stop_reason": gen.stop_reason,
            "cache": cache, "similarity": similarity,
            # a cached answer costs nothing now; the original cost is kept for reference
            "usage": {"input_tokens": gen.input_tokens if fresh else 0,
                      "output_tokens": gen.output_tokens if fresh else 0,
                      "cost_usd": gen.cost_usd if fresh else 0.0,
                      "saved_cost_usd": 0.0 if fresh else gen.cost_usd},
            "timings_ms": {"retrieval": {n: round(v, 2) for n, v in retrieval_ms.items()},
                           "ttft": gen.ttft_ms if fresh else None,
                           "generation": round(gen.latency_ms, 1) if fresh else 0.0,
                           "total": round((time.perf_counter() - t0) * 1e3, 1)},
        }

    def stats(self) -> dict:
        return {"counters": dict(sorted(self.metrics.counters.items())),
                "search_cache": {**self.search_cache.stats.as_dict(),
                                 "entries": len(self.search_cache)},
                "answer_cache": {**self.answer_cache.stats.as_dict(),
                                 "entries": len(self.answer_cache)},
                "queue_depth": self.queue_depth,
                "generator": self.generator.name}
