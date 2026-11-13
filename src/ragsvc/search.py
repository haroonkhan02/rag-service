"""Search over an index bundle: exact dense, BM25, reciprocal-rank fusion and reranking.

Same four modes and the same tokenizer as robotics-docs-rag, with one change: dense search
is an exact matrix product over the bundle's normalised embeddings instead of a Chroma HNSW
query. Exact search returns the true nearest neighbours, needs no database process, and at
3,680 x 384 it takes well under a millisecond.

    bm25           lexical only (exact words: `ros2 topic echo`, `cmd_vel`, flags)
    dense          embeddings only (paraphrases and concepts)
    hybrid         both rankings merged with RRF
    hybrid_rerank  hybrid candidates re-scored by a cross-encoder
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from rank_bm25 import BM25Okapi

from ragsvc.bundle import IndexBundle
from ragsvc.chunking import Chunk
from ragsvc.config import RETRIEVAL_MODES, RetrievalSettings

TOKEN_RE = re.compile(r"[a-z0-9_]+")
STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does", "for", "from", "how",
    "i", "if", "in", "is", "it", "its", "of", "on", "or", "that", "the", "this", "to", "was",
    "what", "when", "where", "which", "who", "why", "will", "with", "you", "your",
})


def tokenize(text: str) -> list[str]:
    """Lower-case word tokens; `cmd_vel` is kept whole *and* split, so both forms match."""
    out: list[str] = []
    for tok in TOKEN_RE.findall(text.lower()):
        if tok in STOPWORDS:
            continue
        out.append(tok)
        if "_" in tok:
            out += [p for p in tok.split("_") if p and p not in STOPWORDS]
    return out


def top_n(scores: np.ndarray, n: int) -> np.ndarray:
    """Indices of the n highest scores, best first; ties broken by lower index.

    A stable full sort rather than argpartition: argpartition picks an arbitrary subset of
    the items tied at the cut-off, so results could change between runs and machines. At a
    few thousand rows the sort costs well under a millisecond.
    """
    if n <= 0:
        return np.empty(0, dtype=np.int64)
    return np.argsort(-scores, kind="stable")[:n]


def rrf_fuse(rankings: Sequence[Sequence[str]], k: int = 60) -> list[tuple[str, float]]:
    """score(d) = sum over rankings of 1 / (k + rank(d)); ranks only, so no score scaling."""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, cid in enumerate(ranking, start=1):
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass
class Hit:
    chunk: Chunk
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # rank in each component retriever


@dataclass
class SearchResult:
    hits: list[Hit]
    timings_ms: dict[str, float]


class Searcher:
    def __init__(self, bundle: IndexBundle, settings: RetrievalSettings | None = None,
                 embedder=None, reranker=None):
        self.bundle = bundle
        self.settings = settings or RetrievalSettings()
        self.embedder, self.reranker = embedder, reranker
        self.chunks = bundle.chunks
        self.ids = [c.chunk_id for c in self.chunks]
        self.by_id = {c.chunk_id: c for c in self.chunks}
        self.matrix = np.ascontiguousarray(bundle.embeddings, dtype=np.float32)
        self.bm25 = BM25Okapi([tokenize(c.text) for c in self.chunks])

    def bm25_search(self, query: str, n: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        return [(self.ids[i], float(scores[i])) for i in top_n(scores, n) if scores[i] > 0]

    def dense_search(self, query_vec: np.ndarray, n: int) -> list[tuple[str, float]]:
        scores = self.matrix @ np.asarray(query_vec, dtype=np.float32)
        return [(self.ids[i], float(scores[i])) for i in top_n(scores, n)]

    def search(self, query: str, mode: str | None = None, k: int | None = None) -> SearchResult:
        s = self.settings
        mode = s.mode if mode is None else mode
        k = s.top_k if k is None else k
        if mode not in RETRIEVAL_MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {RETRIEVAL_MODES}")
        if k < 1:
            raise ValueError("k must be >= 1")
        if mode != "bm25" and self.embedder is None:
            raise RuntimeError(f"mode {mode!r} needs an embedder")
        if mode == "hybrid_rerank" and self.reranker is None:
            raise RuntimeError("hybrid_rerank needs a reranker")
        t: dict[str, float] = {}
        n = max(s.candidates, k)
        ranks: dict[str, dict[str, int]] = {}
        bm: list[tuple[str, float]] = []
        de: list[tuple[str, float]] = []
        if mode in ("bm25", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            bm = self.bm25_search(query, n)
            t["bm25"] = (time.perf_counter() - t0) * 1e3
            for r, (cid, _) in enumerate(bm, 1):
                ranks.setdefault(cid, {})["bm25"] = r
        if mode in ("dense", "hybrid", "hybrid_rerank"):
            t0 = time.perf_counter()
            q = self.embedder.encode_queries([query])[0]
            t["embed"] = (time.perf_counter() - t0) * 1e3
            t0 = time.perf_counter()
            de = self.dense_search(q, n)
            t["dense"] = (time.perf_counter() - t0) * 1e3
            for r, (cid, _) in enumerate(de, 1):
                ranks.setdefault(cid, {})["dense"] = r
        if mode == "bm25":
            ranked = bm
        elif mode == "dense":
            ranked = de
        else:
            ranked = rrf_fuse([[c for c, _ in bm], [c for c, _ in de]], s.rrf_k)
        if mode == "hybrid_rerank":
            t0 = time.perf_counter()
            pool = [cid for cid, _ in ranked[: s.rerank_top]]
            scores = self.reranker.score(query, [self.by_id[c].text for c in pool])
            ranked = sorted(zip(pool, scores, strict=True), key=lambda kv: -kv[1])
            t["rerank"] = (time.perf_counter() - t0) * 1e3
        hits = [Hit(self.by_id[cid], score, ranks.get(cid, {})) for cid, score in ranked[:k]]
        t["total"] = sum(t.values())
        return SearchResult(hits, t)
