import numpy as np
import pytest

from ragsvc.build_index import build_bundle
from ragsvc.config import RetrievalSettings, Settings
from ragsvc.search import Searcher, rrf_fuse, tokenize, top_n
from tests.conftest import LengthReranker


@pytest.fixture
def searcher(docs, embedder):
    bundle = build_bundle(Settings(), docs, embedder)
    return Searcher(bundle, RetrievalSettings(candidates=10, top_k=3), embedder, LengthReranker())


def test_tokenize_keeps_and_splits_snake_case():
    assert tokenize("How do I echo cmd_vel?") == ["echo", "cmd_vel", "cmd", "vel"]


def test_top_n_matches_full_sort_with_ties():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=200).astype(np.float32)  # many ties
    expected = sorted(range(200), key=lambda i: (-scores[i], i))[:17]
    assert top_n(scores, 17).tolist() == expected
    assert top_n(scores, 500).shape == (200,)
    assert top_n(scores, 0).size == 0


def test_rrf_rewards_agreement():
    fused = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    assert [c for c, _ in fused][:2] == ["a", "b"]  # tie on score -> id order
    assert fused[0][1] == pytest.approx(1 / 61 + 1 / 62)
    assert {c for c, _ in fused} == {"a", "b", "c", "d"}


def test_dense_search_is_exact(searcher, embedder):
    q = embedder.encode_queries(["late joining subscriptions durability"])[0]
    got = searcher.dense_search(q, 5)
    brute = sorted(((float(v @ q), c.chunk_id) for v, c in
                    zip(searcher.matrix, searcher.chunks, strict=True)), key=lambda x: -x[0])
    assert [cid for cid, _ in got] == [cid for _, cid in brute[:5]]


def test_modes_return_expected_documents(searcher):
    for mode in ("bm25", "dense", "hybrid"):
        res = searcher.search("transient local durability late joining", mode, 3)
        assert res.hits[0].chunk.doc_id == "Concepts/Topics", mode
        assert res.timings_ms["total"] >= 0
    res = searcher.search("ros2 service call", "hybrid", 3)
    assert res.hits[0].chunk.doc_id == "Tutorials/Services"
    assert set(res.hits[0].ranks) == {"bm25", "dense"}


def test_rerank_reorders_candidates(searcher):
    res = searcher.search("topics", "hybrid_rerank", 3)
    lengths = [len(h.chunk.text) for h in res.hits]
    assert lengths == sorted(lengths)  # the fake reranker prefers short passages
    assert "rerank" in res.timings_ms


def test_bm25_drops_zero_scores_and_bad_inputs(searcher):
    assert searcher.search("zzzz qqqq", "bm25", 3).hits == []
    with pytest.raises(ValueError, match="mode"):
        searcher.search("x", "semantic")
    with pytest.raises(ValueError, match="k must"):
        searcher.search("x", "bm25", 0)


def test_missing_models_fail_clearly(docs, embedder):
    bundle = build_bundle(Settings(), docs, embedder)
    bare = Searcher(bundle)
    assert bare.search("bag record", "bm25").hits
    with pytest.raises(RuntimeError, match="embedder"):
        bare.search("x", "dense")
    with pytest.raises(RuntimeError, match="reranker"):
        Searcher(bundle, embedder=embedder).search("x", "hybrid_rerank")
