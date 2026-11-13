import json

import pytest

from ragsvc.build_index import build_bundle
from ragsvc.config import RetrievalSettings, Settings
from ragsvc.evalset import Question, hit_at_k, is_relevant, mrr_at_k, ndcg_at_k
from ragsvc.parity import aggregate, compare, markdown, reference_rows, score_questions
from ragsvc.search import Searcher


def test_rank_metrics():
    rels = [False, True, False, True]
    assert hit_at_k(rels, 1) == 0.0 and hit_at_k(rels, 2) == 1.0
    assert mrr_at_k(rels, 10) == 0.5 and mrr_at_k(rels, 1) == 0.0
    assert ndcg_at_k([True, False], 10, 1) == 1.0
    assert ndcg_at_k(rels, 10, 2) == pytest.approx((1 / 1.585 + 1 / 2.322) / (1 + 1 / 1.585),
                                                   rel=1e-3)
    assert ndcg_at_k([False], 10, 0) == 0.0


def test_relevance_needs_gold_doc_and_answer_key(chunks):
    q = Question("q", "concept", "?", ["Concepts/Topics"], ["late-joining  SUBSCRIPTIONS"])
    rel = [c for c in chunks if is_relevant(c, q)]
    assert len(rel) == 1 and rel[0].section == "Topics > Durability"
    other = Question("q", "concept", "?", ["Tutorials/Bags"], ["late-joining subscriptions"])
    assert not any(is_relevant(c, other) for c in chunks)


def test_score_and_compare_against_reference(docs, embedder, tmp_path):
    bundle = build_bundle(Settings(), docs, embedder)
    searcher = Searcher(bundle, RetrievalSettings(candidates=10), embedder)
    qs = [Question("q1", "c", "late joining durability", ["Concepts/Topics"],
                   ["late-joining subscriptions"]),
          Question("q2", "c", "record a bag", ["Tutorials/Bags"], ["ros2 bag record"])]
    per_q = score_questions(searcher, qs, "dense")
    agg = aggregate(per_q)
    assert agg["hit@10"] == 1.0 and set(agg) == {"hit@1", "hit@5", "hit@10", "mrr@10",
                                                 "ndcg@10"}
    ref_file = tmp_path / "ref.json"
    ref_file.write_text(json.dumps({"results": [{
        "strategy": "section", "mode": "dense",
        **{m: {"mean": v} for m, v in agg.items()},
        "per_question": [{"id": p["id"], "rels": p["rels"], "top_docs": p["top_docs"]}
                         for p in per_q[:1]]
        + [{"id": "q2", "rels": [False] * 10, "top_docs": []}],
    }]}))
    ref = reference_rows(ref_file, "section")["dense"]
    assert compare(per_q, ref) == {"same_relevance_top10": 1, "same_top5_docs": 1}
    report = {"n_questions": 2, "n_chunks": len(bundle.chunks), "machine": "x", "device": "cpu",
              "modes": {"dense": {"service": agg, "reference": ref["metrics"],
                                  **compare(per_q, ref), "latency_ms": {}}},
              "chunks_identical_to_reference": True}
    md = markdown(report)
    assert "| dense |" in md and "1/2" in md and "byte-identical to the reference index: yes" in md
