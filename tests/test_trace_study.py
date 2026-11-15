"""The trace study on the synthetic corpus: grouping, the breakdown table, the tree."""

import asyncio

import pytest

pytest.importorskip("opentelemetry.sdk")

from ragsvc.config import Settings  # noqa: E402
from ragsvc.generate import ExtractiveGenerator  # noqa: E402
from ragsvc.trace_study import markdown, run_ask, run_breakdown, run_overhead  # noqa: E402

QUESTIONS = ["How do I record topics with ros2 bag?", "What is transient local durability?",
             "How do I call a service from the CLI?"]
PARAPHRASES = ["How do I record topics with ros2 bag please?", "Which robots drove 37.4 miles?"]


def test_breakdown_groups_requests_by_cache_outcome(searcher):
    rows = asyncio.run(run_breakdown(searcher, ExtractiveGenerator(), Settings(), QUESTIONS,
                                     PARAPHRASES))
    by = {r["cache"]: r for r in rows}
    assert by["miss"]["n"] + by.get("semantic", {"n": 0})["n"] == 5 and by["exact"]["n"] == 3
    assert by["miss"]["rag.retrieve"] > 0 and by["miss"]["rag.rerank_ms"] is not None
    assert by["exact"]["rag.retrieve"] is None and by["exact"]["rag.embed"] is None
    assert by["miss"]["request_p50"] >= by["miss"]["rag.search"]


def test_ask_summary_overhead_rows_and_report(searcher):
    gen = ExtractiveGenerator()
    ask, tree = asyncio.run(run_ask(searcher, gen, Settings(), QUESTIONS))
    assert ask["n"] == 3 and ask["generator"] == "extractive"
    assert tree[0].startswith("POST /ask") and any("rag.generate" in line for line in tree)
    assert tree[1].startswith("  rag.ask")
    overhead = asyncio.run(run_overhead(searcher, gen, Settings(), QUESTIONS, rounds=2))
    assert [r["path"] for r in overhead][0] == "exact cache hit"
    assert overhead[0]["spans_per_request"] == 2 and overhead[2]["spans_per_request"] == 3
    assert all(r["requests_per_arm"] == 6 for r in overhead)
    text = markdown({"machine": "m", "device": "cpu", "n_questions": 3,
                     "semantic_threshold": 0.96, "rounds": 2,
                     "breakdown": asyncio.run(run_breakdown(searcher, gen, Settings(),
                                                            QUESTIONS, PARAPHRASES)),
                     "ask": ask, "example_trace": tree, "overhead": overhead})
    assert "## What tracing costs" in text and "| exact |" in text
