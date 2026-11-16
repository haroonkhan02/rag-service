"""Cost arithmetic, usage totals and the workload replay, on the synthetic corpus."""

import asyncio
import dataclasses

import pytest

from ragsvc.config import CacheSettings, Settings
from ragsvc.cost import PRICES, cost_usd
from ragsvc.cost_study import (
    RecordedGenerator,
    context_rows,
    markdown,
    price_rows,
    replay,
    token_profile,
    workload_rows,
)
from ragsvc.evalset import Question
from ragsvc.service import RagService

RECORDS = [
    {"question": "How do I record topics with ros2 bag?", "answer": "Use ros2 bag record [1].",
     "input_tokens": 1000, "output_tokens": 40, "generation_ms": 2000.0, "mode": "hybrid",
     "k": 2, "model": "local"},
    {"question": "What does transient local durability do?", "answer": "It keeps samples [1].",
     "input_tokens": 800, "output_tokens": 20, "generation_ms": 1000.0, "mode": "hybrid",
     "k": 2, "model": "local"},
]
PARAPHRASE = "How can I record topics using ros2 bag?"


def test_cost_is_tokens_times_list_price():
    assert cost_usd("claude-opus-5-5", 1_000_000, 100_000) == pytest.approx(6.0)
    assert cost_usd("claude-haiku-5-5", 2000, 1000) == pytest.approx(0.0007)
    assert cost_usd("some-local-model", 10**9, 10**9) == 0.0


def test_token_profile_and_price_rows():
    p = token_profile(RECORDS)
    assert p["input_tokens"]["mean"] == 900 and p["output_tokens"]["max"] == 40
    assert p["generation_s_mean"] == 1.5
    rows = {r["model"]: r for r in price_rows(900, 30)}
    opus = rows["claude-opus-5-5"]
    assert opus["usd_per_answer"] == pytest.approx(900 * 4e-6 + 30 * 20e-6)
    assert opus["usd_per_1000"] == pytest.approx(1000 * opus["usd_per_answer"])
    assert opus["input_share"] == pytest.approx(0.0036 / 0.0042)
    assert set(rows) == set(PRICES)


def test_usage_totals_count_fresh_answers_and_cached_ones(searcher):
    gen = RecordedGenerator(RECORDS)

    async def run():
        svc = RagService(searcher, gen, Settings())
        for q in (RECORDS[0]["question"], RECORDS[0]["question"], RECORDS[1]["question"]):
            async for _ in svc.ask(q, "hybrid", 2):
                pass
        return svc.stats()["usage"]

    usage = asyncio.run(run())
    assert usage["answers_generated"] == 2 and usage["answers_from_cache"] == 1
    assert usage["input_tokens"] == 1800 and usage["output_tokens"] == 60


def test_replay_charges_a_paraphrase_like_its_original(searcher):
    gen = RecordedGenerator(RECORDS, {PARAPHRASE: RECORDS[0]["question"]})
    workload = [RECORDS[0]["question"], PARAPHRASE, RECORDS[0]["question"], PARAPHRASE]
    off = dataclasses.replace(Settings(), cache=CacheSettings(exact=False, semantic=False))
    row = asyncio.run(replay(searcher, gen, off, workload, "hybrid"))
    assert row["answers_generated"] == 4 and row["input_tokens"] == 4000
    rows = workload_rows(searcher, gen, Settings(), workload, "hybrid")
    assert [r["answers_generated"] for r in rows][:2] == [4, 2]
    assert rows[2]["answers_generated"] + rows[2]["answers_from_cache"] == 4
    assert rows[1]["usd"]["claude-opus-5-5"] == pytest.approx(cost_usd("claude-opus-5-5",
                                                                       2000, 80))
    with pytest.raises(KeyError):
        asyncio.run(replay(searcher, gen, off, ["a question nobody recorded"], "hybrid"))


def test_context_rows_grow_with_k_and_report_the_hit_rate(searcher):
    questions = [Question("q1", "howto", RECORDS[0]["question"], ["Tutorials/Bags"],
                          ["ros2 bag record"]),
                 Question("u1", "unanswerable", "What is the baud rate?")]
    rows = context_rows(searcher, questions, lambda q, hits: 10 + 100 * len(hits), "hybrid",
                        ks=(1, 3))
    assert [r["input_tokens_mean"] for r in rows] == [110, 310]
    assert rows[0]["n"] == 1 and rows[1]["hit_rate"] == 1.0


def test_report_renders(searcher):
    gen = RecordedGenerator(RECORDS)
    profile = token_profile(RECORDS)
    rep = {"model": "local", "mode": "hybrid", "k": 2, "tokens": profile,
           "prices": price_rows(900, 30),
           "context": [{"k": 1, "input_tokens_mean": 500.0, "hit_rate": 0.5, "n": 2},
                       {"k": 5, "input_tokens_mean": 1000.0, "hit_rate": 1.0, "n": 2}],
           "tokenizer_check": {"n": 2, "max_abs_diff": 0},
           "workload": {"n": 2, "n_questions": 2, "paraphrase_share": 0.3,
                        "rows": workload_rows(searcher, gen, Settings(),
                                              [r["question"] for r in RECORDS], "hybrid")}}
    text = markdown(rep)
    assert "| 1 | 500 | 0.50x | 50.0 % |" in text and "| no cache | 2 | 0 |" in text
    assert "2,400 answers per hour" in text
