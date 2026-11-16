"""Answer evaluation on the synthetic corpus with a scripted judge: no model, no network."""

import asyncio
import dataclasses
import types

import pytest

from ragsvc import answer_eval as ae
from ragsvc.answer_report import markdown
from ragsvc.config import CacheSettings, Settings
from ragsvc.evalset import Question
from ragsvc.generate import ABSTAIN, ExtractiveGenerator, GenResult
from ragsvc.service import RagService

QUESTIONS = [
    Question("q1", "howto", "How do I record topics with ros2 bag?", ["Tutorials/Bags"],
             ["ros2 bag record"], "Use ros2 bag record."),
    Question("q2", "concept", "What does transient local durability do?", ["Concepts/Topics"],
             ["late-joining"], "It keeps samples for late-joining subscriptions."),
    Question("u1", "unanswerable", "What is the baud rate of the lidar?", [], [], "Not covered."),
]
BY_ID = {q.id: q for q in QUESTIONS}


class Scripted:
    """Abstains on questions about a baud rate, otherwise quotes the top passage."""

    name = "scripted"

    async def stream(self, question, hits):
        if "baud" in question:
            yield ABSTAIN
            yield GenResult(ABSTAIN, self.name, input_tokens=50, output_tokens=12)
            return
        async for item in ExtractiveGenerator().stream(question, hits):
            if isinstance(item, GenResult):
                item = dataclasses.replace(item, input_tokens=100, output_tokens=20)
            yield item


class FakeJudge:
    def __init__(self, reply):
        self.reply, self.calls = reply, []

    def complete(self, system, user):
        self.calls.append(user)
        return types.SimpleNamespace(text=self.reply(user))


def records(searcher, mode="hybrid"):
    cfg = dataclasses.replace(Settings(), cache=CacheSettings(exact=False, semantic=False))
    svc = RagService(searcher, Scripted(), cfg)
    return asyncio.run(ae.generate(svc, QUESTIONS, mode, 2))


@pytest.mark.parametrize("text, expected", [
    ("Same fact.\n[[CORRECT]]", True),
    ("It says [[CORRECT]] at first, but no.\n[[INCORRECT]]", False),
    ("[[correct]]", True),
    ("I think it is correct.", None),
])
def test_parse_verdict_takes_the_last_tag(text, expected):
    assert ae.parse_verdict(text) is expected


def test_generate_records_citations_relevance_and_usage(searcher):
    recs = {r["id"]: r for r in records(searcher)}
    q1 = recs["q1"]
    assert q1["retrieval_hit"] and q1["cited_relevant"] and q1["key_in_answer"]
    assert q1["citations"] == [1] and q1["invalid_citations"] == []
    assert len(q1["passages"]) == 2 and q1["relevant"][0] is True
    assert (q1["input_tokens"], q1["output_tokens"]) == (100, 20)
    assert recs["u1"]["abstained"] and not recs["u1"]["retrieval_hit"]


def test_only_given_answers_to_answerable_questions_reach_the_judge(searcher):
    items = ae.answer_items(records(searcher), BY_ID)
    assert [i["id"] for i in items] == ["q1", "q2"]
    assert "[1]" not in items[0]["candidate"] and items[0]["reference"] == "Use ros2 bag record."
    judge = FakeJudge(lambda user: "ok [[CORRECT]]" if "bag" in user else "no tag")
    judged = ae.judge_items(judge, items)
    assert [j["verdict"] for j in judged] == [True, None]
    assert "Reference answer: Use ros2 bag record." in judge.calls[0]


def test_scoring_rules(searcher):
    recs = records(searcher)
    rows = {r["id"]: r for r in ae.score(recs, {"q1": True, "q2": None}, BY_ID)}
    assert rows["q1"]["correct"] == 1
    assert rows["q2"]["correct"] == 0 and rows["q2"]["unparsed"]  # no verdict is not a pass
    assert rows["u1"]["correct"] == 1  # abstaining on an unanswerable question is right
    wrong = [dict(r, abstained=r["id"] == "q1") for r in recs]
    rows = {r["id"]: r for r in ae.score(wrong, {"q1": True, "q2": True}, BY_ID)}
    assert rows["q1"]["correct"] == 0  # abstained although answerable: the verdict is ignored
    assert rows["u1"]["correct"] == 0  # answered an unanswerable question
    s = ae.summarize(list(rows.values()))
    assert s["overall"] == {"n": 3, "count": 1, "share": 0.3333}
    assert s["answerable"]["abstained"]["count"] == 1
    assert s["unanswerable"]["abstained"]["count"] == 0
    assert s["by_category"]["howto"]["count"] == 0


def test_controls_have_known_verdicts_and_never_pair_a_question_with_itself():
    items = ae.control_items(QUESTIONS)
    assert len(items) == 6  # three per answerable question
    for it in items:
        if it["kind"] == "same":
            assert it["candidate"] == it["reference"] and it["expected"] is True
        else:
            assert it["candidate"] != it["reference"] and it["expected"] is False
    judge = FakeJudge(lambda user: "[[CORRECT]]")  # a judge that always says yes
    summary = ae.control_summary(ae.judge_items(judge, items))
    assert summary["same"]["as_expected"] == 2 and summary["other"]["as_expected"] == 0
    assert summary["abstain"] == {"n": 2, "as_expected": 0, "unparsed": 0}


def test_judge_agreement_counts():
    a = {"q1": True, "q2": False, "q3": None, "q4": True}
    b = {"q1": True, "q2": True, "q3": False, "q4": False}
    assert ae.judge_agreement(a, b) == {"n": 4, "agree": 2, "both_correct": 1,
                                        "both_incorrect": 1, "only_first": 1, "only_second": 1}


def test_report_end_to_end_with_the_gate(searcher, tmp_path, monkeypatch):
    pytest.importorskip("llmeval")
    out, evals = tmp_path / "reports", tmp_path / "evals"
    monkeypatch.setattr(ae, "load_settings", lambda path=None: Settings())
    monkeypatch.setattr(ae, "load_questions", lambda path: QUESTIONS)
    for name, mode, verdicts in [("base", "hybrid", (True, True)), ("cand", "bm25", (True, False))]:
        recs = records(searcher, mode)
        ae.write_jsonl(out / "answers" / f"{name}.jsonl", recs)
        ae.write_jsonl(out / "answers" / f"{name}.judge-j.jsonl",
                       [{"id": i, "verdict": v, "text": ""}
                        for i, v in zip(("q1", "q2"), verdicts, strict=True)])
    ae.write_jsonl(out / "answers" / "controls.judge-j.jsonl",
                   ae.judge_items(FakeJudge(lambda u: "[[INCORRECT]]"),
                                  ae.control_items(QUESTIONS)))
    ae.main(["--out", str(out), "report", "base", "cand", "--judges", "j",
             "--evals", str(evals)])
    assert (evals / "answers_cand.csv").read_text() == "id,score\nq1,1\nq2,0\nu1,1\n"
    text = (out / "answer_eval.md").read_text()
    assert "## Regression gate" in text and "| base | cand | 3 |" in text
    assert "| j | 0/2 | 2/2 | 2/2 | 0 |" in text


def test_markdown_without_gate_or_second_judge():
    rows = ae.score([{"id": "q1", "category": "howto", "abstained": False, "citations": [1]}],
                    {"q1": True}, BY_ID)
    rep = {"judges": ["j"], "primary_judge": "j", "controls": {}, "agreement": {},
           "runs": {"r": {"mode": "hybrid", "k": 5, "model": "m", "n": 1,
                          **ae.summarize(rows), "by_judge": {}}}}
    text = markdown(rep)
    assert "| hybrid | 100.0 % (1/1) |" in text and "Regression gate" not in text
