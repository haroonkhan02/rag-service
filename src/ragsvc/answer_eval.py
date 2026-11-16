"""`ragsvc-answer-eval`: are the answers right, and do the citations hold?

    generate   run every evaluation question through the service's own `ask` path with one
               retrieval mode and store the answers            -> reports/answers/<name>.jsonl
    judge      grade each answer against the reference answer with an LLM judge from
               llm-eval-harness, plus control items that check the judge itself
                                                   -> reports/answers/<name>.judge-<tag>.jsonl
    report     tables, and llm-eval-harness's regression gate between two runs
                                                   -> reports/answer_eval.md / .json, evals/

Two kinds of check. The first needs no model: did the answer abstain, does it cite, do the
cited numbers exist, is a cited passage actually a relevant one. The second is the judge's
verdict on whether the answer states the reference answer's facts. An unanswerable question
is correct when the service abstains; an abstention on an answerable one is wrong.

`generate` and `judge` are separate commands because each loads one local model, and two
7B models do not fit in memory together.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import random
import re
from pathlib import Path

from ragsvc.config import CacheSettings, load_settings
from ragsvc.evalset import Question, is_relevant, load_questions, norm
from ragsvc.generate import ABSTAIN

JUDGE_SYSTEM = """You grade answers to questions about ROS 2 (Robot Operating System 2).

You get a question, a reference answer written by the person who wrote the question, and a \
candidate answer. Decide whether the candidate states the key fact or facts of the reference.

- Extra detail that does not contradict the reference is fine.
- Ignore wording, length, formatting and citation markers such as [1].
- The candidate is incorrect if it contradicts the reference, leaves out its key fact, \
answers a different question, or says it cannot answer.

Explain your decision in one or two sentences, then end with exactly [[CORRECT]] or \
[[INCORRECT]] on its own line."""

_VERDICT = re.compile(r"\[\[(CORRECT|INCORRECT)\]\]", re.IGNORECASE)
_CITE = re.compile(r"\s*\[\d{1,2}\]")


def judge_prompt(question: str, reference: str, candidate: str) -> str:
    return (f"Question: {question}\n\nReference answer: {reference}\n\n"
            f"Candidate answer: {candidate}")


def parse_verdict(text: str) -> bool | None:
    """The last [[CORRECT]] / [[INCORRECT]] tag; None when the judge gave neither."""
    tags = _VERDICT.findall(text)
    return tags[-1].upper() == "CORRECT" if tags else None


# -- generate --------------------------------------------------------------------------------
def answer_record(q: Question, mode: str, k: int, events: list[dict], chunks: dict) -> dict:
    passages = next(e["passages"] for e in events if e["type"] == "passages")
    final = events[-1]
    if final["type"] != "done":
        return {"id": q.id, "category": q.category, "question": q.question, "mode": mode,
                "k": k, "answer": "", "refused": True}
    rel = [is_relevant(chunks[p["chunk_id"]], q) for p in passages]
    cited = [n for n in final["citations"] if 1 <= n <= len(passages)]
    answer_norm = norm(_CITE.sub("", final["answer"]))
    return {
        "id": q.id, "category": q.category, "question": q.question, "mode": mode, "k": k,
        "answer": final["answer"], "abstained": final["abstained"],
        "citations": final["citations"], "invalid_citations": final["invalid_citations"],
        "passages": [p["chunk_id"] for p in passages], "relevant": rel,
        "retrieval_hit": any(rel), "cited_relevant": any(rel[n - 1] for n in cited),
        "key_in_answer": any(norm(key) in answer_norm for key in q.answer_keys),
        "model": final["model"], "stop_reason": final["stop_reason"],
        "input_tokens": final["usage"]["input_tokens"],
        "output_tokens": final["usage"]["output_tokens"],
        "ttft_ms": final["timings_ms"]["ttft"],
        "generation_ms": final["timings_ms"]["generation"],
    }


async def generate(service, questions: list[Question], mode: str, k: int,
                   progress=None) -> list[dict]:
    chunks = service.searcher.by_id
    out = []
    for i, q in enumerate(questions, 1):
        events = [ev async for ev in service.ask(q.question, mode, k)]
        out.append(answer_record(q, mode, k, events, chunks))
        if progress:
            progress(i, len(questions), out[-1])
    return out


# -- judge -----------------------------------------------------------------------------------
def control_items(questions: list[Question], seed: int = 0) -> list[dict]:
    """Items with a known verdict, to check the judge rather than the answers.

    same       the reference answer as the candidate            -> should be CORRECT
    other      another question's reference answer              -> should be INCORRECT
    abstain    the service's "cannot answer" sentence           -> should be INCORRECT
    """
    qs = [q for q in questions if q.answerable]
    shift = random.Random(seed).randrange(1, len(qs)) if len(qs) > 1 else 0
    items = []
    for i, q in enumerate(qs):
        other = qs[(i + shift) % len(qs)]
        items += [
            {"id": f"{q.id}:same", "kind": "same", "expected": True, "question": q.question,
             "reference": q.reference_answer, "candidate": q.reference_answer},
            {"id": f"{q.id}:other", "kind": "other", "expected": False, "question": q.question,
             "reference": q.reference_answer, "candidate": other.reference_answer},
            {"id": f"{q.id}:abstain", "kind": "abstain", "expected": False,
             "question": q.question, "reference": q.reference_answer, "candidate": ABSTAIN},
        ]
    return items


def judge_items(backend, items: list[dict], progress=None) -> list[dict]:
    out = []
    for i, it in enumerate(items, 1):
        done = backend.complete(JUDGE_SYSTEM,
                                judge_prompt(it["question"], it["reference"], it["candidate"]))
        out.append({"id": it["id"], "verdict": parse_verdict(done.text), "text": done.text,
                    **{n: it[n] for n in ("kind", "expected") if n in it}})
        if progress:
            progress(i, len(items), out[-1])
    return out


def answer_items(records: list[dict], questions: dict[str, Question]) -> list[dict]:
    """What the judge has to look at: non-abstained answers to answerable questions."""
    return [{"id": r["id"], "question": r["question"], "candidate": _CITE.sub("", r["answer"]),
             "reference": questions[r["id"]].reference_answer}
            for r in records
            if questions[r["id"]].answerable and not r.get("refused") and not r["abstained"]]


def make_backend(spec: str, max_tokens: int = 300):
    """`mlx:<hf repo>`, `anthropic:<model>` (requires an Anthropic API key)."""
    from llmeval.backends import AnthropicBackend, MLXBackend

    kind, _, model = spec.partition(":")
    if kind == "mlx":
        return MLXBackend(model, max_tokens=max_tokens)
    if kind == "anthropic":
        return AnthropicBackend(model or "claude-opus-5-5")
    raise ValueError(f"unknown judge backend {spec!r}")


# -- score -----------------------------------------------------------------------------------
def score(records: list[dict], verdicts: dict[str, bool | None],
          questions: dict[str, Question]) -> list[dict]:
    """One row per question with a 0/1 `correct`. An unparsed verdict counts as wrong."""
    rows = []
    for r in records:
        q = questions[r["id"]]
        abstained = bool(r.get("abstained"))
        if not q.answerable:
            correct = abstained
        elif abstained or r.get("refused"):
            correct = False
        else:
            correct = verdicts.get(r["id"]) is True
        rows.append({"id": r["id"], "category": r["category"], "answerable": q.answerable,
                     "abstained": abstained, "correct": int(correct),
                     "unparsed": q.answerable and not abstained and verdicts.get(r["id"]) is None,
                     "retrieval_hit": r.get("retrieval_hit", False),
                     "cited_relevant": r.get("cited_relevant", False),
                     "has_citation": bool(r.get("citations")),
                     "invalid_citation": bool(r.get("invalid_citations")),
                     "key_in_answer": r.get("key_in_answer", False),
                     "truncated": r.get("stop_reason") == "max_tokens"})
    return rows


def share(rows: list[dict], field: str) -> dict:
    n = len(rows)
    hit = sum(bool(r[field]) for r in rows)
    return {"n": n, "count": hit, "share": round(hit / n, 4) if n else None}


def summarize(rows: list[dict]) -> dict:
    ans = [r for r in rows if r["answerable"]]
    una = [r for r in rows if not r["answerable"]]
    given = [r for r in ans if not r["abstained"]]
    found = [r for r in ans if r["retrieval_hit"]]
    missed = [r for r in ans if not r["retrieval_hit"]]
    out = {
        "answerable": {"correct": share(ans, "correct"), "abstained": share(ans, "abstained"),
                       "retrieval_hit": share(ans, "retrieval_hit"),
                       "correct_when_retrieved": share(found, "correct"),
                       "correct_when_not_retrieved": share(missed, "correct"),
                       "abstained_when_not_retrieved": share(missed, "abstained")},
        "unanswerable": {"abstained": share(una, "abstained")},
        "given_answers": {"has_citation": share(given, "has_citation"),
                          "invalid_citation": share(given, "invalid_citation"),
                          "cites_relevant_passage": share(given, "cited_relevant"),
                          "contains_answer_key": share(given, "key_in_answer"),
                          "truncated": share(given, "truncated"),
                          "unparsed_verdict": share(given, "unparsed")},
        "overall": share(rows, "correct"),
        "by_category": {},
    }
    for cat in sorted({r["category"] for r in rows}):
        out["by_category"][cat] = share([r for r in rows if r["category"] == cat], "correct")
    return out


def control_summary(judged: list[dict]) -> dict:
    out = {}
    for kind in ("same", "other", "abstain"):
        items = [j for j in judged if j.get("kind") == kind]
        if items:
            out[kind] = {"n": len(items),
                         "as_expected": sum(j["verdict"] is j["expected"] for j in items),
                         "unparsed": sum(j["verdict"] is None for j in items)}
    return out


def judge_agreement(a: dict[str, bool | None], b: dict[str, bool | None]) -> dict:
    ids = sorted(set(a) & set(b))
    both = sum(a[i] is True and b[i] is True for i in ids)
    neither = sum(a[i] is not True and b[i] is not True for i in ids)
    return {"n": len(ids), "agree": both + neither, "both_correct": both,
            "both_incorrect": neither,
            "only_first": sum(a[i] is True and b[i] is not True for i in ids),
            "only_second": sum(a[i] is not True and b[i] is True for i in ids)}


# -- files -----------------------------------------------------------------------------------
def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
                    encoding="utf-8")


def write_scores(path: Path, rows: list[dict]) -> None:
    """The score file llm-eval-harness's gate reads: id,score."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("id,score\n" + "".join(f"{r['id']},{r['correct']}\n" for r in rows))


def verdicts_of(path: Path) -> dict[str, bool | None]:
    return {j["id"]: j["verdict"] for j in read_jsonl(path) if "kind" not in j}


# -- commands --------------------------------------------------------------------------------
def cmd_generate(args) -> None:
    from ragsvc.serve import build_service

    cfg = load_settings(args.config)
    cfg = dataclasses.replace(
        cfg, cache=CacheSettings(exact=False, semantic=False),
        generation=dataclasses.replace(cfg.generation, backend=args.backend))
    service = build_service(cfg)
    questions = load_questions(cfg.data_dir / "eval" / "questions.jsonl")

    def progress(i, n, rec):
        print(f"[{i}/{n}] {rec['id']} abstained={rec.get('abstained')} "
              f"out={rec.get('output_tokens')}", flush=True)

    records = asyncio.run(generate(service, questions, args.mode, args.k, progress))
    write_jsonl(args.out / "answers" / f"{args.name}.jsonl", records)
    service.close()


def cmd_judge(args) -> None:
    cfg = load_settings(args.config)
    questions = load_questions(cfg.data_dir / "eval" / "questions.jsonl")
    by_id = {q.id: q for q in questions}
    backend = make_backend(args.judge)

    def progress(i, n, j):
        if i % 20 == 0 or i == n:
            print(f"[{i}/{n}] {j['id']} -> {j['verdict']}", flush=True)

    for name in args.names:
        records = read_jsonl(args.out / "answers" / f"{name}.jsonl")
        judged = judge_items(backend, answer_items(records, by_id), progress)
        write_jsonl(args.out / "answers" / f"{name}.judge-{args.tag}.jsonl", judged)
    if args.controls:
        judged = judge_items(backend, control_items(questions), progress)
        write_jsonl(args.out / "answers" / f"controls.judge-{args.tag}.jsonl", judged)


def cmd_report(args) -> None:
    from llmeval.gate import compare, load_scores

    from ragsvc.answer_report import markdown

    cfg = load_settings(args.config)
    by_id = {q.id: q for q in load_questions(cfg.data_dir / "eval" / "questions.jsonl")}
    answers = args.out / "answers"
    rep: dict = {"judges": args.judges, "primary_judge": args.judges[0], "runs": {},
                 "controls": {}, "agreement": {}}
    for tag in args.judges:
        path = answers / f"controls.judge-{tag}.jsonl"
        if path.exists():
            rep["controls"][tag] = control_summary(read_jsonl(path))
    for name in args.names:
        records = read_jsonl(answers / f"{name}.jsonl")
        verdicts = {tag: verdicts_of(answers / f"{name}.judge-{tag}.jsonl")
                    for tag in args.judges}
        rows = score(records, verdicts[args.judges[0]], by_id)
        write_scores(args.evals / f"answers_{name}.csv", rows)
        run = {"mode": records[0]["mode"], "k": records[0]["k"],
               "model": next(r["model"] for r in records if r.get("model")),
               "n": len(rows), **summarize(rows),
               "input_tokens_mean": round(sum(r["input_tokens"] for r in records)
                                          / len(records), 1),
               "output_tokens_mean": round(sum(r["output_tokens"] for r in records)
                                           / len(records), 1),
               "by_judge": {tag: summarize(score(records, verdicts[tag], by_id))["answerable"]
                            ["correct"] for tag in args.judges}}
        rep["runs"][name] = run
        if len(args.judges) > 1:
            rep["agreement"][name] = judge_agreement(verdicts[args.judges[0]],
                                                     verdicts[args.judges[1]])
    if len(args.names) > 1:
        base, cand = args.names[:2]
        result = compare(load_scores(args.evals / f"answers_{base}.csv"),
                         load_scores(args.evals / f"answers_{cand}.csv"))
        rep["gate"] = {"baseline": base, "candidate": cand, **result.as_dict()}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "answer_eval.json").write_text(json.dumps(rep, indent=1) + "\n")
    (args.out / "answer_eval.md").write_text(markdown(rep))
    print(markdown(rep))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", type=Path, default=Path("reports"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate", help="answer every evaluation question")
    g.add_argument("--name", required=True)
    g.add_argument("--mode", default="hybrid_rerank")
    g.add_argument("--k", type=int, default=5)
    g.add_argument("--backend", default="mlx", choices=("mlx", "anthropic", "extractive"))
    g.set_defaults(fn=cmd_generate)
    j = sub.add_parser("judge", help="grade stored answers against the reference answers")
    j.add_argument("names", nargs="+")
    j.add_argument("--judge", required=True, help="mlx:<hf repo> or anthropic:<model>")
    j.add_argument("--tag", required=True, help="short name of the judge for file names")
    j.add_argument("--controls", action="store_true", help="also grade the control items")
    j.set_defaults(fn=cmd_judge)
    r = sub.add_parser("report", help="tables and the regression gate between two runs")
    r.add_argument("names", nargs="+", help="baseline first, then the candidate")
    r.add_argument("--judges", nargs="+", required=True, help="judge tags; the first decides")
    r.add_argument("--evals", type=Path, default=Path("evals"))
    r.set_defaults(fn=cmd_report)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
