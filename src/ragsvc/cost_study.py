"""`ragsvc-cost`: what a query costs, and what moves that number.

    tokens     prompt and answer size of the stored answers (reports/answers/<name>.jsonl)
    price      those token counts at Claude list prices: an estimate, see below
    context    passages per prompt (k) against prompt size and the chance that a relevant
               passage is in the prompt
    workload   the load test's skewed workload replayed through the service with no cache,
               the exact cache, and exact + semantic: answers generated per 1,000 requests

What is measured and what is not. Token counts are measured, but with the local model's
tokenizer, on the prompts and answers of the local model. Claude has a different tokenizer,
writes different answers, and bills its thinking tokens as output; none of that is
measured here, because it requires an Anthropic API key. The dollar figures are therefore
estimates of the order of magnitude, and the ratios between cache settings are the solid
part. Writes reports/cost.md and cost.json.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import statistics
from pathlib import Path

from ragsvc.answer_eval import read_jsonl
from ragsvc.cache import normalize_query
from ragsvc.cache_study import load_jsonl
from ragsvc.config import CacheSettings, Settings, load_settings
from ragsvc.cost import PRICES, cost_usd
from ragsvc.evalset import is_relevant, load_questions
from ragsvc.generate import SYSTEM_PROMPT, GenResult, build_user_message
from ragsvc.loadtest import percentile, zipf_workload

KS = (1, 3, 5, 8, 10)


def token_profile(records: list[dict]) -> dict:
    def stats(field: str) -> dict:
        v = [r[field] for r in records]
        return {"mean": round(statistics.fmean(v), 1), "p50": statistics.median(v),
                "p95": round(percentile(v, 95), 1), "max": max(v)}

    timed = [r["generation_ms"] for r in records if r.get("generation_ms")]
    return {"n": len(records), "input_tokens": stats("input_tokens"),
            "output_tokens": stats("output_tokens"),
            "generation_s_mean": round(statistics.fmean(timed) / 1e3, 2) if timed else None}


def price_rows(input_tokens: float, output_tokens: float) -> list[dict]:
    rows = []
    for model, (p_in, p_out) in PRICES.items():
        per_answer = cost_usd(model, input_tokens, output_tokens)
        rows.append({"model": model, "price_in": p_in, "price_out": p_out,
                     "usd_per_answer": per_answer, "usd_per_1000": 1000 * per_answer,
                     "input_share": (input_tokens * p_in / 1e6) / per_answer})
    return rows


def context_rows(searcher, questions, count_tokens, mode: str, ks=KS) -> list[dict]:
    """Prompt size and hit rate when the model is shown the top k passages."""
    answerable = [q for q in questions if q.answerable]
    tokens: dict[int, list[int]] = {k: [] for k in ks}
    hits: dict[int, int] = dict.fromkeys(ks, 0)
    for q in answerable:
        found = searcher.search(q.question, mode, max(ks)).hits
        rel = [is_relevant(h.chunk, q) for h in found]
        for k in ks:
            tokens[k].append(count_tokens(q.question, found[:k]))
            hits[k] += any(rel[:k])
    return [{"k": k, "input_tokens_mean": round(statistics.fmean(tokens[k]), 1),
             "hit_rate": round(hits[k] / len(answerable), 4), "n": len(answerable)}
            for k in ks]


class RecordedGenerator:
    """Replays stored answers with their token counts instead of running a model.

    A paraphrase is charged what its original question cost: the answer to the other
    wording was never generated, and the prompt differs only by the question line.
    """

    name = "recorded"

    def __init__(self, records: list[dict], aliases: dict[str, str] | None = None):
        self.by_query = {normalize_query(r["question"]): r for r in records}
        for wording, original in (aliases or {}).items():
            self.by_query[normalize_query(wording)] = self.by_query[normalize_query(original)]

    async def stream(self, question, hits):
        r = self.by_query[normalize_query(question)]
        yield r["answer"]
        yield GenResult(r["answer"], self.name, input_tokens=r["input_tokens"],
                        output_tokens=r["output_tokens"])


async def replay(searcher, generator, cfg: Settings, workload: list[str], mode: str) -> dict:
    from ragsvc.service import RagService

    service = RagService(searcher, generator, cfg)
    for q in workload:
        async for _ in service.ask(q, mode):
            pass
    usage, counters = service.usage, service.metrics.counters
    service.close()
    return {"requests": len(workload), "answers_generated": usage.answers_generated,
            "answers_from_cache": usage.answers_from_cache,
            "from_cache_semantic": counters.get("answer_cache_semantic", 0),
            "input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens}


def workload_rows(searcher, generator, cfg: Settings, workload: list[str], mode: str):
    configs = [("no cache", CacheSettings(exact=False, semantic=False)),
               ("exact", dataclasses.replace(cfg.cache, exact=True, semantic=False)),
               (f"exact + semantic ({cfg.cache.semantic_threshold:.2f})",
                dataclasses.replace(cfg.cache, exact=True, semantic=True))]
    rows = []
    for name, cache in configs:
        row = asyncio.run(replay(searcher, generator, dataclasses.replace(cfg, cache=cache),
                                 workload, mode))
        row["config"] = name
        row["usd"] = {m: cost_usd(m, row["input_tokens"], row["output_tokens"]) for m in PRICES}
        rows.append(row)
        print(f"workload {name}: {row['answers_generated']} answers generated")
    return rows


def usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.1 else f"${v:.4f}"


def markdown(rep: dict) -> str:
    t = rep["tokens"]
    L = [
        "# Cost per query",
        "",
        "Token counts are measured; dollar figures are estimates. The counts come from the "
        f"{t['n']} stored answers of `{rep['model']}` ({rep['mode']}, top {rep['k']} "
        "passages), counted with that model's tokenizer and including the system prompt. "
        "Claude tokenizes differently, writes different answers and bills its thinking as "
        "output tokens; measuring that requires an Anthropic API key. Read the dollar columns "
        "as an order of magnitude and the ratios as the result.",
        "",
        "## Tokens per answer",
        "",
        "| | Mean | p50 | p95 | Max |", "|---|---|---|---|---|",
        f"| Prompt (input) tokens | {t['input_tokens']['mean']:,.0f} | "
        f"{t['input_tokens']['p50']:,.0f} | {t['input_tokens']['p95']:,.0f} | "
        f"{t['input_tokens']['max']:,} |",
        f"| Answer (output) tokens | {t['output_tokens']['mean']:,.0f} | "
        f"{t['output_tokens']['p50']:,.0f} | {t['output_tokens']['p95']:,.0f} | "
        f"{t['output_tokens']['max']:,} |",
        "",
        f"The local model spends {t['generation_s_mean']} s of GPU time per answer on "
        "average, and answers one request at a time: about "
        f"{3600 / t['generation_s_mean']:,.0f} answers per hour on this machine, at no price "
        "per token.",
        "",
        "## The same token counts at Claude list prices (estimate)",
        "",
        "| Model | Input $/MTok | Output $/MTok | Per answer | Per 1,000 answers | "
        "Share spent on the prompt |", "|---|---|---|---|---|---|",
    ]
    for r in rep["prices"]:
        L.append(f"| {r['model']} | {r['price_in']:g} | {r['price_out']:g} | "
                 f"{usd(r['usd_per_answer'])} | {usd(r['usd_per_1000'])} | "
                 f"{100 * r['input_share']:.0f} % |")
    L += ["",
          "The prompt dominates: the passages are most of what is paid for, and the answer "
          "is short. Thinking tokens, which are not in this estimate, would raise the output "
          "side.",
          "", "## Passages in the prompt", "",
          f"`{rep['mode']}`, {rep['context'][0]['n']} answerable questions. \"Relevant passage "
          "in the prompt\" is retrieval Hit@k.", "",
          "| Passages (k) | Prompt tokens, mean | Relative to k = 5 | Relevant passage in the "
          "prompt |", "|---|---|---|---|"]
    at5 = next(r["input_tokens_mean"] for r in rep["context"] if r["k"] == 5)
    for r in rep["context"]:
        L.append(f"| {r['k']} | {r['input_tokens_mean']:,.0f} | "
                 f"{r['input_tokens_mean'] / at5:.2f}x | {100 * r['hit_rate']:.1f} % |")
    c = rep["tokenizer_check"]
    L += ["",
          f"Check: the offline count of the k = 5 prompts differs from the token count the "
          f"model reported at generation time by at most {c['max_abs_diff']} tokens over "
          f"{c['n']} prompts."]
    w = rep["workload"]
    L += ["", "## Caching on a skewed workload", "",
          f"{w['n']} requests drawn from the {w['n_questions']} evaluation questions with Zipf "
          f"popularity (s = 1.1, seed 0), {w['paraphrase_share']:.0%} in the paraphrased "
          "wording: the workload of the load test, replayed one request at a time. Stored "
          "answers stand in for the model, so only the number of answers that would have been "
          "generated is counted.", "",
          "| Cache | Answers generated | From cache | of which semantic | Input tokens | "
          "Output tokens | " + " | ".join(f"{m}" for m in PRICES) + " |",
          "|---|---|---|---|---|---|" + "---|" * len(PRICES)]
    for r in w["rows"]:
        L.append(f"| {r['config']} | {r['answers_generated']} | {r['answers_from_cache']} | "
                 f"{r['from_cache_semantic']} | {r['input_tokens']:,} | {r['output_tokens']:,} "
                 f"| " + " | ".join(usd(r["usd"][m]) for m in PRICES) + " |")
    L += ["",
          "Dollar columns: estimated cost of the whole workload on each model. A paraphrase "
          "is charged what its original question cost. How much a cache saves depends "
          "entirely on how often questions repeat; this workload is built to repeat."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--answers", type=Path, default=Path("reports/answers/hybrid_rerank.jsonl"))
    ap.add_argument("--eval-dir", type=Path, default=Path("eval"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--requests", type=int, default=1000)
    args = ap.parse_args(argv)

    from transformers import AutoTokenizer

    from ragsvc.serve import build_service

    cfg = load_settings(args.config)
    records = read_jsonl(args.answers)
    mode, k = records[0]["mode"], records[0]["k"]
    questions = load_questions(cfg.data_dir / "eval" / "questions.jsonl")
    paras = load_jsonl(args.eval_dir / "paraphrases.jsonl")
    by_id = {q.id: q for q in questions}
    searcher = build_service(cfg, RecordedGenerator(records)).searcher
    tokenizer = AutoTokenizer.from_pretrained(records[0]["model"])

    def count_tokens(question: str, hits) -> int:
        messages = [{"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": build_user_message(question, hits)}]
        # render to text first: with tokenize=True newer transformers return a dict
        text = tokenizer.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        return len(tokenizer(text, add_special_tokens=False)["input_ids"])

    profile = token_profile(records)
    rep = {"model": records[0]["model"], "mode": mode, "k": k, "tokens": profile,
           "prices": price_rows(profile["input_tokens"]["mean"],
                                profile["output_tokens"]["mean"]),
           "context": context_rows(searcher, questions, count_tokens, mode)}
    diffs = [abs(count_tokens(r["question"], searcher.search(r["question"], mode, k).hits)
                 - r["input_tokens"]) for r in records]
    rep["tokenizer_check"] = {"n": len(diffs), "max_abs_diff": max(diffs)}
    share = 0.3
    ids = [q.id for q in questions]
    workload = zipf_workload([q.question for q in questions], args.requests, s=1.1, seed=0,
                             paraphrases={ids.index(p["id"]): p["question"] for p in paras},
                             paraphrase_share=share)
    generator = RecordedGenerator(records, {p["question"]: by_id[p["id"]].question
                                            for p in paras})
    rep["workload"] = {"n": args.requests, "n_questions": len(questions),
                       "paraphrase_share": share,
                       "rows": workload_rows(searcher, generator, cfg, workload, mode)}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "cost.json").write_text(json.dumps(rep, indent=1) + "\n")
    (args.out / "cost.md").write_text(markdown(rep))
    print(markdown(rep))


if __name__ == "__main__":
    main()
