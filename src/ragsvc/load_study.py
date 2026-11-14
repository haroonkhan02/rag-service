"""`ragsvc-loadtest`: start the service in different configurations and measure it.

    pipeline    caches and rate limit off, distinct questions: latency and throughput of the
                search pipeline itself at 1 / 4 / 16 clients
    overload    more clients than the queue admits: how much is shed, how slow the rest is
    cache       a skewed (Zipf) workload with paraphrases: no cache, exact, exact + semantic
    ratelimit   one client sending faster than its limit
    ask         streamed answers from a local MLX model: time to first token, tokens/s,
                and the same questions again from the answer cache  (only with --ask)

Each scenario gets a fresh server process, so caches and counters start empty. Writes
reports/load_test.md and load_test.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
from pathlib import Path

import httpx

from ragsvc.cache_study import load_jsonl
from ragsvc.config import load_settings
from ragsvc.evalset import load_questions
from ragsvc.loadtest import (
    ServerProcess,
    closed_loop,
    fixed_rate,
    post_ask,
    summarize,
    zipf_workload,
)

OFF = {"CACHE__EXACT": "false", "CACHE__SEMANTIC": "false", "RATE_LIMIT__ENABLED": "false",
       "GENERATION__BACKEND": "extractive"}
NO_LIMIT = {"RATE_LIMIT__ENABLED": "false", "GENERATION__BACKEND": "extractive"}


def new_client(url: str, timeout: float = 120.0) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=512, max_keepalive_connections=512)
    return httpx.AsyncClient(base_url=url, timeout=timeout, limits=limits)


async def run_closed(url: str, queries, concurrency: int, **kw) -> dict:
    async with new_client(url) as client:
        samples, wall = await closed_loop(client, queries, concurrency, **kw)
    return summarize(samples, wall)


def pipeline(port: int, distinct: list[str], passes: int) -> list[dict]:
    rows = []
    for mode in ("hybrid_rerank", "hybrid"):
        for c in (1, 4, 16):
            with ServerProcess(port, OFF) as srv:
                s = asyncio.run(run_closed(srv.url, distinct * passes, c, mode=mode))
            rows.append({"mode": mode, "concurrency": c, **s})
            print(f"pipeline {mode} c={c}: {s['throughput_rps']} rps, {s['latency_ms']}")
    return rows


def overload(port: int, distinct: list[str], passes: int, max_queue: int) -> list[dict]:
    rows = []
    for c in (32, 128):
        with ServerProcess(port, {**OFF, "SERVER__MAX_QUEUE": max_queue}) as srv:
            s = asyncio.run(run_closed(srv.url, distinct * passes, c, mode="hybrid_rerank"))
            shed = srv.metrics()["counters"].get("shed", 0)
        rows.append({"concurrency": c, "max_queue": max_queue, "shed": shed, **s})
        print(f"overload c={c}: {s['status']}, {s.get('latency_ms')}")
    return rows


def cache(port: int, questions: list[str], paraphrases: dict[int, str], n: int,
          share: float, threshold: float) -> list[dict]:
    workload = zipf_workload(questions, n, s=1.1, seed=0, paraphrases=paraphrases,
                             paraphrase_share=share)
    configs = [("no cache", {"CACHE__EXACT": "false", "CACHE__SEMANTIC": "false"}),
               ("exact", {"CACHE__EXACT": "true", "CACHE__SEMANTIC": "false"}),
               (f"exact + semantic ({threshold:.2f})",
                {"CACHE__EXACT": "true", "CACHE__SEMANTIC": "true"})]
    rows = []
    for name, over in configs:
        with ServerProcess(port, {**NO_LIMIT, **over}) as srv:
            s = asyncio.run(run_closed(srv.url, workload, 8, mode="hybrid_rerank"))
        rows.append({"config": name, "distinct_wordings": len(set(workload)), **s})
        print(f"cache {name}: {s['throughput_rps']} rps, {s['cache']}, {s['latency_ms']}")
    return rows


def ratelimit(port: int, question: str, rate: float, seconds: float, limit: float,
              burst: int) -> dict:
    n = round(rate * seconds)
    with ServerProcess(port, {"GENERATION__BACKEND": "extractive",
                              "RATE_LIMIT__RATE_PER_S": limit,
                              "RATE_LIMIT__BURST": burst}) as srv:
        async def run():
            async with new_client(srv.url) as client:
                return await fixed_rate(client, [question] * n, rate, mode="hybrid_rerank",
                                        headers={"X-API-Key": "loadtest"})
        samples, wall = asyncio.run(run())
    s = summarize(samples, wall)
    return {"sent_rate_per_s": rate, "seconds": seconds, "limit_per_s": limit, "burst": burst,
            "expected_allowed": round(burst + limit * (n - 1) / rate), **s}


def ask(port: int, model: str, fresh: list[str], concurrent: list[str]) -> dict:
    over = {"RATE_LIMIT__ENABLED": "false", "GENERATION__BACKEND": "mlx",
            "GENERATION__MLX_MODEL": model}

    def detail(samples) -> dict:
        ok = [s for s in samples if s.status == 200 and s.extra.get("output_tokens")]
        if not ok:
            return {}
        rate = [s.extra["output_tokens"] / (s.extra["generation_ms"] / 1e3) for s in ok
                if s.extra["generation_ms"]]
        return {"output_tokens_median": statistics.median(s.extra["output_tokens"] for s in ok),
                "input_tokens_median": statistics.median(s.extra["input_tokens"] for s in ok),
                "tokens_per_s_median": round(statistics.median(rate), 1) if rate else None,
                "abstained": sum(s.extra["abstained"] for s in ok),
                "no_citation": sum(s.extra["citations"] == 0 and not s.extra["abstained"]
                                   for s in ok),
                "invalid_citations": sum(s.extra["invalid_citations"] > 0 for s in ok),
                "truncated": sum(s.extra["stop_reason"] == "max_tokens" for s in ok)}

    out = {"model": model}
    with ServerProcess(port, over, ready_timeout_s=900) as srv:
        async def run(queries, c):
            async with new_client(srv.url, timeout=600.0) as client:
                await post_ask(client, "What is a launch file?")  # warm the model
                samples, wall = await closed_loop(client, queries, c, send=post_ask)
            return {**summarize(samples, wall), **detail(samples)}

        out["sequential"] = asyncio.run(run(fresh, 1))
        print("ask sequential:", out["sequential"])
        out["repeat_cached"] = asyncio.run(run(fresh, 1))
        print("ask cached:", out["repeat_cached"])
        out["concurrent_4"] = asyncio.run(run(concurrent, 4))
        print("ask c=4:", out["concurrent_4"])
    return out


def lat(s: dict) -> str:
    v = s.get("latency_ms")
    return f"{v['p50']:.1f} | {v['p95']:.1f} | {v['p99']:.1f}" if v else "- | - | -"


def ms(value) -> str:
    if value is None:
        return "-"
    return f"{value:,.0f}" if value >= 10 else f"{value:.1f}"


def markdown(rep: dict) -> str:
    L = [
        "# Load test",
        "",
        f"Service and client on the same machine ({rep['machine']}), one server process with "
        "one model worker thread, embedder and cross-encoder on "
        f"`{rep['device']}`. `/search`, {rep['n_distinct']} distinct questions unless noted. "
        "Closed loop: each client sends its next request when the previous one returns. "
        "Latency is measured at the client in milliseconds, over successful requests.",
        "",
        "## Search pipeline, caches off",
        "",
        "| Mode | Clients | Requests | Throughput (req/s) | p50 | p95 | p99 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rep["pipeline"]:
        L.append(f"| {r['mode']} | {r['concurrency']} | {r['ok']}/{r['requests']} | "
                 f"{r['throughput_rps']:.1f} | {lat(r)} |")
    L += ["", "## Overload, caches off (hybrid_rerank)", "",
          f"Queue limit {rep['overload'][0]['max_queue']} waiting searches; beyond that the "
          "service answers 503 immediately. Rejected clients move straight on to their next "
          "question, so the same question is often in flight twice and the second request "
          "shares the first one's result.", "",
          "| Clients | Requests | 200 (computed) | 200 (shared) | 503 | Searches/s | p50 "
          "| p95 | p99 |", "|---|---|---|---|---|---|---|---|---|"]
    for r in rep["overload"]:
        computed = r["cache"].get("miss", 0)
        L.append(f"| {r['concurrency']} | {r['requests']} | {computed} | "
                 f"{r['cache'].get('shared', 0)} | {r['status'].get('503', 0)} | "
                 f"{computed / r['wall_s']:.1f} | {lat(r)} |")
    c = rep["cache"]
    L += ["", "## Caching on a skewed workload (hybrid_rerank, 8 clients)", "",
          f"{c['n']} requests drawn from the {c['n_questions']} evaluation questions with Zipf "
          f"popularity (s = 1.1, seed 0); {c['paraphrase_share']:.0%} of them use the "
          f"paraphrased wording ({c['rows'][0]['distinct_wordings']} distinct wordings in "
          "total). \"shared\" = joined an identical search already in flight.", "",
          "| Cache | Throughput (req/s) | p50 | p95 | p99 | miss | exact | semantic | shared |",
          "|---|---|---|---|---|---|---|---|---|"]
    for r in c["rows"]:
        k = r["cache"]
        L.append(f"| {r['config']} | {r['throughput_rps']:.1f} | {lat(r)} | {k.get('miss', 0)} "
                 f"| {k.get('exact', 0)} | {k.get('semantic', 0)} | {k.get('shared', 0)} |")
    r = rep["ratelimit"]
    L += ["", "## Rate limit", "",
          f"One client sends {r['sent_rate_per_s']:g} req/s for {r['seconds']:g} s against a "
          f"limit of {r['limit_per_s']:g} req/s with a burst of {r['burst']}: "
          f"{r['status'].get('200', 0)} of {r['requests']} accepted "
          f"(token bucket predicts {r['expected_allowed']}), "
          f"{r['status'].get('429', 0)} answered 429 with Retry-After."]
    if rep.get("ask"):
        a = rep["ask"]
        L += ["", "## Streamed answers (`/ask`, local model)", "",
              f"Generator: `{a['model']}` through mlx-lm on the same machine, greedy decoding, "
              "so these numbers describe this laptop and model, not Claude. Time to first "
              "token includes retrieval and reading the prompt (a median of "
              f"{a['sequential'].get('input_tokens_median', 0):.0f} input tokens); output "
              "tokens/s is over the whole generation, prompt included. The model generates "
              "one answer at a time, so with four clients most of the wait is the queue.", "",
              "| Run | Requests | TTFT p50 | TTFT p95 | Total p50 | Total p95 "
              "| Output tokens/s | Output tokens (median) |", "|---|---|---|---|---|---|---|---|"]
        for name, key in (("One client, new questions", "sequential"),
                          ("Same questions again (answer cache)", "repeat_cached"),
                          ("Four clients, new questions", "concurrent_4")):
            s = a[key]
            t, v = s.get("ttft_ms", {}), s.get("latency_ms", {})
            L.append(f"| {name} | {s['ok']}/{s['requests']} | {ms(t.get('p50'))} | "
                     f"{ms(t.get('p95'))} | {ms(v.get('p50'))} | {ms(v.get('p95'))} | "
                     f"{s.get('tokens_per_s_median') or '-'} | "
                     f"{s.get('output_tokens_median', '-')} |")
        s, n = a["sequential"], a["sequential"]["ok"]
        L += ["", f"Of the {n} answers to new questions: {s.get('abstained', 0)} abstained, "
              f"{s.get('no_citation', 0)} had no citation, {s.get('invalid_citations', 0)} "
              f"cited a passage number that does not exist, {s.get('truncated', 0)} hit the "
              "token limit. Answer quality is not judged here."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--eval-dir", type=Path, default=Path("eval"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--only", nargs="+", default=None,
                    choices=("pipeline", "overload", "cache", "ratelimit", "ask", "report"),
                    help="scenarios to (re)run; `report` only re-renders the saved results")
    ap.add_argument("--ask", action="store_true", help="also run the MLX /ask scenario")
    ap.add_argument("--cache-requests", type=int, default=1000)
    args = ap.parse_args(argv)
    cfg = load_settings(args.config)
    seed = load_questions(cfg.data_dir / "eval" / "questions.jsonl")
    ids = [q.id for q in seed]
    paras = load_jsonl(args.eval_dir / "paraphrases.jsonl")
    nears = load_jsonl(args.eval_dir / "near_misses.jsonl")
    questions = [q.question for q in seed]
    distinct = questions + [p["question"] for p in paras] + [p["question"] for p in nears]
    which = set(args.only or ["pipeline", "overload", "cache", "ratelimit"])
    if args.ask:
        which.add("ask")

    from ragsvc.models import pick_device

    path = args.out / "load_test.json"
    rep = json.loads(path.read_text()) if path.exists() else {}
    rep.update({"machine": f"{platform.system()} {platform.machine()}",
                "device": pick_device(cfg.embedding.device), "n_distinct": len(distinct)})
    if "pipeline" in which:
        rep["pipeline"] = pipeline(args.port, distinct, passes=2)
    if "overload" in which:
        rep["overload"] = overload(args.port, distinct, 4, cfg.server.max_queue)
    if "cache" in which:
        share = 0.3
        rep["cache"] = {"n": args.cache_requests, "n_questions": len(questions),
                        "paraphrase_share": share,
                        "rows": cache(args.port, questions,
                                      {ids.index(p["id"]): p["question"] for p in paras},
                                      args.cache_requests, share, cfg.cache.semantic_threshold)}
    if "ratelimit" in which:
        r = cfg.rate_limit
        rep["ratelimit"] = ratelimit(args.port, questions[0], 20.0, 10.0, r.rate_per_s, r.burst)
        print("ratelimit:", rep["ratelimit"]["status"])
    if "ask" in which:
        answerable = [q.question for q in seed if q.answerable]
        rep["ask"] = ask(args.port, cfg.generation.mlx_model, answerable[:20], answerable[20:40])
    args.out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rep, indent=1) + "\n")
    needed = ("pipeline", "overload", "cache", "ratelimit")
    if all(k in rep for k in needed):
        (args.out / "load_test.md").write_text(markdown(rep))
        print(markdown(rep))


if __name__ == "__main__":
    main()
