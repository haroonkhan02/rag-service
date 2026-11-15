"""`ragsvc-trace-study`: what the traces say, and what tracing costs.

    breakdown   run the evaluation questions through /search three times (new, repeated,
                paraphrased) and read the time per step off the exported spans, grouped by
                cache outcome
    ask         a few /ask requests: retrieval against generation, time to first token
    overhead    the same requests with tracing off and on, in alternating blocks, for a
                fast path (cache hit), a medium one (hybrid) and a slow one (rerank)
    example     one /ask trace printed as a tree

The service runs in this process and is driven through its ASGI app, so there is no
network between client and server and latencies are lower than in the load test. Spans go
to an in-memory exporter. Writes reports/tracing.md and tracing.json.
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import platform
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ragsvc.cache_study import load_jsonl
from ragsvc.config import CacheSettings, RateLimitSettings, Settings, load_settings
from ragsvc.evalset import load_questions
from ragsvc.loadtest import percentile

STEPS = ("rag.search", "rag.embed", "rag.retrieve")
STAGES = ("rag.bm25_ms", "rag.dense_ms", "rag.rerank_ms", "rag.queue_wait_ms")


def duration_ms(span) -> float:
    return (span.end_time - span.start_time) / 1e6


def group_traces(spans) -> list[dict]:
    """One dict per trace: span name -> span (names are unique within our traces)."""
    traces: dict[int, dict] = defaultdict(dict)
    for s in spans:
        traces[s.context.trace_id][s.name] = s
    return list(traces.values())


def breakdown_rows(traces: list[dict], root: str = "POST /search") -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for t in traces:
        if root in t and "rag.search" in t:
            groups[t["rag.search"].attributes.get("rag.cache", "?")].append(t)
    rows = []
    for cache in ("miss", "semantic", "exact"):
        ts = groups.get(cache, [])
        if not ts:
            continue
        row = {"cache": cache, "n": len(ts),
               "request_p50": statistics.median(duration_ms(t[root]) for t in ts),
               "request_p95": percentile([duration_ms(t[root]) for t in ts], 95)}
        for name in STEPS:
            vals = [duration_ms(t[name]) for t in ts if name in t]
            row[name] = statistics.median(vals) if vals else None
        for attr in STAGES:
            vals = [t["rag.retrieve"].attributes[attr] for t in ts
                    if "rag.retrieve" in t and attr in t["rag.retrieve"].attributes]
            row[attr] = statistics.median(vals) if vals else None
        row["http_layer"] = statistics.median(
            duration_ms(t[root]) - duration_ms(t["rag.search"]) for t in ts)
        rows.append(row)
    return rows


def render_tree(trace: dict) -> list[str]:
    spans = sorted(trace.values(), key=lambda s: s.start_time)
    ids = {s.context.span_id for s in spans}
    children: dict[int | None, list] = defaultdict(list)
    for s in spans:
        parent = s.parent.span_id if s.parent and s.parent.span_id in ids else None
        children[parent].append(s)
    keep = ("rag.cache", "rag.mode", "rag.k", "rag.rerank_ms", "rag.queue_wait_ms",
            "gen_ai.request.model", "gen_ai.usage.input_tokens", "gen_ai.usage.output_tokens",
            "rag.ttft_ms", "http.response.status_code")
    lines: list[str] = []

    def walk(span, depth: int) -> None:
        attrs = ", ".join(f"{k.split('.')[-1]}={_short(span.attributes[k])}"
                          for k in keep if k in span.attributes)
        lines.append(f"{'  ' * depth}{span.name:<{24 - 2 * depth}} {duration_ms(span):8.1f} ms"
                     + (f"   {attrs}" if attrs else ""))
        for c in children[span.context.span_id]:
            walk(c, depth + 1)

    for root in children[None]:
        walk(root, 0)
    return lines


def _short(v):
    return round(v, 1) if isinstance(v, float) else v


async def post(client, path: str, query: str, mode: str | None = None) -> float:
    body = {"query": query, "stream": False} if path == "/ask" else {"query": query}
    if mode:
        body["mode"] = mode
    t0 = time.perf_counter()
    r = await client.post(path, json=body)
    r.raise_for_status()
    return (time.perf_counter() - t0) * 1e3


def make_app(searcher, generator, cfg: Settings, exporter=None, batch: bool = False):
    from ragsvc.app import create_app
    from ragsvc.config import TracingSettings
    from ragsvc.service import RagService
    from ragsvc.tracing import Tracer, setup_tracing

    tracer = (setup_tracing(TracingSettings(enabled=True), exporter, batch=batch)
              if exporter is not None else Tracer())
    return create_app(RagService(searcher, generator, cfg, tracer))


def client_for(app):
    import httpx

    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://svc",
                             timeout=600.0)


class DropExporter:
    """Accepts spans and throws them away: the cost of tracing without a collector."""

    def __init__(self) -> None:
        self.spans = 0

    def export(self, spans):
        from opentelemetry.sdk.trace.export import SpanExportResult

        self.spans += len(spans)
        return SpanExportResult.SUCCESS

    def shutdown(self) -> None:
        pass

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


async def run_breakdown(searcher, generator, cfg, questions, paraphrases) -> list[dict]:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    async with client_for(make_app(searcher, generator, cfg, exporter)) as client:
        for batch in (questions, questions, paraphrases):
            for q in batch:
                await post(client, "/search", q)
    return breakdown_rows(group_traces(exporter.get_finished_spans()))


async def run_ask(searcher, generator, cfg, questions) -> tuple[dict, list[str]]:
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    exporter = InMemorySpanExporter()
    async with client_for(make_app(searcher, generator, cfg, exporter)) as client:
        for q in questions:
            await post(client, "/ask", q)
    traces = [t for t in group_traces(exporter.get_finished_spans()) if "rag.generate" in t]

    def med(name):
        return statistics.median(duration_ms(t[name]) for t in traces)

    gen = [t["rag.generate"].attributes for t in traces]
    row = {"n": len(traces), "generator": generator.name,
           "request_p50": med("POST /ask"), "search_p50": med("rag.search"),
           "generate_p50": med("rag.generate"),
           "ttft_p50": statistics.median(a["rag.ttft_ms"] for a in gen if "rag.ttft_ms" in a),
           "input_tokens_p50": statistics.median(a["gen_ai.usage.input_tokens"] for a in gen),
           "output_tokens_p50": statistics.median(a["gen_ai.usage.output_tokens"] for a in gen)}
    slowest = max(traces, key=lambda t: duration_ms(t["POST /ask"]))
    return row, render_tree(slowest)


async def run_overhead(searcher, generator, cfg, questions, rounds: int) -> list[dict]:
    off = dataclasses.replace(cfg, cache=CacheSettings(exact=False, semantic=False))
    paths = [("exact cache hit", cfg, "hybrid_rerank", True),
             ("hybrid, no cache", off, "hybrid", False),
             ("hybrid + rerank, no cache", off, "hybrid_rerank", False)]
    rows = []
    for name, settings, mode, prime in paths:
        drop = DropExporter()
        apps = {"off": make_app(searcher, generator, settings),
                "on": make_app(searcher, generator, settings, drop, batch=True)}
        lat: dict[str, list[float]] = {"off": [], "on": []}
        blocks: dict[str, list[float]] = {"off": [], "on": []}
        async with client_for(apps["off"]) as c_off, client_for(apps["on"]) as c_on:
            clients = {"off": c_off, "on": c_on}
            for c in clients.values():  # warm both; fills the cache for the hit path
                for q in questions if prime else questions[:10]:
                    await post(c, "/search", q, mode)
            apps["on"].state.service.tracer.flush()  # the batch processor exports late
            before = drop.spans
            for r in range(rounds):
                for arm in (("off", "on") if r % 2 == 0 else ("on", "off")):
                    block = [await post(clients[arm], "/search", q, mode) for q in questions]
                    lat[arm] += block
                    blocks[arm].append(statistics.median(block))
        apps["on"].state.service.tracer.flush()
        n = len(lat["on"])
        m_off, m_on = statistics.median(lat["off"]), statistics.median(lat["on"])
        rows.append({"path": name, "requests_per_arm": n,
                     "off_p50_ms": m_off, "on_p50_ms": m_on,
                     "off_mean_ms": statistics.fmean(lat["off"]),
                     "on_mean_ms": statistics.fmean(lat["on"]),
                     "off_block_p50_ms": blocks["off"], "on_block_p50_ms": blocks["on"],
                     "overhead_ms": m_on - m_off,
                     "overhead_pct": 100.0 * (m_on - m_off) / m_off,
                     "spans_per_request": (drop.spans - before) / n})
        print(f"overhead {name}: off {m_off:.3f} ms, on {m_on:.3f} ms")
    return rows


def f(v, digits: int = 1) -> str:
    return "-" if v is None else f"{v:,.{digits}f}"


def markdown(rep: dict) -> str:
    L = [
        "# Tracing",
        "",
        f"Measured from the service's own OpenTelemetry spans ({rep['machine']}, models on "
        f"`{rep['device']}`). The service runs in the measuring process and is called through "
        "its ASGI app, one request at a time, so there is no network hop and no queueing; "
        "times are milliseconds.",
        "",
        "## Where a search spends its time",
        "",
        f"`/search`, hybrid + rerank, default caches. The {rep['n_questions']} evaluation "
        "questions are sent once (all misses), sent again (exact hits), then sent in their "
        "paraphrased wording (a semantic hit when the similarity reaches "
        f"{rep['semantic_threshold']}, else a miss). Medians per cache outcome.",
        "",
        "| Cache outcome | Requests | Request p50 | p95 | rag.search | rag.embed | "
        "rag.retrieve | BM25 | Dense | Rerank | Wait for worker | HTTP layer |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["breakdown"]:
        L.append(f"| {r['cache']} | {r['n']} | {f(r['request_p50'], 2)} | "
                 f"{f(r['request_p95'], 2)} | {f(r['rag.search'], 2)} | {f(r['rag.embed'], 2)} "
                 f"| {f(r['rag.retrieve'], 2)} | {f(r['rag.bm25_ms'], 2)} | "
                 f"{f(r['rag.dense_ms'], 2)} | {f(r['rag.rerank_ms'], 2)} | "
                 f"{f(r['rag.queue_wait_ms'], 2)} | {f(r['http_layer'], 2)} |")
    L += ["",
          "`rag.search`, `rag.embed` and `rag.retrieve` are span durations; BM25, dense, "
          "rerank and the wait for a model worker are attributes of `rag.retrieve`. The HTTP "
          "layer is the server span minus `rag.search`: request parsing, validation, the rate "
          "limiter and JSON encoding."]
    if rep.get("ask"):
        a = rep["ask"]
        L += ["", "## Where an answer spends its time", "",
              f"`/ask` without streaming, {a['n']} new questions, generator `{a['generator']}`.",
              "",
              "| Request p50 | rag.search p50 | rag.generate p50 | Time to first token p50 | "
              "Input tokens p50 | Output tokens p50 |",
              "|---|---|---|---|---|---|",
              f"| {f(a['request_p50'], 0)} | {f(a['search_p50'], 0)} | "
              f"{f(a['generate_p50'], 0)} | {f(a['ttft_p50'], 0)} | "
              f"{f(a['input_tokens_p50'], 0)} | {f(a['output_tokens_p50'], 0)} |",
              "", "The slowest of these requests as a trace:", "", "```text",
              *rep["example_trace"], "```"]
    L += ["", "## What tracing costs", "",
          f"The same `/search` requests with tracing off and on, in {rep['rounds']} "
          "alternating blocks per arm. With tracing on, spans are created, batched and "
          "handed to an exporter that discards them, so this is the cost inside the service "
          "without the cost of shipping spans to a collector.",
          "",
          "| Path | Requests per arm | Spans per request | p50 off | p50 on | Difference "
          "| Relative | Block medians off | Block medians on |",
          "|---|---|---|---|---|---|---|---|---|"]

    def spread(v: list[float]) -> str:
        return f"{min(v):.3f} to {max(v):.3f}"

    for r in rep["overhead"]:
        L.append(f"| {r['path']} | {r['requests_per_arm']} | {r['spans_per_request']:.0f} | "
                 f"{r['off_p50_ms']:.3f} | {r['on_p50_ms']:.3f} | {r['overhead_ms']:+.3f} | "
                 f"{r['overhead_pct']:+.1f} % | {spread(r['off_block_p50_ms'])} | "
                 f"{spread(r['on_block_p50_ms'])} |")
    L += ["",
          "The last two columns give the range of the per-block medians. Where the two "
          "ranges overlap, the difference is not distinguishable from run-to-run variation."]
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--eval-dir", type=Path, default=Path("eval"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--ask", type=int, default=10, help="number of /ask requests (0 = skip)")
    ap.add_argument("--rounds", type=int, default=4, help="blocks per arm in the overhead test")
    args = ap.parse_args(argv)

    from ragsvc.generate import ExtractiveGenerator, make_generator
    from ragsvc.models import pick_device
    from ragsvc.serve import build_service

    cfg = dataclasses.replace(load_settings(args.config),
                              rate_limit=RateLimitSettings(enabled=False))
    generator = make_generator(cfg.generation) if args.ask else ExtractiveGenerator()
    base = build_service(cfg, generator)
    searcher, generator = base.searcher, base.generator
    seed = [q for q in load_questions(cfg.data_dir / "eval" / "questions.jsonl") if q.answerable]
    by_id = {p["id"]: p["question"] for p in load_jsonl(args.eval_dir / "paraphrases.jsonl")}
    questions = [q.question for q in seed]
    paraphrases = [by_id[q.id] for q in seed if q.id in by_id]

    rep = {"machine": f"{platform.system()} {platform.machine()}",
           "device": pick_device(cfg.embedding.device), "n_questions": len(questions),
           "semantic_threshold": cfg.cache.semantic_threshold, "rounds": args.rounds}
    rep["breakdown"] = asyncio.run(run_breakdown(searcher, generator, cfg, questions,
                                                 paraphrases))
    if args.ask:
        rep["ask"], rep["example_trace"] = asyncio.run(
            run_ask(searcher, generator, cfg, questions[: args.ask]))
    rep["overhead"] = asyncio.run(run_overhead(searcher, generator, cfg, questions,
                                               args.rounds))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "tracing.json").write_text(json.dumps(rep, indent=1) + "\n")
    (args.out / "tracing.md").write_text(markdown(rep))
    print(markdown(rep))


if __name__ == "__main__":
    main()
