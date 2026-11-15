"""Tracing: span tree, attributes, propagation and sampling, with an in-memory exporter."""

import asyncio
import dataclasses

import pytest

pytest.importorskip("opentelemetry.sdk")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode  # noqa: E402

from ragsvc.app import create_app  # noqa: E402
from ragsvc.config import (  # noqa: E402
    RateLimitSettings,
    ServerSettings,
    Settings,
    TracingSettings,
)
from ragsvc.generate import ExtractiveGenerator  # noqa: E402
from ragsvc.service import Overloaded, RagService  # noqa: E402
from ragsvc.tracing import NOOP, Tracer, setup_tracing  # noqa: E402

REMOTE_TRACE = "0af7651916cd43dd8448eb211c80319c"
REMOTE_SPAN = "b7ad6b7169203331"


def traced(searcher, cfg=None, **tracing):
    exporter = InMemorySpanExporter()
    tracer = setup_tracing(TracingSettings(enabled=True, **tracing), exporter, batch=False)
    return RagService(searcher, ExtractiveGenerator(), cfg or Settings(), tracer), exporter


def by_name(exporter):
    return {s.name: s for s in exporter.get_finished_spans()}


def parent_id(span):
    return span.parent.span_id if span.parent else None


def test_disabled_tracer_hands_out_the_noop_span():
    tracer = setup_tracing(TracingSettings())
    assert not tracer.enabled
    span = tracer.start("anything", **{"a": 1})
    assert span is NOOP and span.child("x") is NOOP and span.trace_id is None
    with span:
        span.set(a=None)
    assert not Tracer().enabled


def test_search_span_tree_and_attributes(searcher):
    svc, exporter = traced(searcher)
    asyncio.run(svc.search("How do I record a bag?", "hybrid_rerank", 3))
    spans = by_name(exporter)
    assert set(spans) == {"rag.search", "rag.embed", "rag.retrieve"}
    root = spans["rag.search"]
    assert parent_id(root) is None
    assert parent_id(spans["rag.embed"]) == root.context.span_id
    assert parent_id(spans["rag.retrieve"]) == root.context.span_id
    assert root.attributes["rag.mode"] == "hybrid_rerank" and root.attributes["rag.k"] == 3
    assert root.attributes["rag.cache"] == "miss" and root.attributes["rag.hits"] == 3
    assert "rag.cache.similarity" not in root.attributes  # None is dropped, not exported
    stage = spans["rag.retrieve"].attributes
    assert {"rag.bm25_ms", "rag.dense_ms", "rag.rerank_ms", "rag.queue_wait_ms"} <= set(stage)


def test_cache_hit_is_one_span_without_model_work(searcher):
    svc, exporter = traced(searcher)

    async def run():
        await svc.search("How do I record a bag?", "hybrid", 3)
        exporter.clear()
        await svc.search("how do i record a bag", "hybrid", 3)

    asyncio.run(run())
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["rag.search"]
    assert spans[0].attributes["rag.cache"] == "exact"


def test_ask_trace_has_search_and_generation_with_usage(searcher):
    svc, exporter = traced(searcher)

    async def run():
        return [ev async for ev in svc.ask("How do I record a bag?", "hybrid", 2)]

    asyncio.run(run())
    spans = by_name(exporter)
    ask = spans["rag.ask"]
    assert {s.context.trace_id for s in spans.values()} == {ask.context.trace_id}
    assert parent_id(spans["rag.search"]) == ask.context.span_id
    assert parent_id(spans["rag.generate"]) == ask.context.span_id
    gen = spans["rag.generate"].attributes
    assert gen["gen_ai.request.model"] == "extractive"
    assert gen["gen_ai.response.finish_reasons"] == ("end_turn",)
    assert "gen_ai.usage.output_tokens" in gen and "rag.ttft_ms" in gen
    assert ask.attributes["rag.cache"] == "miss" and ask.attributes["rag.citations"] == 1


def test_overload_is_recorded_as_an_error_span(searcher):
    cfg = dataclasses.replace(Settings(), server=ServerSettings(model_workers=1, max_queue=0))
    svc, exporter = traced(searcher, cfg)
    svc._waiting = 1  # the only worker slot is taken

    async def run():
        with pytest.raises(Overloaded):
            await svc.search("anything", "bm25", 1)

    asyncio.run(run())
    spans = by_name(exporter)
    assert spans["rag.search"].status.status_code == StatusCode.ERROR
    assert spans["rag.retrieve"].events[0].name == "exception"


def test_http_request_joins_the_callers_trace(searcher):
    svc, exporter = traced(searcher)
    client = TestClient(create_app(svc))
    r = client.post("/search", json={"query": "ros2 bag record", "mode": "bm25"},
                    headers={"traceparent": f"00-{REMOTE_TRACE}-{REMOTE_SPAN}-01"})
    assert r.status_code == 200 and r.headers["x-trace-id"] == REMOTE_TRACE
    spans = by_name(exporter)
    server = spans["POST /search"]
    assert format(server.context.trace_id, "032x") == REMOTE_TRACE
    assert format(server.parent.span_id, "016x") == REMOTE_SPAN
    assert server.attributes["http.response.status_code"] == 200
    assert parent_id(spans["rag.search"]) == server.context.span_id


def test_streamed_answer_is_traced_to_the_end_and_probes_are_not(searcher):
    svc, exporter = traced(searcher)
    client = TestClient(create_app(svc))
    assert client.get("/healthz").status_code == 200
    assert client.get("/metrics").status_code == 200
    assert exporter.get_finished_spans() == ()
    r = client.post("/ask", json={"query": "How do I record topics?", "mode": "hybrid"})
    assert r.status_code == 200 and "event: done" in r.text
    spans = by_name(exporter)
    assert set(spans) == {"POST /ask", "rag.ask", "rag.search", "rag.embed", "rag.retrieve",
                          "rag.generate"}
    server, ask = spans["POST /ask"], spans["rag.ask"]
    assert parent_id(ask) == server.context.span_id
    # the server span stays open until the last byte of the stream
    assert server.end_time >= ask.end_time >= spans["rag.generate"].end_time
    assert r.headers["x-trace-id"] == format(server.context.trace_id, "032x")


def test_rejected_requests_are_traced_with_their_status(searcher):
    cfg = dataclasses.replace(Settings(), rate_limit=RateLimitSettings(rate_per_s=0.001,
                                                                       burst=1))
    svc, exporter = traced(searcher, cfg)
    client = TestClient(create_app(svc))
    body = {"query": "ros2 bag record", "mode": "bm25"}
    assert client.post("/search", json=body).status_code == 200
    exporter.clear()
    assert client.post("/search", json=body).status_code == 429
    spans = exporter.get_finished_spans()
    assert [s.name for s in spans] == ["POST /search"]
    assert spans[0].attributes["http.response.status_code"] == 429


def test_sampling_drops_local_traces_but_keeps_sampled_callers(searcher):
    svc, exporter = traced(searcher, sample_ratio=0.0)
    client = TestClient(create_app(svc))
    body = {"query": "ros2 bag record", "mode": "bm25"}
    assert client.post("/search", json=body).status_code == 200
    assert exporter.get_finished_spans() == ()
    client.post("/search", json={**body, "k": 2},
                headers={"traceparent": f"00-{REMOTE_TRACE}-{REMOTE_SPAN}-01"})
    assert {s.name for s in exporter.get_finished_spans()} == {"POST /search", "rag.search",
                                                               "rag.retrieve"}
