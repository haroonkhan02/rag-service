"""HTTP layer through FastAPI's test client: status codes, SSE framing, limits."""

import dataclasses
import json

import pytest
from fastapi.testclient import TestClient

from ragsvc.app import create_app
from ragsvc.config import RateLimitSettings, ServerSettings, Settings
from ragsvc.generate import ExtractiveGenerator
from ragsvc.ratelimit import RateLimiter
from ragsvc.service import Overloaded, RagService


def make_client(searcher, cfg=None, **kw):
    service = RagService(searcher, ExtractiveGenerator(), cfg or Settings())
    return TestClient(create_app(service, **kw)), service


def parse_sse(text):
    events = []
    for block in text.strip().split("\n\n"):
        lines = block.split("\n")
        assert lines[0].startswith("event: ") and lines[1].startswith("data: ")
        events.append((lines[0][7:], json.loads(lines[1][6:])))
    return events


def test_health_ready_and_metrics(searcher):
    client, _ = make_client(searcher)
    assert client.get("/healthz").json() == {"status": "ok"}
    ready = client.get("/readyz").json()
    assert ready["status"] == "ready" and ready["chunks"] == len(searcher.chunks)
    assert ready["generator"] == "extractive"
    client.post("/search", json={"query": "ros2 bag record", "mode": "bm25"})
    m = client.get("/metrics").json()
    assert m["counters"]["search_requests"] == 1 and m["counters"]["search_cache_miss"] == 1
    assert m["search_cache"]["entries"] == 1 and m["rate_limit"]["allowed"] == 1
    client.post("/ask", json={"query": "ros2 bag record", "mode": "bm25", "stream": False})
    client.post("/ask", json={"query": "ros2 bag record", "mode": "bm25", "stream": False})
    usage = client.get("/metrics").json()["usage"]
    assert usage["answers_generated"] == 1 and usage["answers_from_cache"] == 1
    assert usage["cost_usd"] == 0.0  # the extractive generator has no price


def test_search_returns_passages_timings_and_cache_status(searcher):
    client, _ = make_client(searcher)
    body = {"query": "How do I record topics with ros2 bag?", "mode": "hybrid", "k": 2}
    r = client.post("/search", json=body)
    assert r.status_code == 200
    out = r.json()
    assert out["cache"] == "miss" and len(out["passages"]) == 2
    assert out["passages"][0]["doc_id"] == "Tutorials/Bags"
    assert {"bm25", "dense", "service_total"} <= set(out["timings_ms"])
    assert client.post("/search", json=body).json()["cache"] == "exact"


@pytest.mark.parametrize("body, fragment", [
    ({}, "query"),
    ({"query": ""}, "query"),
    ({"query": "   "}, "empty"),
    ({"query": "x" * 1001}, "longer"),
    ({"query": "q", "mode": "fuzzy"}, "unknown mode"),
    ({"query": "q", "k": 0}, "k"),
    ({"query": "q", "k": 500}, "k must be"),
])
def test_invalid_requests_are_rejected_with_422(searcher, body, fragment):
    client, service = make_client(searcher)
    for path in ("/search", "/ask"):
        r = client.post(path, json=body)
        assert r.status_code == 422 and fragment in r.text
    assert "search_requests" not in service.metrics.counters


def test_ask_streams_server_sent_events(searcher):
    client, _ = make_client(searcher)
    r = client.post("/ask", json={"query": "How do I record topics with ros2 bag?",
                                  "mode": "hybrid", "k": 2})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    events = parse_sse(r.text)
    names = [n for n, _ in events]
    assert names[0] == "passages" and names[-1] == "done" and "token" in names
    done = events[-1][1]
    assert "".join(d["text"] for n, d in events if n == "token").strip() == done["answer"]
    assert done["citations"] == [1] and done["model"] == "extractive"


def test_ask_without_streaming_returns_one_json_document(searcher):
    client, _ = make_client(searcher)
    body = {"query": "What do services return?", "mode": "hybrid", "k": 2, "stream": False}
    out = client.post("/ask", json=body).json()
    assert out["answer"].endswith("[1]") and len(out["passages"]) == 2
    assert out["cache"] == "miss" and "type" not in out
    assert client.post("/ask", json=body).json()["cache"] == "exact"


def test_rate_limit_returns_429_with_retry_after_per_client(searcher):
    cfg = dataclasses.replace(Settings(), rate_limit=RateLimitSettings(rate_per_s=0.5, burst=2))
    client, service = make_client(searcher, cfg)
    body = {"query": "ros2 bag record", "mode": "bm25"}
    codes = [client.post("/search", json=body).status_code for _ in range(3)]
    assert codes == [200, 200, 429]
    r = client.post("/search", json=body)
    assert r.headers["retry-after"] == "2" and r.json()["detail"] == "rate limit exceeded"
    # another client (another key) has its own bucket; health checks are never limited
    assert client.post("/search", json=body, headers={"X-API-Key": "b"}).status_code == 200
    assert client.get("/healthz").status_code == 200
    assert service.metrics.counters["rate_limited"] == 2


def test_rate_limit_can_be_disabled(searcher):
    cfg = dataclasses.replace(Settings(), rate_limit=RateLimitSettings(enabled=False))
    client, _ = make_client(searcher, cfg)
    codes = {client.post("/search", json={"query": "q", "mode": "bm25"}).status_code
             for _ in range(40)}
    assert codes == {200} and "rate_limit" not in client.get("/metrics").json()


def test_api_keys_are_enforced_when_configured(searcher):
    client, _ = make_client(searcher, api_keys=frozenset({"secret"}),
                            limiter=RateLimiter(100, 100))
    body = {"query": "ros2 bag record", "mode": "bm25"}
    assert client.post("/search", json=body).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.post("/search", json=body, headers={"X-API-Key": "secret"}).status_code == 200
    assert client.get("/healthz").status_code == 200


def test_overload_becomes_503_for_search_and_for_streams(searcher):
    cfg = dataclasses.replace(Settings(), server=ServerSettings(max_queue=0))
    client, service = make_client(searcher, cfg)

    async def full(*a, **kw):
        raise Overloaded("search queue is full")

    service._in_pool = full
    for path in ("/search", "/ask"):
        r = client.post(path, json={"query": "ros2 bag record", "mode": "bm25"})
        assert r.status_code == 503 and r.headers["retry-after"] == "1"


def test_error_after_the_stream_started_is_reported_in_band(searcher):
    class Failing:
        name = "failing"

        async def stream(self, question, hits):
            yield "partial "
            raise RuntimeError("model crashed")

    service = RagService(searcher, Failing(), Settings())
    client = TestClient(create_app(service))
    r = client.post("/ask", json={"query": "ros2 bag record", "mode": "bm25"})
    events = parse_sse(r.text)
    assert r.status_code == 200 and events[-1] == ("error", {"type": "error",
                                                            "message": "RuntimeError"})
    assert len(service.answer_cache) == 0
