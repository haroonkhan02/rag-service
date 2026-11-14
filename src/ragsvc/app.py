"""HTTP API (FastAPI).

    GET  /healthz    the process is up
    GET  /readyz     the index and models are loaded (503 until then)
    POST /search     {"query", "mode"?, "k"?} -> ranked passages, per-stage timings, cache
    POST /ask        same body + "stream"? -> a cited answer; Server-Sent Events by default
                     (events: passages, token, done | refusal | error), JSON with stream=false
    GET  /metrics    counters, cache statistics, queue depth

Clients are identified by the `X-API-Key` header, else by address, and rate limited per
client (429 + Retry-After). If the environment variable RAGSVC_API_KEYS holds a
comma-separated list of keys, a valid key is required (401 otherwise). When the search
queue is full the service answers 503 + Retry-After rather than queueing without bound.
"""

from __future__ import annotations

import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from ragsvc.ratelimit import RateLimiter
from ragsvc.service import Overloaded, RagService, passages_payload

API_KEYS_ENV = "RAGSVC_API_KEYS"


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str | None = None
    k: int | None = Field(default=None, ge=1)


class AskRequest(SearchRequest):
    stream: bool = True


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def create_app(service: RagService, api_keys: frozenset[str] | None = None,
               limiter: RateLimiter | None = None) -> FastAPI:
    settings = service.settings
    if api_keys is None:
        api_keys = frozenset(k.strip() for k in os.environ.get(API_KEYS_ENV, "").split(",")
                             if k.strip())
    if limiter is None and settings.rate_limit.enabled:
        limiter = RateLimiter(settings.rate_limit.rate_per_s, settings.rate_limit.burst)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        service.close()

    app = FastAPI(title="RAG Service", version="0.1.0", lifespan=lifespan)
    app.state.service, app.state.limiter = service, limiter

    def client_id(request: Request) -> str:
        key = request.headers.get("x-api-key")
        if api_keys:
            if key not in api_keys:
                raise HTTPException(401, "missing or invalid API key")
            return f"key:{key}"
        if key:
            return f"key:{key}"
        return f"ip:{request.client.host if request.client else 'unknown'}"

    def guard(request: Request) -> str:
        client = client_id(request)
        if limiter is not None:
            decision = limiter.check(client)
            if not decision.allowed:
                service.metrics.inc("rate_limited")
                raise HTTPException(429, "rate limit exceeded", headers={
                    "Retry-After": str(max(1, round(decision.retry_after_s + 0.5)))})
        return client

    def check(body: SearchRequest) -> None:
        if len(body.query) > settings.server.max_query_chars:
            raise HTTPException(422, f"query longer than {settings.server.max_query_chars} "
                                "characters")
        if not body.query.strip():
            raise HTTPException(422, "query is empty")
        try:
            service.resolve(body.mode, body.k)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.exception_handler(Overloaded)
    async def overloaded(request: Request, exc: Overloaded):
        return JSONResponse({"detail": "service overloaded, retry shortly"}, status_code=503,
                            headers={"Retry-After": "1"})

    @app.get("/healthz")
    async def healthz():
        return {"status": "ok"}

    @app.get("/readyz")
    async def readyz():
        manifest = service.searcher.bundle.manifest
        return {"status": "ready", "chunks": len(service.searcher.chunks),
                "index_sha256": manifest.get("sha256", {}),
                "generator": service.generator.name}

    @app.get("/metrics")
    async def metrics():
        out = service.stats()
        if limiter is not None:
            out["rate_limit"] = {"allowed": limiter.allowed, "rejected": limiter.rejected}
        return out

    @app.post("/search")
    async def search(body: SearchRequest, client: str = Depends(guard)):
        check(body)
        out = await service.search(body.query.strip(), body.mode, body.k)
        return {"query": body.query, "cache": out.cache, "similarity": out.similarity,
                "matched_query": out.matched_query,
                "timings_ms": {n: round(v, 2) for n, v in out.timings_ms.items()},
                "passages": passages_payload(out.hits)}

    @app.post("/ask")
    async def ask(body: AskRequest, client: str = Depends(guard)):
        check(body)
        events = service.ask(body.query.strip(), body.mode, body.k)
        if not body.stream:
            passages, final = [], None
            async for ev in events:
                if ev["type"] == "passages":
                    passages = ev["passages"]
                elif ev["type"] in ("done", "refusal"):
                    final = ev
            if final["type"] == "refusal":
                return JSONResponse({"detail": final["message"], "refused": True},
                                    status_code=422)
            return {**{n: v for n, v in final.items() if n != "type"}, "passages": passages}
        # run up to the first event before answering, so an overload or a bad request
        # still becomes a proper HTTP status instead of a broken stream
        first = await anext(events)

        async def body_stream() -> AsyncIterator[str]:
            try:
                yield sse(first["type"], first)
                async for ev in events:
                    yield sse(ev["type"], ev)
            except Exception as exc:  # the status line is already sent: report in-band
                service.metrics.inc("stream_errors")
                yield sse("error", {"type": "error", "message": type(exc).__name__})

        return StreamingResponse(body_stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache",
                                          "X-Accel-Buffering": "no"})

    return app
