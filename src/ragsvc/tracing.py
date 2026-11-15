"""Request tracing with OpenTelemetry, behind a small wrapper.

One trace per request, with a span per step:

    POST /ask                      server span; joins the caller's trace via `traceparent`
      rag.ask                      cache outcome, tokens, cost
        rag.search                 mode, k, cache outcome
          rag.embed                query embedding, including time spent waiting for a worker
          rag.retrieve             BM25 + dense + rerank; stage times and queue wait as attributes
        rag.generate               model, tokens, time to first token, stop reason

Parents are passed explicitly instead of through OpenTelemetry's context variables. A
streamed answer is an async generator that Starlette resumes from another task, and a
context attached in one task cannot be detached in another, so "current span" tracking
breaks exactly on the endpoint that most needs tracing.

OpenTelemetry is an optional dependency (`pip install rag-service[otel]`). Without it, or
with tracing disabled, `Tracer.start` returns a no-op span and the service runs unchanged.
"""

from __future__ import annotations

from ragsvc.config import TracingSettings


class Span:
    """A no-op span; also the interface of a real one."""

    trace_id: str | None = None

    def set(self, **attrs) -> None:
        pass

    def child(self, name: str, **attrs) -> Span:
        return self

    def end(self, error: BaseException | None = None) -> None:
        pass

    def __enter__(self) -> Span:
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.end(exc)
        return False


NOOP = Span()


def _clean(attrs: dict) -> dict:
    """OpenTelemetry rejects None; dots are nicer than keyword-safe underscores."""
    return {k: v for k, v in attrs.items() if v is not None}


class _OtelSpan(Span):
    def __init__(self, tracer: Tracer, span):
        self._tracer, self._span, self._ended = tracer, span, False
        self.trace_id = format(span.get_span_context().trace_id, "032x")

    def set(self, **attrs) -> None:
        self._span.set_attributes(_clean(attrs))

    def child(self, name: str, **attrs) -> Span:
        return self._tracer.start(name, self, **attrs)

    def end(self, error: BaseException | None = None) -> None:
        if self._ended:
            return
        self._ended = True
        if error is not None:
            from opentelemetry.trace import Status, StatusCode

            # a client that goes away mid-stream is not a server error
            if isinstance(error, Exception):
                self._span.record_exception(error)
                self._span.set_status(Status(StatusCode.ERROR, type(error).__name__))
            else:
                self._span.set_attribute("rag.cancelled", True)
        self._span.end()


class Tracer:
    def __init__(self, otel_tracer=None, provider=None):
        self._tracer, self._provider = otel_tracer, provider

    @property
    def enabled(self) -> bool:
        return self._tracer is not None

    def start(self, name: str, parent: Span | None = None, traceparent: str | None = None,
              kind: str = "internal", **attrs) -> Span:
        """Start a span under `parent`, or under a remote W3C `traceparent`, or as a root."""
        if self._tracer is None:
            return NOOP
        from opentelemetry import trace
        from opentelemetry.context import Context

        if isinstance(parent, _OtelSpan):
            context = trace.set_span_in_context(parent._span, Context())
        elif traceparent:
            from opentelemetry.trace.propagation.tracecontext import (
                TraceContextTextMapPropagator,
            )

            context = TraceContextTextMapPropagator().extract({"traceparent": traceparent},
                                                              Context())
        else:
            context = Context()
        span_kind = trace.SpanKind.SERVER if kind == "server" else trace.SpanKind.INTERNAL
        span = self._tracer.start_span(name, context=context, kind=span_kind,
                                       attributes=_clean(attrs))
        return _OtelSpan(self, span)

    def flush(self) -> None:
        if self._provider is not None:
            self._provider.force_flush()

    def shutdown(self) -> None:
        if self._provider is not None:
            self._provider.shutdown()


def setup_tracing(cfg: TracingSettings, exporter=None, batch: bool = True) -> Tracer:
    """Build a tracer from the settings. `exporter` overrides the configured one (tests,
    studies). The provider is private to the returned tracer: nothing global is installed.
    """
    if not cfg.enabled and exporter is None:
        return Tracer()
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
    except ImportError as exc:
        raise RuntimeError("tracing.enabled needs OpenTelemetry: "
                           "pip install 'rag-service[otel]'") from exc
    from opentelemetry.sdk.trace.export import (
        BatchSpanProcessor,
        ConsoleSpanExporter,
        SimpleSpanProcessor,
    )
    from opentelemetry.sdk.trace.sampling import ParentBased, TraceIdRatioBased

    if exporter is None:
        if cfg.exporter == "console":
            exporter = ConsoleSpanExporter()
        else:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

            # an empty endpoint lets the SDK read OTEL_EXPORTER_OTLP_ENDPOINT or use
            # http://localhost:4318
            exporter = OTLPSpanExporter(endpoint=cfg.endpoint or None)
    provider = TracerProvider(
        resource=Resource.create({"service.name": cfg.service_name}),
        # the caller's sampling decision wins; otherwise keep `sample_ratio` of the traces
        sampler=ParentBased(TraceIdRatioBased(cfg.sample_ratio)),
    )
    # batching exports off the request path; the simple processor exports inline
    provider.add_span_processor((BatchSpanProcessor if batch else SimpleSpanProcessor)(exporter))
    return Tracer(provider.get_tracer("ragsvc"), provider)


class TraceMiddleware:
    """ASGI middleware: a server span per API request, ended when the response body ends.

    Pure ASGI (not BaseHTTPMiddleware) so the span covers a streamed answer to its last
    byte and also covers requests rejected before the route runs (401, 429, 503). The span
    is handed to the route as `request.state.span`; the trace id goes back to the client in
    `X-Trace-Id`, so a slow request can be looked up from a client log.
    """

    SKIP = frozenset({"/healthz", "/readyz", "/metrics"})

    def __init__(self, app, tracer: Tracer):
        self.app, self.tracer = app, tracer

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not self.tracer.enabled or scope["path"] in self.SKIP:
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        span = self.tracer.start(
            f"{scope['method']} {scope['path']}", traceparent=headers.get("traceparent"),
            kind="server", **{"http.request.method": scope["method"],
                              "url.path": scope["path"]})
        scope.setdefault("state", {})["span"] = span

        async def traced_send(message):
            if message["type"] == "http.response.start":
                span.set(**{"http.response.status_code": message["status"]})
                if span.trace_id:
                    message = {**message, "headers": [
                        *message.get("headers", []),
                        (b"x-trace-id", span.trace_id.encode("ascii"))]}
            await send(message)

        try:
            await self.app(scope, receive, traced_send)
        except BaseException as exc:
            span.end(exc)
            raise
        span.end()
