# Tracing

Measured from the service's own OpenTelemetry spans (Darwin arm64, models on `mps`). The service runs in the measuring process and is called through its ASGI app, one request at a time, so there is no network hop and no queueing; times are milliseconds.

## Where a search spends its time

`/search`, hybrid + rerank, default caches. The 61 evaluation questions are sent once (all misses), sent again (exact hits), then sent in their paraphrased wording (a semantic hit when the similarity reaches 0.96, else a miss). Medians per cache outcome.

| Cache outcome | Requests | Request p50 | p95 | rag.search | rag.embed | rag.retrieve | BM25 | Dense | Rerank | Wait for worker | HTTP layer |
|---|---|---|---|---|---|---|---|---|---|---|---|
| miss | 99 | 161.57 | 168.85 | 161.28 | 5.66 | 155.98 | 1.91 | 0.21 | 153.01 | 0.15 | 0.32 |
| semantic | 23 | 5.72 | 6.42 | 5.47 | 5.40 | - | - | - | - | - | 0.28 |
| exact | 61 | 0.18 | 0.25 | 0.01 | - | - | - | - | - | - | 0.17 |

`rag.search`, `rag.embed` and `rag.retrieve` are span durations; BM25, dense, rerank and the wait for a model worker are attributes of `rag.retrieve`. The HTTP layer is the server span minus `rag.search`: request parsing, validation, the rate limiter and JSON encoding.

## Where an answer spends its time

`/ask` without streaming, 10 new questions, generator `mlx:Qwen2.5-7B-Instruct-4bit`.

| Request p50 | rag.search p50 | rag.generate p50 | Time to first token p50 | Input tokens p50 | Output tokens p50 |
|---|---|---|---|---|---|
| 2,604 | 174 | 2,430 | 1,370 | 1,232 | 32 |

The slowest of these requests as a trace:

```text
POST /ask                  5046.5 ms   status_code=200
  rag.ask                  5046.2 ms   cache=miss, mode=hybrid_rerank, k=5
    rag.search              182.0 ms   cache=miss, mode=hybrid_rerank, k=5
      rag.embed              19.3 ms
      rag.retrieve          162.5 ms   rerank_ms=160.2, queue_wait_ms=0.2
    rag.generate           4864.1 ms   model=mlx:Qwen2.5-7B-Instruct-4bit, input_tokens=1183, output_tokens=96, ttft_ms=1313.7
```

## What tracing costs

The same `/search` requests with tracing off and on, in 4 alternating blocks per arm. With tracing on, spans are created, batched and handed to an exporter that discards them, so this is the cost inside the service without the cost of shipping spans to a collector.

| Path | Requests per arm | Spans per request | p50 off | p50 on | Difference | Relative | Block medians off | Block medians on |
|---|---|---|---|---|---|---|---|---|
| exact cache hit | 244 | 2 | 0.219 | 0.253 | +0.034 | +15.6 % | 0.217 to 0.221 | 0.250 to 0.256 |
| hybrid, no cache | 244 | 3 | 7.881 | 7.971 | +0.090 | +1.1 % | 7.776 to 8.024 | 7.842 to 8.093 |
| hybrid + rerank, no cache | 244 | 3 | 161.737 | 162.003 | +0.266 | +0.2 % | 161.367 to 161.982 | 161.771 to 162.689 |

The last two columns give the range of the per-block medians. Where the two ranges overlap, the difference is not distinguishable from run-to-run variation.
