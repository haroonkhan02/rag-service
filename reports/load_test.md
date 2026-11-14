# Load test

Service and client on the same machine (Darwin arm64), one server process with one model worker thread, embedder and cross-encoder on `mps`. `/search`, 162 distinct questions unless noted. Closed loop: each client sends its next request when the previous one returns. Latency is measured at the client in milliseconds, over successful requests.

## Search pipeline, caches off

| Mode | Clients | Requests | Throughput (req/s) | p50 | p95 | p99 |
|---|---|---|---|---|---|---|
| hybrid_rerank | 1 | 324/324 | 7.1 | 154.4 | 159.8 | 160.8 |
| hybrid_rerank | 4 | 324/324 | 7.1 | 571.2 | 633.8 | 637.4 |
| hybrid_rerank | 16 | 324/324 | 7.1 | 2264.2 | 2446.4 | 2489.5 |
| hybrid | 1 | 324/324 | 115.0 | 8.2 | 9.7 | 15.6 |
| hybrid | 4 | 324/324 | 127.1 | 30.3 | 33.9 | 57.2 |
| hybrid | 16 | 324/324 | 127.7 | 120.1 | 127.2 | 204.1 |

## Overload, caches off (hybrid_rerank)

Queue limit 64 waiting searches; beyond that the service answers 503 immediately. Rejected clients move straight on to their next question, so the same question is often in flight twice and the second request shares the first one's result.

| Clients | Requests | 200 (computed) | 200 (shared) | 503 | Searches/s | p50 | p95 | p99 |
|---|---|---|---|---|---|---|---|---|
| 32 | 648 | 648 | 0 | 0 | 7.1 | 4473.3 | 4768.6 | 4828.3 |
| 128 | 648 | 124 | 184 | 340 | 7.1 | 4714.9 | 9084.9 | 9298.4 |

## Caching on a skewed workload (hybrid_rerank, 8 clients)

1000 requests drawn from the 66 evaluation questions with Zipf popularity (s = 1.1, seed 0); 30% of them use the paraphrased wording (105 distinct wordings in total). "shared" = joined an identical search already in flight.

| Cache | Throughput (req/s) | p50 | p95 | p99 | miss | exact | semantic | shared |
|---|---|---|---|---|---|---|---|---|
| no cache | 9.7 | 882.3 | 1125.7 | 1195.7 | 775 | 0 | 0 | 225 |
| exact | 67.5 | 0.7 | 983.0 | 1127.3 | 105 | 866 | 0 | 29 |
| exact + semantic (0.96) | 74.8 | 1.0 | 799.5 | 1046.3 | 92 | 764 | 77 | 67 |

## Rate limit

One client sends 20 req/s for 10 s against a limit of 5 req/s with a burst of 20: 69 of 200 accepted (token bucket predicts 70), 131 answered 429 with Retry-After.

## Streamed answers (`/ask`, local model)

Generator: `mlx-community/Qwen2.5-7B-Instruct-4bit` through mlx-lm on the same machine, greedy decoding, so these numbers describe this laptop and model, not Claude. Time to first token includes retrieval and reading the prompt (a median of 1232 input tokens); output tokens/s is over the whole generation, prompt included. The model generates one answer at a time, so with four clients most of the wait is the queue.

| Run | Requests | TTFT p50 | TTFT p95 | Total p50 | Total p95 | Output tokens/s | Output tokens (median) |
|---|---|---|---|---|---|---|---|
| One client, new questions | 20/20 | 1,460 | 1,673 | 2,993 | 5,083 | 15.3 | 39.5 |
| Same questions again (answer cache) | 20/20 | 0.5 | 0.6 | 0.6 | 0.7 | - | - |
| Four clients, new questions | 20/20 | 14,110 | 21,944 | 18,399 | 24,214 | 18.8 | 83.0 |

Of the 20 answers to new questions: 2 abstained, 1 had no citation, 0 cited a passage number that does not exist, 0 hit the token limit. Answer quality is not judged here.
