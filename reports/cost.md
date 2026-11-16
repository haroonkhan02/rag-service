# Cost per query

Token counts are measured; dollar figures are estimates. The counts come from the 66 stored answers of `mlx-community/Qwen2.5-7B-Instruct-4bit` (hybrid_rerank, top 5 passages), counted with that model's tokenizer and including the system prompt. Claude tokenizes differently, writes different answers and bills its thinking as output tokens; measuring that requires an Anthropic API key. Read the dollar columns as an order of magnitude and the ratios as the result.

## Tokens per answer

| | Mean | p50 | p95 | Max |
|---|---|---|---|---|
| Prompt (input) tokens | 1,283 | 1,272 | 1,606 | 2,107 |
| Answer (output) tokens | 63 | 44 | 157 | 319 |

The local model spends 3.73 s of GPU time per answer on average, and answers one request at a time: about 965 answers per hour on this machine, at no price per token.

## The same token counts at Claude list prices (estimate)

| Model | Input $/MTok | Output $/MTok | Per answer | Per 1,000 answers | Share spent on the prompt |
|---|---|---|---|---|---|
| claude-opus-5-5 | 4 | 20 | $0.0064 | $6.39 | 80 % |
| claude-sonnet-5-5 | 2 | 10 | $0.0032 | $3.19 | 80 % |
| claude-haiku-5-5 | 0.1 | 0.5 | $0.0002 | $0.16 | 80 % |

The prompt dominates: the passages are most of what is paid for, and the answer is short. Thinking tokens, which are not in this estimate, would raise the output side.

## Passages in the prompt

`hybrid_rerank`, 61 answerable questions. "Relevant passage in the prompt" is retrieval Hit@k.

| Passages (k) | Prompt tokens, mean | Relative to k = 5 | Relevant passage in the prompt |
|---|---|---|---|
| 1 | 408 | 0.32x | 59.0 % |
| 3 | 842 | 0.66x | 77.0 % |
| 5 | 1,282 | 1.00x | 80.3 % |
| 8 | 1,944 | 1.52x | 85.2 % |
| 10 | 2,370 | 1.85x | 88.5 % |

Check: the offline count of the k = 5 prompts differs from the token count the model reported at generation time by at most 0 tokens over 66 prompts.

## Caching on a skewed workload

1000 requests drawn from the 66 evaluation questions with Zipf popularity (s = 1.1, seed 0), 30% in the paraphrased wording: the workload of the load test, replayed one request at a time. Stored answers stand in for the model, so only the number of answers that would have been generated is counted.

| Cache | Answers generated | From cache | of which semantic | Input tokens | Output tokens | claude-opus-5-5 | claude-sonnet-5-5 | claude-haiku-5-5 |
|---|---|---|---|---|---|---|---|---|
| no cache | 1000 | 0 | 0 | 1,241,840 | 55,440 | $6.08 | $3.04 | $0.15 |
| exact | 105 | 895 | 0 | 133,598 | 6,587 | $0.67 | $0.33 | $0.0167 |
| exact + semantic (0.96) | 90 | 910 | 133 | 113,984 | 5,668 | $0.57 | $0.28 | $0.0142 |

Dollar columns: estimated cost of the whole workload on each model. A paraphrase is charged what its original question cost. How much a cache saves depends entirely on how often questions repeat; this workload is built to repeat.
