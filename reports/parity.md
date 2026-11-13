# Parity with robotics-docs-rag

61 answerable questions, section chunking, 3680 chunks. "ref" = published results of robotics-docs-rag (Chroma HNSW); "svc" = this service (exact NumPy search). Same relevance = identical relevant/not pattern in the top 10; same top-5 = identical document list.

| Mode | MRR@10 svc | MRR@10 ref | Hit@1 svc | Hit@1 ref | Hit@5 svc | Hit@5 ref | nDCG@10 svc | nDCG@10 ref | Same relevance | Same top-5 |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 | 0.506 | 0.506 | 0.377 | 0.377 | 0.721 | 0.721 | 0.517 | 0.517 | 61/61 | 60/61 |
| dense | 0.596 | 0.596 | 0.459 | 0.459 | 0.738 | 0.738 | 0.603 | 0.603 | 61/61 | 61/61 |
| hybrid | 0.574 | 0.574 | 0.426 | 0.426 | 0.787 | 0.787 | 0.609 | 0.609 | 60/61 | 56/61 |
| hybrid_rerank | 0.690 | 0.690 | 0.590 | 0.590 | 0.803 | 0.803 | 0.692 | 0.692 | 61/61 | 61/61 |

Median / p95 time per stage in ms (warm, one query at a time, arm64, embedding device `mps`):

| Mode | Stage | Median | p95 |
|---|---|---|---|
| bm25 | bm25 | 1.21 | 1.81 |
| bm25 | total | 1.21 | 1.81 |
| dense | dense | 0.16 | 0.19 |
| dense | embed | 4.66 | 10.11 |
| dense | total | 4.81 | 10.29 |
| hybrid | bm25 | 1.53 | 2.13 |
| hybrid | dense | 0.19 | 0.20 |
| hybrid | embed | 5.00 | 5.22 |
| hybrid | total | 6.73 | 7.43 |
| hybrid_rerank | bm25 | 1.79 | 2.53 |
| hybrid_rerank | dense | 0.21 | 0.21 |
| hybrid_rerank | embed | 5.45 | 5.96 |
| hybrid_rerank | rerank | 146.57 | 150.01 |
| hybrid_rerank | total | 152.59 | 158.23 |

Chunk file byte-identical to the reference index: yes.
