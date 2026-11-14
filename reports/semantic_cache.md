# Semantic cache: hit rate against wrong reuse

Cache seeded with the 66 evaluation questions. Query embeddings: `BAAI/bge-small-en-v1.5` (cosine similarity). 61 hand-written paraphrases (reuse of the source question = correct), 35 hand-written near misses (any reuse = wrong), and every evaluation question against the others (any reuse = wrong).

Similarity to the nearest cached question:

| Set | min | p10 | median | p90 | max |
|---|---|---|---|---|---|
| Paraphrase to its source | 0.801 | 0.882 | 0.951 | 0.976 | 0.991 |
| Near miss to nearest question | 0.611 | 0.754 | 0.844 | 0.937 | 0.950 |
| Evaluation question to nearest other | 0.521 | 0.576 | 0.697 | 0.766 | 0.851 |

Retrieval quality is what the 61 paraphrases are served (cached passages on a hit, a fresh `hybrid_rerank` search on a miss), judged against each paraphrase's own gold passages.

| Threshold | Paraphrase hits | Wrong (paraphrase) | Wrong (near miss, of 35) | Wrong (distinct, of 66) | Served Hit@5 | Served MRR@10 |
|---|---|---|---|---|---|---|
| 0.80 | 100.0% (61/61) | 0 | 27 | 2 | 0.803 | 0.690 |
| 0.81 | 98.4% (60/61) | 0 | 26 | 2 | 0.803 | 0.690 |
| 0.82 | 98.4% (60/61) | 0 | 23 | 2 | 0.803 | 0.690 |
| 0.83 | 98.4% (60/61) | 0 | 21 | 2 | 0.803 | 0.690 |
| 0.84 | 98.4% (60/61) | 0 | 19 | 2 | 0.803 | 0.690 |
| 0.85 | 96.7% (59/61) | 0 | 16 | 2 | 0.820 | 0.705 |
| 0.86 | 96.7% (59/61) | 0 | 15 | 0 | 0.820 | 0.705 |
| 0.87 | 96.7% (59/61) | 0 | 14 | 0 | 0.820 | 0.705 |
| 0.88 | 91.8% (56/61) | 0 | 12 | 0 | 0.820 | 0.716 |
| 0.89 | 88.5% (54/61) | 0 | 11 | 0 | 0.820 | 0.711 |
| 0.90 | 86.9% (53/61) | 0 | 10 | 0 | 0.820 | 0.711 |
| 0.91 | 85.2% (52/61) | 0 | 8 | 0 | 0.820 | 0.711 |
| 0.92 | 80.3% (49/61) | 0 | 8 | 0 | 0.820 | 0.712 |
| 0.93 | 73.8% (45/61) | 0 | 4 | 0 | 0.836 | 0.726 |
| 0.94 | 60.7% (37/61) | 0 | 3 | 0 | 0.836 | 0.746 |
| 0.95 | 54.1% (33/61) | 0 | 0 | 0 | 0.836 | 0.738 |
| 0.96 **(default)** | 37.7% (23/61) | 0 | 0 | 0 | 0.869 | 0.750 |
| 0.97 | 24.6% (15/61) | 0 | 0 | 0 | 0.869 | 0.753 |
| 0.98 | 6.6% (4/61) | 0 | 0 | 0 | 0.885 | 0.754 |
| 0.99 | 3.3% (2/61) | 0 | 0 | 0 | 0.885 | 0.754 |

No cache, paraphrases searched fresh: Hit@5 0.885, MRR@10 0.763. Original wording: Hit@5 0.803, MRR@10 0.690.

Default threshold: 0.96, the lowest value with no wrong reuse on any set and at least 0.01 above the most similar wrong question (0.9495). It is chosen on these same questions, so wrong reuse on new traffic can be higher than zero.

Closest near misses (the questions that set the threshold):

| Similarity | Near miss | Cached question |
|---|---|---|
| 0.950 | How do I record all topics into a bag with my own name for it? | How do I record only two topics into a bag with my own name for it? |
| 0.948 | What command do I use for a subsequent release of my package into a ROS distro? | What command do I use for the first release of my package into a ROS distro? |
| 0.943 | How can I give the turtlesim node a different namespace when I launch it? | How can I give the turtlesim node a different name when I launch it? |
| 0.937 | Where do I register my C++ node's executable so ros2 run can find it? | Where do I register my Python node's executable so ros2 run can find it? |
| 0.925 | How do I build all the packages in my workspace's src folder? | How do I install all the dependencies of the packages in my workspace's src folder? |
