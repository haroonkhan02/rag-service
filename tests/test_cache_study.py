import numpy as np

from ragsvc.cache_study import THRESHOLDS, nearest, pick_threshold, quantiles, sweep


def test_nearest_can_ignore_the_query_itself():
    v = np.eye(3, dtype=np.float32)
    v[2] = [0.6, 0.8, 0.0]
    idx, sim = nearest(v, v)
    assert idx.tolist() == [0, 1, 2] and np.allclose(sim, 1.0)
    idx, sim = nearest(v, v, exclude_self=True)
    assert idx.tolist() == [2, 2, 1] and np.allclose(sim, [0.6, 0.8, 0.8])


def test_quantiles_of_a_small_sample():
    q = quantiles([0.5, 0.9, 0.7, 0.6, 0.8])
    assert (q["min"], q["median"], q["max"]) == (0.5, 0.7, 0.9)


def test_sweep_counts_hits_wrong_reuse_and_served_quality():
    good, bad = [True] + [False] * 9, [False] * 10
    rows = sweep(
        seed_ids=["q1", "q2"], para_ids=["q1", "q2"],
        para_idx=np.array([0, 0]),            # the q2 paraphrase is nearest to q1: wrong
        para_sim=np.array([0.97, 0.91]),
        near_sim=np.array([0.93, 0.70]), distinct_sim=np.array([0.60, 0.60]),
        fresh_rels={"q1": good, "q2": good},
        seed_rels_for={("q1", "q1"): good, ("q2", "q1"): bad},
        thresholds=(0.90, 0.92, 0.95, 0.98))
    low, mid, high, top = rows
    assert (low["paraphrase_correct_hits"], low["paraphrase_wrong_hits"]) == (1, 1)
    assert low["near_miss_wrong_hits"] == 1 and low["served_hit@5"] == 0.5
    assert (mid["paraphrase_wrong_hits"], mid["near_miss_wrong_hits"]) == (0, 1)
    assert mid["served_hit@5"] == 1.0  # the wrong reuse is gone: q2 is searched fresh
    assert high["paraphrase_hit_rate"] == 0.5 and high["near_miss_wrong_hits"] == 0
    assert top["paraphrase_hit_rate"] == 0.0 and top["served_mrr@10"] == 1.0
    # 0.95 is clean and clears the worst wrong question (0.93) by the margin
    assert pick_threshold(rows, highest_wrong=0.93) == 0.95
    assert pick_threshold(rows, highest_wrong=0.945) == 0.98
    assert pick_threshold(rows, highest_wrong=0.99) is None


def test_default_grid_covers_080_to_099():
    assert THRESHOLDS[0] == 0.80 and THRESHOLDS[-1] == 0.99 and len(THRESHOLDS) == 20
