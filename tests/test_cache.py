import numpy as np
import pytest

from ragsvc.cache import QueryCache, normalize_query


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def unit(*xs):
    v = np.array(xs, dtype=np.float32)
    return v / np.linalg.norm(v)


def test_normalize_query_ignores_case_spacing_and_final_punctuation():
    assert normalize_query("  What is  a Node?? ") == "what is a node"
    assert normalize_query("ros2 topic echo /cmd_vel") == "ros2 topic echo /cmd_vel"


def test_exact_hit_and_scope_isolation():
    c = QueryCache()
    c.put(("hybrid", 5), "What is a node?", "A")
    hit = c.get_exact(("hybrid", 5), "what is a NODE")
    assert hit.value == "A" and hit.kind == "exact"
    assert c.get_exact(("bm25", 5), "What is a node?") is None
    assert c.get_exact(("hybrid", 3), "What is a node?") is None
    assert c.stats.exact_hits == 1 and c.stats.misses == 2


def test_entries_expire_after_ttl():
    clock = Clock()
    c = QueryCache(ttl_s=10, semantic_threshold=0.9, clock=clock)
    c.put("s", "q", "A", unit(1, 0))
    clock.t = 9.9
    assert c.get_exact("s", "q").value == "A"
    clock.t = 10.0
    assert c.get_exact("s", "q") is None
    assert c.get_semantic("s", unit(1, 0)) is None
    assert c.stats.expired == 1 and len(c) == 0


def test_least_recently_used_entry_is_evicted():
    c = QueryCache(max_entries=2)
    c.put("s", "a", 1)
    c.put("s", "b", 2)
    c.get_exact("s", "a")  # a is now the most recently used
    c.put("s", "c", 3)
    assert c.get_exact("s", "b") is None
    assert c.get_exact("s", "a").value == 1 and c.get_exact("s", "c").value == 3
    assert c.stats.evictions == 1


def test_semantic_hit_needs_similarity_above_threshold():
    c = QueryCache(semantic_threshold=0.9)
    c.put("s", "how do I record a bag", "BAG", unit(1, 0, 0))
    c.put("s", "what is a node", "NODE", unit(0, 1, 0))
    hit = c.get_semantic("s", unit(1, 0.1, 0))
    assert hit.value == "BAG" and hit.kind == "semantic"
    assert hit.matched_query == "how do I record a bag"
    assert hit.similarity == pytest.approx(0.995, abs=1e-3)
    assert c.get_semantic("s", unit(1, 1, 0)) is None  # cos 0.707 to both
    assert c.get_semantic("other", unit(1, 0, 0)) is None


def test_semantic_lookup_sees_new_and_evicted_entries():
    c = QueryCache(max_entries=1, semantic_threshold=0.9)
    c.put("s", "a", "A", unit(1, 0))
    assert c.get_semantic("s", unit(1, 0)).value == "A"
    c.put("s", "b", "B", unit(0, 1))  # evicts a
    assert c.get_semantic("s", unit(1, 0)) is None
    assert c.get_semantic("s", unit(0, 1)).value == "B"


def test_semantic_disabled_without_threshold_and_entries_without_vectors_are_skipped():
    c = QueryCache()
    c.put("s", "a", "A", unit(1, 0))
    assert c.get_semantic("s", unit(1, 0)) is None
    c = QueryCache(semantic_threshold=0.5)
    c.put("s", "a", "A")  # stored without an embedding (e.g. BM25-only mode)
    assert c.get_semantic("s", unit(1, 0)) is None
    assert c.get_exact("s", "a").value == "A"
