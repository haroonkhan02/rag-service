"""In-process query cache with two ways to hit: the same question, or a close paraphrase.

    exact      key = (scope, normalised question); a dictionary lookup
    semantic   nearest cached question by cosine similarity of the query embeddings,
               reused only above a threshold

`scope` is everything else that changes the result (retrieval mode, k, the generator), so
a result computed with one configuration is never served for another. Entries expire after
a TTL and the least recently used one is evicted when the cache is full.

The semantic path trades correctness for hit rate: a threshold that is too low serves the
answer to a *different* question. `ragsvc-cache-study` measures that trade on paraphrases
and near-miss questions; the default threshold comes from there.
"""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_WS = re.compile(r"\s+")


def normalize_query(query: str) -> str:
    """Case, surrounding/repeated whitespace and trailing ?!. do not change a question."""
    return _WS.sub(" ", query).strip().rstrip("?!. ").lower()


@dataclass
class CacheHit:
    value: Any
    kind: str  # "exact" | "semantic"
    similarity: float = 1.0
    matched_query: str = ""


@dataclass
class _Entry:
    value: Any
    query: str
    expires: float
    vec: np.ndarray | None = None


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    misses: int = 0
    evictions: int = 0
    expired: int = 0

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class _Scope:
    keys: list[tuple] = field(default_factory=list)
    matrix: np.ndarray | None = None
    dirty: bool = True


class QueryCache:
    def __init__(self, max_entries: int = 2048, ttl_s: float = 3600.0,
                 semantic_threshold: float | None = None,
                 clock: Callable[[], float] = time.monotonic):
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self.max_entries, self.ttl_s = max_entries, ttl_s
        self.semantic_threshold = semantic_threshold
        self.clock = clock
        self.stats = CacheStats()
        self._entries: OrderedDict[tuple, _Entry] = OrderedDict()
        self._scopes: dict[Hashable, _Scope] = {}
        self._lock = threading.Lock()

    def __len__(self) -> int:
        return len(self._entries)

    # -- internals (call with the lock held) ---------------------------------------------
    def _drop(self, key: tuple) -> None:
        self._entries.pop(key, None)
        scope = self._scopes.get(key[0])
        if scope is not None:
            scope.dirty = True

    def _live(self, key: tuple) -> _Entry | None:
        entry = self._entries.get(key)
        if entry is None:
            return None
        if entry.expires <= self.clock():
            self._drop(key)
            self.stats.expired += 1
            return None
        return entry

    def _scope_matrix(self, scope_id: Hashable) -> _Scope:
        scope = self._scopes.setdefault(scope_id, _Scope())
        if scope.dirty:
            now = self.clock()
            scope.keys = [k for k, e in self._entries.items()
                          if k[0] == scope_id and e.vec is not None and e.expires > now]
            scope.matrix = (np.stack([self._entries[k].vec for k in scope.keys])
                            if scope.keys else None)
            scope.dirty = False
        return scope

    # -- public --------------------------------------------------------------------------
    def get_exact(self, scope: Hashable, query: str, count_miss: bool = True) -> CacheHit | None:
        key = (scope, normalize_query(query))
        with self._lock:
            entry = self._live(key)
            if entry is None:
                if count_miss:
                    self.stats.misses += 1
                return None
            self._entries.move_to_end(key)
            self.stats.exact_hits += 1
            return CacheHit(entry.value, "exact", 1.0, entry.query)

    def get_semantic(self, scope: Hashable, vec: np.ndarray) -> CacheHit | None:
        """Nearest cached question in this scope, if it is similar enough."""
        if self.semantic_threshold is None:
            return None
        with self._lock:
            sc = self._scope_matrix(scope)
            if sc.matrix is None:
                self.stats.misses += 1
                return None
            sims = sc.matrix @ np.asarray(vec, dtype=np.float32)
            best = int(np.argmax(sims))
            entry = self._live(sc.keys[best])
            if entry is None or float(sims[best]) < self.semantic_threshold:
                self.stats.misses += 1
                return None
            self._entries.move_to_end(sc.keys[best])
            self.stats.semantic_hits += 1
            return CacheHit(entry.value, "semantic", float(sims[best]), entry.query)

    def put(self, scope: Hashable, query: str, value: Any, vec: np.ndarray | None = None) -> None:
        key = (scope, normalize_query(query))
        if vec is not None:
            vec = np.asarray(vec, dtype=np.float32)
        with self._lock:
            self._entries[key] = _Entry(value, query, self.clock() + self.ttl_s, vec)
            self._entries.move_to_end(key)
            self._scopes.setdefault(scope, _Scope()).dirty = True
            while len(self._entries) > self.max_entries:
                old, _ = next(iter(self._entries.items()))
                self._drop(old)
                self.stats.evictions += 1

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
            self._scopes.clear()
