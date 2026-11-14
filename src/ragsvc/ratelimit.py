"""Token-bucket rate limiter, one bucket per client.

A bucket holds up to `burst` tokens and refills at `rate_per_s`. A request takes one token;
with none left it is rejected and told how long to wait (the HTTP layer turns that into
429 + Retry-After). Buckets are created on first use and the least recently used ones are
dropped beyond `max_clients`, so memory stays bounded however many client ids show up.

State is per process: several replicas behind a load balancer would each allow the full
rate. Sharing the buckets (e.g. in Redis) is the fix and is out of scope here.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class Decision:
    allowed: bool
    retry_after_s: float = 0.0
    remaining: float = 0.0


class RateLimiter:
    def __init__(self, rate_per_s: float, burst: int, max_clients: int = 10_000,
                 clock: Callable[[], float] = time.monotonic):
        if rate_per_s <= 0 or burst < 1:
            raise ValueError("rate_per_s must be > 0 and burst >= 1")
        self.rate, self.burst, self.max_clients, self.clock = rate_per_s, burst, max_clients, clock
        self._buckets: OrderedDict[str, tuple[float, float]] = OrderedDict()  # tokens, updated
        self._lock = threading.Lock()
        self.allowed = 0
        self.rejected = 0

    def check(self, client: str, cost: float = 1.0) -> Decision:
        now = self.clock()
        with self._lock:
            tokens, updated = self._buckets.get(client, (float(self.burst), now))
            tokens = min(float(self.burst), tokens + (now - updated) * self.rate)
            if tokens >= cost:
                tokens -= cost
                decision = Decision(True, 0.0, tokens)
                self.allowed += 1
            else:
                decision = Decision(False, (cost - tokens) / self.rate, tokens)
                self.rejected += 1
            self._buckets[client] = (tokens, now)
            self._buckets.move_to_end(client)
            while len(self._buckets) > self.max_clients:
                self._buckets.popitem(last=False)
        return decision
