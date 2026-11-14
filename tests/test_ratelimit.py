import pytest

from ragsvc.ratelimit import RateLimiter


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_burst_then_reject_with_retry_after():
    clock = Clock()
    rl = RateLimiter(rate_per_s=2.0, burst=3, clock=clock)
    assert [rl.check("a").allowed for _ in range(3)] == [True, True, True]
    d = rl.check("a")
    assert not d.allowed and d.retry_after_s == pytest.approx(0.5)
    assert rl.allowed == 3 and rl.rejected == 1


def test_tokens_refill_at_the_configured_rate_up_to_burst():
    clock = Clock()
    rl = RateLimiter(rate_per_s=2.0, burst=3, clock=clock)
    for _ in range(3):
        rl.check("a")
    clock.t = 0.5  # one token back
    assert rl.check("a").allowed and not rl.check("a").allowed
    clock.t = 100.0  # long idle: capped at burst, not 200 tokens
    assert [rl.check("a").allowed for _ in range(4)] == [True, True, True, False]


def test_clients_have_separate_buckets():
    rl = RateLimiter(rate_per_s=1.0, burst=1, clock=Clock())
    assert rl.check("a").allowed and not rl.check("a").allowed
    assert rl.check("b").allowed


def test_bucket_table_is_bounded():
    rl = RateLimiter(rate_per_s=1.0, burst=1, max_clients=2, clock=Clock())
    for c in ("a", "b", "c"):
        rl.check(c)
    assert list(rl._buckets) == ["b", "c"]
    assert rl.check("a").allowed  # forgotten client starts with a full bucket


def test_invalid_settings_rejected():
    with pytest.raises(ValueError):
        RateLimiter(0, 1)
    with pytest.raises(ValueError):
        RateLimiter(1, 0)
