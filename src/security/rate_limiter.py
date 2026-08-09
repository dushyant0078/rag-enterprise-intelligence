"""Token-bucket rate limiting, per API key. Real, deterministic, and
testable with an injectable clock -- not a placeholder.
"""

from __future__ import annotations

import time
from collections.abc import Callable


class _Bucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, tokens: float, last_refill: float) -> None:
        self.tokens = tokens
        self.last_refill = last_refill


class TokenBucketRateLimiter:
    def __init__(
        self,
        capacity: int = 60,
        refill_per_second: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.capacity = capacity
        self.refill_per_second = refill_per_second
        self._clock = clock
        self._buckets: dict[str, _Bucket] = {}

    def allow(self, key: str) -> bool:
        now = self._clock()
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.capacity), last_refill=now)
            self._buckets[key] = bucket

        elapsed = now - bucket.last_refill
        bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.refill_per_second)
        bucket.last_refill = now

        if bucket.tokens < 1.0:
            return False
        bucket.tokens -= 1.0
        return True
