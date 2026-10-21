"""Exactly-once *effects* on top of at-least-once delivery.

Webhook providers retry (Twilio on timeouts, Nylas on non-2xx), and Celery with acks_late
redelivers on worker crash. Claim the message id atomically before processing; a duplicate sees
the claim and skips.
"""

from __future__ import annotations

import time
from typing import Protocol


class IdempotencyStore(Protocol):
    def claim(self, key: str, ttl_s: int = 86400) -> bool: ...

    def release(self, key: str) -> None: ...


class RedisIdempotency:
    def __init__(self, url: str) -> None:
        import redis

        self.r = redis.Redis.from_url(url)

    def claim(self, key: str, ttl_s: int = 86400) -> bool:
        return bool(self.r.set(key, "1", nx=True, ex=ttl_s))

    def release(self, key: str) -> None:
        self.r.delete(key)


class MemoryIdempotency:
    def __init__(self) -> None:
        self._keys: dict[str, float] = {}

    def claim(self, key: str, ttl_s: int = 86400) -> bool:
        now = time.monotonic()
        if key in self._keys and self._keys[key] > now:
            return False
        self._keys[key] = now + ttl_s
        return True

    def release(self, key: str) -> None:
        self._keys.pop(key, None)
