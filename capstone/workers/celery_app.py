"""Celery app and tasks.

    uv run celery -A capstone.workers.celery_app worker --loglevel=INFO --concurrency=4

Reliability settings:
- acks_late + reject_on_worker_lost: a crash mid-task re-queues it (at-least-once), and
  idempotency keys make the effects exactly-once.
- prefetch_multiplier=1: long LLM tasks don't pile up behind one busy worker.
- autoretry with exponential backoff + jitter for transient provider errors; after max retries
  the message goes to the dead-letter list for a human.
"""

from __future__ import annotations

import json
import logging
from functools import lru_cache
from typing import Any

from celery import Celery

from capstone.core.config import get_settings
from capstone.workers.channels.base import InboundMessage, Sender
from capstone.workers.idempotency import IdempotencyStore, MemoryIdempotency, RedisIdempotency
from capstone.workers.inbound import handle_inbound

log = logging.getLogger(__name__)
settings = get_settings()

app = Celery("fieldops", broker=settings.redis_url, backend=settings.redis_url)
app.conf.update(
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    worker_prefetch_multiplier=1,
    task_serializer="json",
    result_expires=3600,
    task_routes={"fieldops.process_inbound": {"queue": "inbound"}},
    task_default_queue="inbound",
)


@lru_cache
def _idempotency() -> IdempotencyStore:
    try:
        store = RedisIdempotency(settings.redis_url)
        store.r.ping()
        return store
    except Exception:
        log.warning("redis unavailable: using in-memory idempotency (dev only)")
        return MemoryIdempotency()


@lru_cache
def _senders() -> dict[str, Sender]:
    senders: dict[str, Sender] = {}
    if settings.twilio_account_sid:
        from capstone.workers.channels.twilio import TwilioSender

        tw = TwilioSender(settings.twilio_account_sid, settings.twilio_auth_token)
        senders |= {"sms": tw, "whatsapp": tw}
    if settings.nylas_api_key:
        from capstone.workers.channels.email import NylasSender

        senders["email"] = NylasSender(settings.nylas_api_key, settings.nylas_grant_id)
    return senders


def process(payload: dict[str, Any]) -> dict[str, Any]:
    from capstone.app.runtime import get_runtime

    return handle_inbound(InboundMessage(**payload), get_runtime(), _idempotency(), _senders())


@app.task(name="fieldops.process_inbound", bind=True, autoretry_for=(Exception,),
          retry_backoff=2, retry_backoff_max=120, retry_jitter=True, max_retries=5)
def process_inbound(self, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return process(payload)
    except Exception:
        if self.request.retries >= self.max_retries:
            try:
                import redis

                redis.Redis.from_url(settings.redis_url).rpush(
                    "fieldops:dead_letter", json.dumps(payload))
            except Exception:
                log.exception("dead-letter push failed for %s", payload.get("message_id"))
        raise
