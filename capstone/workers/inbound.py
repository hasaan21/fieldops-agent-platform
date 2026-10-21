"""Channel-agnostic inbound processing, shared by the Celery task and the inline dev path."""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any, Protocol

from capstone.agents.job_intake.responder import TemplateResponder
from capstone.workers.channels.base import InboundMessage, Sender
from capstone.workers.idempotency import IdempotencyStore

log = logging.getLogger(__name__)


class AgentRuntime(Protocol):
    crm: Any

    def handle_message(self, tenant_id: str, channel: str, sender: str, text: str,
                       message_id: str | None = None) -> dict[str, Any]: ...


def handle_inbound(msg: InboundMessage, runtime: AgentRuntime, idem: IdempotencyStore,
                   senders: Mapping[str, Sender]) -> dict[str, Any]:
    if not idem.claim(msg.idempotency_key):
        log.info("duplicate delivery %s", msg.idempotency_key)
        return {"status": "duplicate"}
    try:
        result = runtime.handle_message(msg.tenant_id, msg.channel, msg.sender, msg.text,
                                        message_id=msg.message_id)
        if result["status"] == "pending_review":
            tenant = runtime.crm.tenant(msg.tenant_id)
            reply = TemplateResponder().respond(tenant=tenant, channel=msg.channel,
                                                outcome="pending_approval", message=msg.text,
                                                facts={})
        else:
            reply = result.get("reply") or ""
        sender = senders.get(msg.channel)
        if sender and reply and msg.channel != "voice":
            sid = sender.send(to=msg.sender, text=reply, from_=msg.recipient,
                              thread_ref=msg.thread_ref or msg.message_id)
            result["sent_id"] = sid
        result["reply_sent"] = reply
        return result
    except Exception:
        idem.release(msg.idempotency_key)  # allow the retry to process it
        raise
