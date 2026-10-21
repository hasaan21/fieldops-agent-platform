from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

Channel = Literal["sms", "whatsapp", "email", "voice", "web"]


class InboundMessage(BaseModel):
    """Channel-agnostic message, the only shape the agent runtime sees."""

    provider: str
    channel: Channel
    tenant_id: str
    sender: str
    recipient: str
    text: str
    message_id: str
    thread_ref: str | None = None
    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @property
    def idempotency_key(self) -> str:
        return f"inbound:{self.provider}:{self.message_id}"


class Sender(Protocol):
    def send(self, *, to: str, text: str, from_: str, thread_ref: str | None = None) -> str: ...


class NumberDirectory:
    """Maps the number/address a customer contacted to the tenant that owns it.

    In production this is a table: tenant_channels(tenant_id, channel, address).
    """

    def __init__(self, mapping: dict[str, str] | None = None) -> None:
        self.mapping = mapping or {
            "+441610000001": "t1", "whatsapp:+441610000001": "t1", "jobs@northern.example": "t1",
            "+441130000002": "t2", "whatsapp:+441130000002": "t2", "jobs@brightspark.example": "t2",
            "+441410000003": "t3", "whatsapp:+441410000003": "t3", "jobs@allcare.example": "t3",
        }

    def tenant_for(self, address: str) -> str:
        tenant = self.mapping.get(address.strip().lower()) or self.mapping.get(address.strip())
        if not tenant:
            raise LookupError(f"no tenant owns {address}")
        return tenant
