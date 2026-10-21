"""Tenant context.

Rule: the tenant is decided by the server (from the API key or the inbound channel number),
passed to the graph as immutable runtime *context*, and closed over by every tool. The LLM never
sees a `tenant_id` parameter, so a prompt injection cannot switch tenants.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


class TenantIsolationError(PermissionError):
    pass


@dataclass(frozen=True)
class Tenant:
    tenant_id: str
    name: str
    trades: tuple[str, ...]
    auto_approve_bookings: bool = False
    tone: str = "friendly"
    facts: dict[str, Any] = field(default_factory=dict, hash=False, compare=False)

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> Tenant:
        return cls(
            tenant_id=row["tenant_id"],
            name=row["name"],
            trades=tuple(row["trades"]),
            auto_approve_bookings=row["auto_approve_bookings"],
            tone=row["tone"],
            facts={k: v for k, v in row.items()
                   if k not in {"tenant_id", "name", "trades", "auto_approve_bookings", "tone"}},
        )


def assert_same_tenant(expected: str, row: dict[str, Any] | None) -> None:
    if row is not None and row.get("tenant_id") != expected:
        raise TenantIsolationError(f"row belongs to {row.get('tenant_id')}, not {expected}")
