"""A tenant-scoped CRM repository over the synthetic world.

Stands in for a real FSM platform's APIs. Every public method takes `tenant_id` first and only
returns rows from that tenant. tests/test_tenancy.py proves it.
"""

from __future__ import annotations

import itertools
import re
from datetime import datetime, timedelta
from threading import Lock
from typing import Any

from capstone.core.tenancy import Tenant, TenantIsolationError
from capstone.data.loader import load_table

SLOT_HOURS = [8, 10, 13, 15]
LEAD_DAYS = {"emergency": 0, "high": 1, "normal": 3, "low": 7}


def _norm_phone(p: str) -> str:
    digits = re.sub(r"\D", "", p or "")
    return "44" + digits[1:] if digits.startswith("0") else digits


class CRMRepository:
    def __init__(self, now: datetime | None = None) -> None:
        self.now = now or datetime(2026, 6, 1, 9, 0)
        self._tenants = {t["tenant_id"]: t for t in load_table("tenants")}
        self._customers = load_table("customers")
        self._sites = load_table("sites")
        self._assets = load_table("assets")
        self._jobs: list[dict[str, Any]] = [dict(j) for j in load_table("jobs")]
        self._booked: set[tuple[str, str, str]] = set()  # (tenant, engineer, slot)
        self._ids = itertools.count(1)
        self._lock = Lock()

    # -- tenants
    def tenant(self, tenant_id: str) -> Tenant:
        if tenant_id not in self._tenants:
            raise TenantIsolationError(f"unknown tenant {tenant_id}")
        return Tenant.from_row(self._tenants[tenant_id])

    # -- customers
    def find_customer(self, tenant_id: str, *, phone: str | None = None, email: str | None = None,
                      name: str | None = None) -> dict[str, Any] | None:
        self.tenant(tenant_id)
        for c in self._customers:
            if c["tenant_id"] != tenant_id:
                continue
            if phone and _norm_phone(c["phone"]) == _norm_phone(phone):
                return self._with_sites(c)
            if email and c["email"].lower() == email.lower():
                return self._with_sites(c)
            if name and c["name"].lower() == name.lower():
                return self._with_sites(c)
        return None

    def _with_sites(self, customer: dict[str, Any]) -> dict[str, Any]:
        sites = [s for s in self._sites if s["customer_id"] == customer["customer_id"]
                 and s["tenant_id"] == customer["tenant_id"]]
        assets = [a for a in self._assets if a["customer_id"] == customer["customer_id"]
                  and a["tenant_id"] == customer["tenant_id"]]
        return {**customer, "sites": sites, "assets": assets}

    def get_asset(self, tenant_id: str, asset_id: str) -> dict[str, Any]:
        self.tenant(tenant_id)
        for a in self._assets:
            if a["asset_id"] == asset_id:
                if a["tenant_id"] != tenant_id:
                    raise TenantIsolationError(f"asset {asset_id} not in {tenant_id}")
                return a
        raise KeyError(f"unknown asset {asset_id}")

    # -- scheduling
    def find_available_slots(self, tenant_id: str, category: str, urgency: str,
                             limit: int = 3) -> list[dict[str, Any]]:
        tenant = self.tenant(tenant_id)
        if category not in tenant.trades:
            return []
        start = (self.now + timedelta(days=LEAD_DAYS.get(urgency, 3))).replace(minute=0)
        engineers = [f"{tenant_id}-eng{i}" for i in range(1, 7)]
        slots: list[dict[str, Any]] = []
        for day in range(14):
            d = start + timedelta(days=day)
            if d.weekday() == 6 and urgency != "emergency":
                continue
            for h in SLOT_HOURS:
                ts = d.replace(hour=h)
                if ts <= self.now and urgency != "emergency":
                    continue
                for eng in engineers:
                    if (tenant_id, eng, ts.isoformat()) not in self._booked:
                        slots.append({"start": ts.isoformat(), "engineer_id": eng})
                        break
                if len(slots) >= limit:
                    return slots
        return slots

    def create_job(self, tenant_id: str, *, customer_id: str | None, site_id: str | None,
                   category: str, urgency: str, description: str, slot: dict[str, Any],
                   address: str | None = None, idempotency_key: str | None = None
                   ) -> dict[str, Any]:
        self.tenant(tenant_id)
        if customer_id and not any(c["customer_id"] == customer_id and c["tenant_id"] == tenant_id
                                   for c in self._customers):
            raise TenantIsolationError(f"customer {customer_id} not in {tenant_id}")
        with self._lock:
            if idempotency_key:
                for j in self._jobs:
                    if j.get("idempotency_key") == idempotency_key and j["tenant_id"] == tenant_id:
                        return j
            key = (tenant_id, slot["engineer_id"], slot["start"])
            if key in self._booked:
                raise ValueError("slot already taken")
            self._booked.add(key)
            job = {
                "job_id": f"{tenant_id}-new{next(self._ids):05d}",
                "tenant_id": tenant_id,
                "customer_id": customer_id,
                "site_id": site_id,
                "address": address,
                "category": category,
                "urgency": urgency,
                "description": description,
                "scheduled_start": slot["start"],
                "engineer_id": slot["engineer_id"],
                "status": "scheduled",
                "created_at": self.now.isoformat(),
                "idempotency_key": idempotency_key,
            }
            self._jobs.append(job)
            return job

    def get_open_jobs(self, tenant_id: str, customer_id: str) -> list[dict[str, Any]]:
        self.tenant(tenant_id)
        return [j for j in self._jobs if j["tenant_id"] == tenant_id
                and j["customer_id"] == customer_id and j["status"] == "scheduled"]

    def job_history(self, tenant_id: str, customer_id: str, limit: int = 5) -> list[dict[str, Any]]:
        self.tenant(tenant_id)
        rows = [j for j in self._jobs if j["tenant_id"] == tenant_id
                and j["customer_id"] == customer_id]
        return sorted(rows, key=lambda j: j["created_at"], reverse=True)[:limit]
