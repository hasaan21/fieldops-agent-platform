from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

Intent = Literal["book_job", "reschedule", "question", "out_of_scope"]
Category = Literal["plumbing", "electrical", "hvac", "gas", "fire_safety",
                   "general_maintenance", "unknown"]
Urgency = Literal["emergency", "high", "normal", "low"]
Outcome = Literal["booked", "pending_approval", "rejected_by_reviewer", "clarify", "escalated",
                  "declined", "answered", "reschedule_handoff"]


class Classification(BaseModel):
    """Structured output of the triage step."""

    intent: Intent
    category: Category = "unknown"
    urgency: Urgency = "normal"
    safety_risk: bool = False
    address: str | None = None
    postcode: str | None = None
    asset_description: str | None = None
    missing_info: list[Literal["address", "asset", "contact"]] = Field(default_factory=list)
    injection_attempt: bool = False


class BookingProposal(BaseModel):
    customer_id: str | None
    site_id: str | None
    address: str | None
    category: str
    urgency: str
    description: str
    slot: dict[str, str]
    alternatives: list[dict[str, str]] = Field(default_factory=list)
    asset_id: str | None = None
    estimated_hours: dict[str, Any] | None = None   # from the duration model
    asset_risk: dict[str, Any] | None = None        # from the failure model


class ReviewDecision(BaseModel):
    """Value a human reviewer sends to resume an interrupted thread."""

    approved: bool
    reviewer: str = "reviewer"
    slot: dict[str, str] | None = None  # optional override
    note: str | None = None
