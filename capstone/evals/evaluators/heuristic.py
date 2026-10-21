"""Deterministic evaluators: cheap, fast, and the first line of the CI gate.

Each evaluator takes (outputs, reference) and returns {"key", "score", "comment"}, the same
shape LangSmith feedback uses, so the same functions work locally and in `langsmith.evaluate`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from pydantic import ValidationError

from capstone.agents.job_intake.schemas import Classification

Result = dict[str, Any]
Evaluator = Callable[[dict[str, Any], dict[str, Any]], Result | None]

URGENCY_ORDER = ["low", "normal", "high", "emergency"]
CHANNEL_LIMITS = {"sms": 320, "whatsapp": 480, "voice": 400, "email": 1200, "web": 1200}


def _r(key: str, score: float | bool, comment: str = "") -> Result:
    return {"key": key, "score": float(score), "comment": comment}


def outcome_correct(out: dict[str, Any], ref: dict[str, Any]) -> Result:
    exp, got = ref["expected_outcome"], out.get("outcome")
    # pending/rejected mean "would have booked, waiting for a human": still correct routing
    ok = got == exp or (exp == "booked" and got in {"pending_approval", "rejected_by_reviewer"})
    return _r("outcome_correct", ok, f"expected={exp} got={got}")


def intent_correct(out: dict[str, Any], ref: dict[str, Any]) -> Result:
    got = (out.get("classification") or {}).get("intent")
    return _r("intent_correct", got == ref["intent"], f"expected={ref['intent']} got={got}")


def category_correct(out: dict[str, Any], ref: dict[str, Any]) -> Result | None:
    if ref["intent"] != "book_job" or ref.get("category") in (None, "unknown"):
        return None  # not applicable; excluded from the mean rather than counted as a pass
    got = (out.get("classification") or {}).get("category")
    return _r("category_correct", got == ref["category"], f"expected={ref['category']} got={got}")


def urgency_score(out: dict[str, Any], ref: dict[str, Any]) -> Result | None:
    """Ordinal partial credit: off-by-one = 0.5. Under-triaging an emergency scores 0."""
    if ref["intent"] != "book_job":
        return None
    got = (out.get("classification") or {}).get("urgency", "normal")
    exp = ref["urgency"]
    if exp == "emergency" and got != "emergency":
        return _r("urgency_score", 0, f"UNDER-TRIAGED emergency as {got}")
    dist = abs(URGENCY_ORDER.index(got) - URGENCY_ORDER.index(exp))
    return _r("urgency_score", {0: 1.0, 1: 0.5}.get(dist, 0.0), f"expected={exp} got={got}")


def safety_escalation(out: dict[str, Any], ref: dict[str, Any]) -> Result | None:
    """Recall on safety cases is what matters: a missed gas leak is a critical failure."""
    if not ref["should_escalate"]:
        return None
    ok = out.get("outcome") == "escalated" and "create_job" not in out.get("trajectory", [])
    return _r("safety_escalation", ok, "" if ok else "SAFETY: emergency not escalated")


def trajectory_match(out: dict[str, Any], ref: dict[str, Any]) -> Result:
    """Expected tools must appear in order (other tools may be interleaved)."""
    expected = ref.get("expected_tools") or []
    got = out.get("trajectory", [])
    remaining = iter(got)
    in_order = all(t in remaining for t in expected)
    side_effect_unexpected = "create_job" in got and "create_job" not in expected
    return _r("trajectory_match", in_order and not side_effect_unexpected,
              f"expected={expected} got={got}")


def no_sensitive_leak(out: dict[str, Any], ref: dict[str, Any]) -> Result:
    reply = out.get("reply") or ""
    leaked = [s for s in ref.get("must_not_contain", []) if s.lower() in reply.lower()]
    return _r("no_sensitive_leak", not leaked, f"leaked={leaked}" if leaked else "")


def must_contain_any(out: dict[str, Any], ref: dict[str, Any]) -> Result | None:
    needles = ref.get("must_contain_any")
    if not needles:
        return None
    reply = (out.get("reply") or "").lower()
    ok = any(n.lower() in reply for n in needles)
    return _r("must_contain_any", ok, f"needles={needles}")


def reply_length_ok(out: dict[str, Any], ref: dict[str, Any]) -> Result:
    limit = CHANNEL_LIMITS.get(out.get("channel", "web"), 1200)
    n = len(out.get("reply") or "")
    return _r("reply_length_ok", 0 < n <= limit, f"{n} chars (limit {limit})")


def classification_schema_valid(out: dict[str, Any], ref: dict[str, Any]) -> Result:
    try:
        Classification.model_validate(out.get("classification") or {})
        return _r("schema_valid", True)
    except ValidationError as e:
        return _r("schema_valid", False, str(e)[:200])


ALL: list[Evaluator] = [outcome_correct, intent_correct, category_correct, urgency_score,
                        safety_escalation, trajectory_match, no_sensitive_leak, must_contain_any,
                        reply_length_ok, classification_schema_valid]

# Reference-free subset, usable on live production traffic (online scoring).
REFERENCE_FREE: list[Evaluator] = [reply_length_ok, classification_schema_valid]
