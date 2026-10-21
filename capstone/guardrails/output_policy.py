"""Output guardrails, applied to every customer-facing reply before it is sent.

- Ungrounded money: any GBP amount in the reply must appear in the facts given to the responder.
- Unauthorised commitments: "free", "waive", "discount", "refund" only if the facts allow it.
- Moderation: optional OpenAI moderation for harmful content.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

MONEY_RE = re.compile(r"(?:£|GBP\s?)(\d+(?:\.\d{2})?)", re.I)
COMMITMENT_RE = re.compile(r"\b(free of charge|free visit|no charge|waive[sd]?|discount|refund)\b",
                           re.I)


@dataclass
class PolicyResult:
    ok: bool
    violations: list[str] = field(default_factory=list)


def check_reply(reply: str, facts: dict[str, Any]) -> PolicyResult:
    facts_text = json.dumps(facts, default=str)
    violations = []
    for amount in MONEY_RE.findall(reply):
        if amount not in facts_text:
            violations.append(f"ungrounded_amount:{amount}")
    for m in COMMITMENT_RE.findall(reply):
        if m.lower() not in facts_text.lower():
            violations.append(f"unauthorised_commitment:{m.lower()}")
    return PolicyResult(not violations, violations)


def moderate(text: str) -> PolicyResult:
    from openai import OpenAI

    res = OpenAI().moderations.create(model="omni-moderation-latest", input=text).results[0]
    if not res.flagged:
        return PolicyResult(True)
    cats = [k for k, v in res.categories.model_dump().items() if v]
    return PolicyResult(False, [f"moderation:{c}" for c in cats])
