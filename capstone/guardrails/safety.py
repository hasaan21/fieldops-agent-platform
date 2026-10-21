"""Deterministic safety net: life-safety keywords force escalation whatever the classifier said.

Defence in depth: the LLM handles nuance, and this catches what it misses (multi-intent
messages, odd phrasing, model regressions). It's tuned for recall; a false positive only costs a
human callback.
"""

from __future__ import annotations

import re

LIFE_SAFETY = re.compile(
    r"smell(s|ing)? (of )?gas|gas (smell|leak)|smell gas|carbon monoxide|\bco alarm|"
    r"spark(s|ing)?|burning smell|smell of burning|smoke (coming|from)|exposed (live )?wires?|"
    r"electric shock",
    re.I,
)


def life_safety_risk(text: str) -> bool:
    return bool(LIFE_SAFETY.search(text))
