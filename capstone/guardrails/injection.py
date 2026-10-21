"""Prompt-injection defence, layered:

1. Detection: cheap heuristics here (an optional classifier model is a Week 4 stretch goal).
2. Spotlighting: untrusted text (emails, tool outputs, KB chunks) is wrapped in delimiters and
   the system prompt says content inside them is data, never instructions.
3. Capability limits (the real defence): the LLM cannot pick the tenant, cannot book without a
   human/policy approval step, and only has tools scoped to the current customer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|rules|prompts?)",
    r"disregard (your|all|the) (rules|instructions|guidelines)",
    r"you are now (in )?(admin|developer|dan|god) mode",
    r"^\s*system\s*:",
    r"\bsystem\s*:\s*you are",
    r"reveal (your|the) (system )?prompt",
    r"(list|show|give me) (every|all) (customer|user)s?'? ?(phone|email|data|details|numbers)",
    r"act as (an? )?(administrator|admin|root)",
    r"</?(system|instructions?)>",
]
_RX = [re.compile(p, re.I | re.M) for p in PATTERNS]


@dataclass(frozen=True)
class InjectionScan:
    flagged: bool
    matches: tuple[str, ...]


def scan(text: str) -> InjectionScan:
    hits = tuple(m.group(0) for rx in _RX if (m := rx.search(text)))
    return InjectionScan(bool(hits), hits)


def spotlight(text: str, source: str = "untrusted") -> str:
    """Wrap untrusted content in delimiters, and strip any delimiters the content forges."""
    cleaned = text.replace("<<<", "‹‹‹").replace(">>>", "›››")
    return f"<<<{source.upper()}_CONTENT\n{cleaned}\n>>>"
