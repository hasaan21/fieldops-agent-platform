"""PII redaction applied to inbound messages before they reach the LLM or the trace store.

Customer names, phone numbers and addresses are needed to book jobs, so they are kept.
Payment data and secrets are never needed, so they are always removed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
CVV_RE = re.compile(r"\b(?:cvv|cvc|security code)\s*[:#]?\s*\d{3,4}\b", re.I)
EXPIRY_RE = re.compile(r"\bexp(?:iry|ires)?\.?\s*[:]?\s*\d{2}\s*/\s*\d{2,4}\b", re.I)
IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b")
SORT_ACC_RE = re.compile(r"\b\d{2}-\d{2}-\d{2}\s+\d{8}\b")
NI_RE = re.compile(r"\b[A-CEGHJ-PR-TW-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b", re.I)
SECRET_RE = re.compile(r"\b(?:sk|pk|pl|ghp|xox[bp])[-_][A-Za-z0-9_-]{16,}\b")


def luhn_ok(number: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", number)]
    if not 13 <= len(digits) <= 19:
        return False
    checksum = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            d = d - 9 if d > 9 else d
        checksum += d
    return checksum % 10 == 0


@dataclass
class RedactionResult:
    text: str
    redactions: list[str] = field(default_factory=list)


def redact(text: str) -> RedactionResult:
    found: list[str] = []

    def card(m: re.Match[str]) -> str:
        if luhn_ok(m.group(0)):
            found.append("card_number")
            return "[REDACTED_CARD]"
        return m.group(0)

    out = CARD_RE.sub(card, text)
    for name, rx, token in [
        ("cvv", CVV_RE, "[REDACTED_CVV]"),
        ("card_expiry", EXPIRY_RE, "[REDACTED_EXPIRY]"),
        ("iban", IBAN_RE, "[REDACTED_IBAN]"),
        ("bank_account", SORT_ACC_RE, "[REDACTED_BANK]"),
        ("national_insurance", NI_RE, "[REDACTED_NI]"),
        ("secret", SECRET_RE, "[REDACTED_SECRET]"),
    ]:
        out, n = rx.subn(token, out)
        found += [name] * n
    return RedactionResult(out, found)
