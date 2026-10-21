"""Triage step. Two implementations behind one Protocol:

- LLMClassifier: versioned prompt plus structured output (the production path).
- RuleBasedClassifier: keyword baseline. It runs offline in CI, and every LLM prompt version has
  to beat it in evals. If the prompt can't beat regexes, something is wrong.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel

from capstone.agents.job_intake.schemas import Classification
from capstone.core.tenancy import Tenant
from capstone.prompts.registry import PromptRef, get_prompt


class Classifier(Protocol):
    @property
    def prompt_ref(self) -> PromptRef | None: ...

    def classify(self, message: str, *, tenant: Tenant, customer: dict[str, Any] | None,
                 channel: str, history: list[Any]) -> Classification: ...


def customer_summary(customer: dict[str, Any] | None) -> str:
    if not customer:
        return "none (new contact)"
    postcodes = ", ".join(s["postcode"] for s in customer.get("sites", []))
    assets = ", ".join(sorted({a["model"] for a in customer.get("assets", [])}))
    return f"{customer['name']} (sites: {postcodes or 'none'}; equipment: {assets or 'unknown'})"


class LLMClassifier:
    def __init__(self, model: BaseChatModel, prompt_version: str | None = None) -> None:
        self.template, self.prompt_ref = get_prompt("job_intake_classify", prompt_version)
        self.chain = self.template | model.with_structured_output(Classification)

    def classify(self, message: str, *, tenant: Tenant, customer: dict[str, Any] | None,
                 channel: str, history: list[Any]) -> Classification:
        result = self.chain.invoke({
            "tenant_name": tenant.name,
            "trades": ", ".join(tenant.trades),
            "customer_summary": customer_summary(customer),
            "channel": channel,
            "message": message,
            "history": history,
        })
        assert isinstance(result, Classification)
        return result


_ASSET_CATEGORY = [
    # before fire_safety: "door closer on the fire exit" is a carpentry job, not an alarm job
    (r"door closer|fire (exit|door)s? (is|are|won't|doesn't)", "general_maintenance"),
    (r"\bgas\b|boiler|thermacore|heatwell|combi", "gas"),
    (r"air ?con|\bac\b|coolstream|heat pump|verdant|cooling|hvac", "hvac"),
    (r"water heater|aquaflow|hot water|tap|drain|leak|pouring|ceiling|toilet|pipe", "plumbing"),
    (r"fire alarm|sentinel|\bfire\b|smoke", "fire_safety"),
    (r"spark|socket|power|rcd|trip|electric|voltix|eicr|fuse|lights?", "electrical"),
    (r"door|paint|repaint|closer|window|lock", "general_maintenance"),
]
_EMERGENCY = (r"smell (of )?gas|gas smell|smell gas|spark|burning|pouring|flood|in alarm"
              r"|can't silence")
_HIGH = (r"no hot water|no heating|lost power|won't reset|stopped (cooling|working)"
         r"|lock(ing|s)? out|error code|code [A-Z]\d")
_LOW = r"quote|when convenient|at some point|repaint|next quarter"
_EXFILTRATION = r"(list|give|show|send|export)( me)? (every|all|other) (customer|user|account)"
_POSTCODE = re.compile(r"\b([A-Z]{1,2}\d{1,2}[A-Z]?)\s*(\d[A-Z]{2})\b")


class RuleBasedClassifier:
    prompt_ref: PromptRef | None = None

    def classify(self, message: str, *, tenant: Tenant, customer: dict[str, Any] | None,
                 channel: str, history: list[Any]) -> Classification:
        text = message.lower()
        injection = bool(re.search(r"ignore (all )?previous|disregard your|admin mode|system:",
                                   text))
        if re.search(r"reschedul|move (tomorrow|my|our|the)|push .* back|cancel", text):
            return Classification(intent="reschedule", injection_attempt=injection)
        is_question = text.strip().endswith("?") and re.search(
            r"how much|opening hours|what does|how long|warranty|what are", text)
        category = next((cat for rx, cat in _ASSET_CATEGORY if re.search(rx, text)), "unknown")
        if category == "unknown" and (injection or re.search(_EXFILTRATION, text)):
            # nothing to fix, only an attack: decline rather than ask for an address
            return Classification(intent="out_of_scope", injection_attempt=True)
        if is_question and not re.search(r"send someone|book|come out|visit", text):
            return Classification(intent="question", category="unknown", urgency="low",
                                  injection_attempt=injection)
        urgency = ("emergency" if re.search(_EMERGENCY, text) else
                   "high" if re.search(_HIGH, text) else
                   "low" if re.search(_LOW, text) else "normal")
        intent = "out_of_scope" if category != "unknown" and category not in tenant.trades \
            else "book_job"
        pc = _POSTCODE.search(message)
        addr = re.search(r"address is (.+?)(?:\s+[A-Z]{1,2}\d{1,2}[A-Z]?\s*\d[A-Z]{2})?\.?$",
                         message, re.I)
        missing: list[Any] = []
        if not customer and not pc:
            missing.append("address")
        if category == "unknown":
            missing.append("asset")
        return Classification(
            intent=intent,  # type: ignore[arg-type]
            category=category,  # type: ignore[arg-type]
            urgency=urgency,  # type: ignore[arg-type]
            safety_risk=urgency == "emergency" and category in {"gas", "electrical"},
            address=addr.group(1).strip() if addr else None,
            postcode=" ".join(pc.groups()) if pc else None,
            missing_info=missing,
            injection_attempt=injection,
        )
