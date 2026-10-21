"""Customer-facing reply. The graph decides the outcome; the responder only phrases it."""

from __future__ import annotations

import json
from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.output_parsers import StrOutputParser

from capstone.core.tenancy import Tenant
from capstone.prompts.registry import PromptRef, get_prompt


class Responder(Protocol):
    @property
    def prompt_ref(self) -> PromptRef | None: ...

    def respond(self, *, tenant: Tenant, channel: str, outcome: str, message: str,
                facts: dict[str, Any]) -> str: ...


class LLMResponder:
    def __init__(self, model: BaseChatModel, prompt_version: str | None = None) -> None:
        template, self.prompt_ref = get_prompt("job_intake_respond", prompt_version,
                                               with_history=False)
        self.chain = template | model | StrOutputParser()

    def respond(self, *, tenant: Tenant, channel: str, outcome: str, message: str,
                facts: dict[str, Any]) -> str:
        return self.chain.invoke({
            "tenant_name": tenant.name,
            "tone": tenant.tone,
            "channel": channel,
            "outcome": outcome,
            "message": message,
            "facts": json.dumps(facts, indent=2, default=str),
        })


class TemplateResponder:
    prompt_ref: PromptRef | None = None

    def respond(self, *, tenant: Tenant, channel: str, outcome: str, message: str,
                facts: dict[str, Any]) -> str:
        name = tenant.name
        match outcome:
            case "booked":
                return (f"Thanks - {name} has booked an engineer for {facts['slot']['start']}. "
                        f"Job ref {facts['job_id']}.")
            case "pending_approval":
                return f"Thanks, {name} has received your request and will confirm a time shortly."
            case "rejected_by_reviewer":
                return f"Thanks for contacting {name}. A member of our team will call you."
            case "clarify":
                need = " and ".join(facts.get("missing_info", [])) or "a few more details"
                return f"Thanks for contacting {name}. Could you tell us the {need}?"
            case "escalated":
                return ("For your safety: if you smell gas, turn off the supply at the meter, "
                        f"open windows and leave the property. {name} is calling you now. "
                        f"Emergency line: {facts.get('emergency_line', '')}.")
            case "declined":
                return f"Sorry, {name} doesn't cover that type of work."
            case "answered":
                return str(facts.get("answer", ""))
            case "reschedule_handoff":
                return f"No problem - {name}'s scheduling team will contact you to rearrange."
        return f"Thanks for contacting {name}."
