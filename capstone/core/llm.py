from __future__ import annotations

from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from capstone.core.config import get_settings

Role = Literal["agent", "judge"]


def get_chat_model(role: Role = "agent", **kwargs) -> BaseChatModel:
    """Provider-agnostic chat model. Swap providers with FIELDOPS_CHAT_MODEL=anthropic:claude-..."""
    s = get_settings()
    spec = s.fieldops_chat_model if role == "agent" else s.fieldops_judge_model
    defaults = {"temperature": 0.0, "timeout": 30, "max_retries": 2}
    return init_chat_model(spec, **{**defaults, **kwargs})
