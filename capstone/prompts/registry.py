"""Prompt registry: LangSmith Hub in production, local YAML as the source of truth in git.

Local layout: capstone/prompts/<name>.yaml
    name: job-intake-classify
    tags: {dev: v2, staging: v2, prod: v1}
    versions:
      v1: {system: "...", human: "..."}

Promotion = moving a tag. `scripts/push_prompts.py` mirrors versions and tags to LangSmith, and
the CI eval gate (Week 2) is what is allowed to move `staging`/`prod`.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from capstone.core.config import get_settings

PROMPT_DIR = Path(__file__).parent


@dataclass(frozen=True)
class PromptRef:
    name: str
    version: str
    tag: str | None
    source: str

    def __str__(self) -> str:
        return f"{self.name}@{self.version}" + (f" ({self.tag})" if self.tag else "")


@lru_cache
def _load(name: str) -> dict[str, Any]:
    path = PROMPT_DIR / f"{name}.yaml"
    with path.open() as fh:
        data: dict[str, Any] = yaml.safe_load(fh)
    return data


def list_versions(name: str) -> list[str]:
    return list(_load(name)["versions"])


def resolve(name: str, tag_or_version: str | None = None) -> tuple[str, str | None]:
    spec = _load(name)
    key = tag_or_version or get_settings().fieldops_prompt_tag
    if key in spec["versions"]:
        return key, None
    if key in spec.get("tags", {}):
        return spec["tags"][key], key
    raise KeyError(f"{name}: no version or tag {key!r}")


def build_template(version_spec: dict[str, Any], with_history: bool) -> ChatPromptTemplate:
    messages: list[Any] = [("system", version_spec["system"])]
    for shot in version_spec.get("few_shot", []):
        messages += [("human", shot["input"]), ("ai", shot["output"])]
    if with_history:
        messages.append(MessagesPlaceholder("history"))
    if version_spec.get("human"):
        messages.append(("human", version_spec["human"]))
    return ChatPromptTemplate.from_messages(messages, template_format="jinja2")


def get_prompt(name: str, tag_or_version: str | None = None, *, with_history: bool = True
               ) -> tuple[ChatPromptTemplate, PromptRef]:
    s = get_settings()
    if s.fieldops_prompt_source == "langsmith" and s.langsmith_api_key:
        from langsmith import Client

        spec = _load(name)
        key = tag_or_version or s.fieldops_prompt_tag
        prompt = Client().pull_prompt(f"{spec['name']}:{key}")
        return prompt, PromptRef(spec["name"], key, key, "langsmith")
    version, tag = resolve(name, tag_or_version)
    spec = _load(name)
    tmpl = build_template(spec["versions"][version], with_history)
    return tmpl, PromptRef(spec["name"], version, tag, "local")
