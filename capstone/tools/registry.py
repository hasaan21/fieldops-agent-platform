"""Tool registry: declarative tool specs, per-tenant enablement, LangChain tool materialisation.

A ToolSpec is data (it can live in a DB and be edited in an agent-builder UI). Materialising a
spec for a tenant binds the tenant, credentials and policies server-side. The LLM only sees the
name, description and JSON schema.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

ToolKind = Literal["python", "http", "mcp", "recipe"]


class ToolSpec(BaseModel):
    name: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    description: str = Field(min_length=20)
    kind: ToolKind
    input_schema: dict[str, Any]
    config: dict[str, Any] = Field(default_factory=dict)
    side_effects: bool = False          # writes data / sends messages / costs money
    requires_approval: bool = False     # route through human review before executing
    timeout_s: float = 10.0
    version: str = "1"


# (tenant_id, args) -> result. The tenant is always the first positional argument.
Impl = Callable[[str, dict[str, Any]], Any]


@dataclass
class ToolRegistry:
    specs: dict[str, ToolSpec] = field(default_factory=dict)
    impls: dict[str, Impl] = field(default_factory=dict)
    enabled: dict[str, set[str]] = field(default_factory=dict)  # tenant -> tool names

    def register(self, spec: ToolSpec, impl: Impl) -> None:
        if spec.name in self.specs:
            raise ValueError(f"duplicate tool {spec.name}")
        self.specs[spec.name] = spec
        self.impls[spec.name] = impl

    def enable(self, tenant_id: str, *names: str) -> None:
        unknown = set(names) - set(self.specs)
        if unknown:
            raise KeyError(f"unknown tools {unknown}")
        self.enabled.setdefault(tenant_id, set()).update(names)

    def call(self, tenant_id: str, name: str, args: dict[str, Any]) -> Any:
        if name not in self.enabled.get(tenant_id, set()):
            raise PermissionError(f"tool {name} not enabled for {tenant_id}")
        _validate(self.specs[name].input_schema, args)
        return self.impls[name](tenant_id, args)

    def tools_for(self, tenant_id: str, *, include_side_effects: bool = True
                  ) -> list[StructuredTool]:
        out = []
        for name in sorted(self.enabled.get(tenant_id, set())):
            spec = self.specs[name]
            if spec.side_effects and not include_side_effects:
                continue
            out.append(StructuredTool(
                name=spec.name, description=spec.description, args_schema=spec.input_schema,
                func=lambda _n=name, **kw: self.call(tenant_id, _n, kw),
                metadata={"kind": spec.kind, "side_effects": spec.side_effects,
                          "requires_approval": spec.requires_approval}))
        return out


def _validate(schema: dict[str, Any], args: dict[str, Any]) -> None:
    """Minimal JSON-schema check (required + types + enums). Use `jsonschema` for full coverage."""
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool,
             "object": dict, "array": list}
    for req in schema.get("required", []):
        if req not in args:
            raise ValueError(f"missing required argument {req!r}")
    props = schema.get("properties", {})
    if schema.get("additionalProperties") is False:
        extra = set(args) - set(props)
        if extra:
            raise ValueError(f"unexpected arguments {extra}")
    for k, v in args.items():
        p = props.get(k, {})
        if "type" in p and not isinstance(v, types[p["type"]]):  # type: ignore[arg-type]
            raise ValueError(f"{k} must be {p['type']}")
        if "enum" in p and v not in p["enum"]:
            raise ValueError(f"{k} must be one of {p['enum']}")
