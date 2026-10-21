"""Recipe (workflow) tools: a fixed multi-step sequence exposed to the agent as ONE tool.

Why: "reschedule a job" is lookup, then find slots, then update, then notify. Letting the LLM
chain four calls is slower, costlier and less reliable than one deterministic sub-graph.

    config:
      steps:
        - {id: jobs,  tool: get_open_jobs,        args: {customer_id: "{{input.customer_id}}"}}
        - {id: slots, tool: find_available_slots, args: {category: "{{steps.jobs.0.category}}",
                                                         urgency: "normal"}}
      output: "{{steps.slots}}"
"""

from __future__ import annotations

import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from capstone.tools.registry import Impl, ToolRegistry, ToolSpec

_REF = re.compile(r"^\{\{\s*([\w.]+)\s*\}\}$")


def _lookup(path: str, scope: dict[str, Any]) -> Any:
    cur: Any = scope
    for part in path.split("."):
        cur = cur[int(part)] if isinstance(cur, list) else cur[part]
    return cur


def _render(value: Any, scope: dict[str, Any]) -> Any:
    if isinstance(value, str) and (m := _REF.match(value)):
        return _lookup(m.group(1), scope)
    if isinstance(value, dict):
        return {k: _render(v, scope) for k, v in value.items()}
    if isinstance(value, list):
        return [_render(v, scope) for v in value]
    return value


class RecipeState(TypedDict):
    input: dict[str, Any]
    steps: dict[str, Any]


def build_recipe_graph(spec: ToolSpec, registry: ToolRegistry, tenant_id: str):
    g = StateGraph(RecipeState)
    prev = START
    for step in spec.config["steps"]:
        def node(state: RecipeState, _s: dict[str, Any] = step) -> dict[str, Any]:
            args = _render(_s["args"], {"input": state["input"], "steps": state["steps"]})
            result = registry.call(tenant_id, _s["tool"], args)
            return {"steps": {**state["steps"], _s["id"]: result}}
        g.add_node(step["id"], node)
        g.add_edge(prev, step["id"])
        prev = step["id"]
    g.add_edge(prev, END)
    return g.compile()


def make_recipe_impl(spec: ToolSpec, registry: ToolRegistry) -> Impl:
    def impl(tenant_id: str, args: dict[str, Any]) -> Any:
        graph = build_recipe_graph(spec, registry, tenant_id)
        state = graph.invoke({"input": args, "steps": {}})
        return _render(spec.config.get("output", "{{steps}}"), {"input": args, **state})

    return impl
