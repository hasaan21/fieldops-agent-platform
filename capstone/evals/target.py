"""The system under test: one dataset example in, structured outputs out."""

from __future__ import annotations

import time
import uuid
from typing import Any

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from capstone.agents.job_intake.graph import IntakeContext, IntakeDeps, build_graph, default_deps
from capstone.core.crm import CRMRepository


class IntakeTarget:
    def __init__(self, deps: IntakeDeps | None = None, *, prompt_version: str | None = None,
                 offline: bool | None = None) -> None:
        self.deps = deps or default_deps(CRMRepository(), prompt_version=prompt_version,
                                         offline=offline)
        self.app = build_graph(self.deps).compile(checkpointer=InMemorySaver(),
                                                  store=InMemoryStore())

    @property
    def config_id(self) -> str:
        c = self.deps.classifier
        ref = getattr(c, "prompt_ref", None)
        base = f"{type(c).__name__}:{ref}" if ref else type(c).__name__
        return base + ("+ml" if self.deps.models is not None else "")

    def __call__(self, inputs: dict[str, Any]) -> dict[str, Any]:
        ctx = IntakeContext(tenant_id=inputs["tenant_id"], channel=inputs["channel"],
                            sender=inputs.get("sender", ""), review_mode="auto_approve")
        t0 = time.perf_counter()
        try:
            out = self.app.invoke({"messages": [HumanMessage(inputs["message"])]},
                                  {"configurable": {"thread_id": uuid.uuid4().hex},
                                   "metadata": {"tenant_id": inputs["tenant_id"],
                                                "eval_config": self.config_id}},
                                  context=ctx)
            error = None
        except Exception as e:  # an eval run must record failures, not crash
            out, error = {}, f"{type(e).__name__}: {e}"
        return {
            "outcome": out.get("outcome"),
            "classification": out.get("classification"),
            "trajectory": out.get("trajectory", []),
            "reply": out.get("reply"),
            "flags": out.get("flags", []),
            "proposal": out.get("proposal"),
            "channel": inputs["channel"],
            "latency_ms": round((time.perf_counter() - t0) * 1000),
            "error": error,
        }
