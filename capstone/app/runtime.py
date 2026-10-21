"""Process-wide runtime: compiled graphs, checkpointer, store, review queue."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.types import Command

from capstone.agents.job_intake.graph import IntakeContext, build_graph, default_deps
from capstone.core.config import get_settings
from capstone.core.crm import CRMRepository
from capstone.observability.ledger import Ledger, LedgerCallback, RunRecord
from capstone.observability.otel import agent_span, current_trace_id, record_run_metrics


@dataclass
class ReviewQueue:
    """Pending human-in-the-loop items. Mirrors the review_items table in Postgres."""

    items: dict[str, dict[str, dict[str, Any]]] = field(default_factory=dict)

    def add(self, tenant_id: str, thread_id: str, payload: dict[str, Any],
            ctx: IntakeContext) -> None:
        self.items.setdefault(tenant_id, {})[thread_id] = {
            "thread_id": thread_id, "payload": payload, "context": ctx.__dict__}

    def pending(self, tenant_id: str) -> list[dict[str, Any]]:
        return list(self.items.get(tenant_id, {}).values())

    def pop(self, tenant_id: str, thread_id: str) -> dict[str, Any] | None:
        return self.items.get(tenant_id, {}).pop(thread_id, None)


class Runtime:
    def __init__(self) -> None:
        s = get_settings()
        self.crm = CRMRepository()
        self.reviews = ReviewQueue()
        if s.database_url:
            from langgraph.checkpoint.postgres import PostgresSaver
            from langgraph.store.postgres import PostgresStore
            from psycopg import Connection
            from psycopg.rows import dict_row

            conn_kwargs: dict[str, Any] = {"autocommit": True, "prepare_threshold": 0,
                                           "row_factory": dict_row}
            self._conn = Connection.connect(s.database_url, **conn_kwargs)
            self.checkpointer: Any = PostgresSaver(self._conn)
            self.checkpointer.setup()
            self.store: Any = PostgresStore(self._conn)
            self.store.setup()
        else:
            self.checkpointer = InMemorySaver()
            self.store = InMemoryStore()
        deps = default_deps(self.crm)
        if s.fieldops_ml_tools:
            from capstone.ml.serving import get_model_service

            deps.models = get_model_service()
        ref = getattr(deps.classifier, "prompt_ref", None)
        self.prompt_ref = f"{ref.name}:{ref.version}" if ref else type(deps.classifier).__name__
        self.ledger = Ledger(s.database_url)
        self.intake = build_graph(deps).compile(checkpointer=self.checkpointer, store=self.store)

    @staticmethod
    def thread_id(tenant_id: str, channel: str, sender: str) -> str:
        # Tenant-prefixed so a thread can never be resumed under another tenant's credentials.
        return f"{tenant_id}:{channel}:{sender or uuid.uuid4().hex}"

    def handle_message(self, tenant_id: str, channel: str, sender: str, text: str,
                       message_id: str | None = None) -> dict[str, Any]:
        ctx = IntakeContext(tenant_id=tenant_id, channel=channel, sender=sender)
        thread = self.thread_id(tenant_id, channel, sender)
        out, usage = self._invoke({"messages": [HumanMessage(text, id=message_id)]}, thread, ctx)
        return self._result(out, tenant_id, thread, ctx) | {"usage": usage}

    def _invoke(self, payload: Any, thread: str, ctx: IntakeContext
                ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Run the graph inside an OTel span with the cost/latency ledger attached."""
        ledger_cb = LedgerCallback()
        with agent_span("agent.job_intake", tenant_id=ctx.tenant_id, channel=ctx.channel,
                        thread_id=thread) as span:
            cfg: RunnableConfig = {
                "configurable": {"thread_id": thread},
                "callbacks": [ledger_cb],
                "metadata": {"tenant_id": ctx.tenant_id, "channel": ctx.channel,
                             "agent": "job_intake", "otel_trace_id": current_trace_id()},
                "tags": [f"tenant:{ctx.tenant_id}", f"channel:{ctx.channel}"]}
            t0 = time.perf_counter()
            out = self.intake.invoke(payload, cfg, context=ctx)
            rec = RunRecord(run_id=uuid.uuid4().hex, tenant_id=ctx.tenant_id,
                            agent="job_intake", channel=ctx.channel,
                            latency_ms=round((time.perf_counter() - t0) * 1000),
                            outcome=out.get("outcome") or
                            ("pending_review" if "__interrupt__" in out else None),
                            prompt_ref=self.prompt_ref, llm_calls=ledger_cb.calls)
            row = self.ledger.record(rec)
            record_run_metrics(row)
            span.set_attribute("fieldops.cost_usd", row["cost_usd"])
            span.set_attribute("fieldops.outcome", str(row["outcome"]))
            if row["budget_violations"]:
                span.add_event("budget_violation", {"detail": row["budget_violations"]})
        usage = {k: row[k] for k in ("latency_ms", "cost_usd", "input_tokens", "output_tokens",
                                     "llm_calls", "budget_violations")}
        return out, usage

    def resume(self, tenant_id: str, thread_id: str, decision: dict[str, Any]) -> dict[str, Any]:
        if not thread_id.startswith(f"{tenant_id}:"):
            raise PermissionError("thread belongs to another tenant")
        item = self.reviews.pop(tenant_id, thread_id)
        if item is None:
            raise KeyError(thread_id)
        ctx = IntakeContext(**item["context"])
        out, usage = self._invoke(Command(resume=decision), thread_id, ctx)
        return self._result(out, tenant_id, thread_id, ctx) | {"usage": usage}

    def _result(self, out: dict[str, Any], tenant_id: str, thread: str,
                ctx: IntakeContext) -> dict[str, Any]:
        if "__interrupt__" in out:
            payload = out["__interrupt__"][0].value
            self.reviews.add(tenant_id, thread, payload, ctx)
            return {"thread_id": thread, "status": "pending_review", "review": payload}
        return {"thread_id": thread, "status": "done", "outcome": out.get("outcome"),
                "reply": out.get("reply"), "job": out.get("job"),
                "trajectory": out.get("trajectory", [])}


@lru_cache
def get_runtime() -> Runtime:
    return Runtime()
