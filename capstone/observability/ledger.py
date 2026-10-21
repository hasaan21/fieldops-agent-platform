"""Cost and latency ledger: one row per agent run, with every LLM call inside it.

LangSmith already shows tokens and latency per trace. The ledger exists because:
- finance and tenants ask "what did tenant t2 cost last month?", which is a SQL question,
  not a trace-search question;
- budgets need enforcing *in the request path* (flag or degrade when a voice turn is over
  1.5s or a conversation is over its cost cap), not discovered on a dashboard next week;
- it must survive switching tracing vendors.

`LedgerCallback` is a LangChain callback handler, so it sees every chat-model call in the graph
(classifier, responder, ReAct answerer, judge) without touching agent code.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult

# USD per 1M tokens (input, output). Keep this in config/DB in production and version it:
# a price change must not silently rewrite historical costs, so cost is computed at write time.
PRICES: dict[str, tuple[float, float]] = {
    "gpt-4.1": (2.00, 8.00),
    "gpt-4.1-mini": (0.40, 1.60),
    "gpt-4.1-nano": (0.10, 0.40),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-sonnet-4": (3.00, 15.00),
    "claude-haiku-4": (1.00, 5.00),
    "text-embedding-3-small": (0.02, 0.0),
}

# Per-turn latency budgets by channel (ms). Voice is the hard one: callers hang up.
LATENCY_BUDGET_MS = {"voice": 1500, "sms": 8000, "whatsapp": 8000, "web": 6000, "email": 30000}
MAX_COST_PER_RUN_USD = 0.02


def price_for(model: str) -> tuple[float, float]:
    model = model.split(":")[-1].lower()
    for name in sorted(PRICES, key=len, reverse=True):  # longest prefix wins
        if model.startswith(name):
            return PRICES[name]
    return (0.0, 0.0)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    pin, pout = price_for(model)
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


@dataclass
class LLMCall:
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: int
    cost_usd: float


@dataclass
class RunRecord:
    run_id: str
    tenant_id: str
    agent: str
    channel: str
    latency_ms: int
    outcome: str | None = None
    prompt_ref: str | None = None
    llm_calls: list[LLMCall] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def input_tokens(self) -> int:
        return sum(c.input_tokens for c in self.llm_calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.llm_calls)

    @property
    def cost_usd(self) -> float:
        return sum(c.cost_usd for c in self.llm_calls)

    @property
    def model(self) -> str | None:
        return self.llm_calls[0].model if self.llm_calls else None

    def budget_violations(self) -> list[str]:
        out = []
        budget = LATENCY_BUDGET_MS.get(self.channel, 8000)
        if self.latency_ms > budget:
            out.append(f"latency {self.latency_ms}ms > {budget}ms ({self.channel})")
        if self.cost_usd > MAX_COST_PER_RUN_USD:
            out.append(f"cost ${self.cost_usd:.4f} > ${MAX_COST_PER_RUN_USD}")
        return out

    def to_row(self) -> dict[str, Any]:
        row = asdict(self)
        row.pop("llm_calls")
        return row | {"input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                      "cost_usd": round(self.cost_usd, 6), "model": self.model,
                      "llm_calls": len(self.llm_calls),
                      "budget_violations": self.budget_violations()}


class LedgerCallback(BaseCallbackHandler):
    """Collects token usage and latency for each chat-model call within one agent run."""

    def __init__(self) -> None:
        self.calls: list[LLMCall] = []
        self._starts: dict[UUID, tuple[float, str]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(self, serialized: dict[str, Any], messages: Any, *, run_id: UUID,
                            **kwargs: Any) -> None:
        model = (kwargs.get("metadata") or {}).get("ls_model_name") or \
            (serialized or {}).get("kwargs", {}).get("model_name") or \
            (serialized or {}).get("kwargs", {}).get("model") or "unknown"
        with self._lock:
            self._starts[run_id] = (time.perf_counter(), str(model))

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        with self._lock:
            t0, model = self._starts.pop(run_id, (time.perf_counter(), "unknown"))
        usage = _usage(response)
        model = (response.llm_output or {}).get("model_name") or model
        self.calls.append(LLMCall(
            model=model, input_tokens=usage[0], output_tokens=usage[1],
            latency_ms=round((time.perf_counter() - t0) * 1000),
            cost_usd=cost_usd(model, *usage)))


def _usage(response: LLMResult) -> tuple[int, int]:
    """Token usage across providers: prefer message.usage_metadata (standardised in LC 0.3)."""
    for gens in response.generations:
        for g in gens:
            meta = getattr(getattr(g, "message", None), "usage_metadata", None)
            if meta:
                return int(meta.get("input_tokens", 0)), int(meta.get("output_tokens", 0))
    tu = (response.llm_output or {}).get("token_usage") or {}
    return int(tu.get("prompt_tokens", 0)), int(tu.get("completion_tokens", 0))


class Ledger:
    """In-memory ledger with an optional Postgres sink (table `run_ledger`, infra/postgres)."""

    def __init__(self, database_url: str = "", max_rows: int = 10_000) -> None:
        self.database_url = database_url
        self.rows: list[dict[str, Any]] = []
        self.max_rows = max_rows
        self._lock = threading.Lock()

    def record(self, rec: RunRecord) -> dict[str, Any]:
        row = rec.to_row()
        with self._lock:
            self.rows.append(row)
            del self.rows[:-self.max_rows]
        if self.database_url:
            self._write_pg(row)
        return row

    def _write_pg(self, row: dict[str, Any]) -> None:
        import psycopg

        with psycopg.connect(self.database_url) as conn:
            conn.execute(
                "INSERT INTO run_ledger (run_id, tenant_id, agent, channel, prompt_ref, model, "
                "input_tokens, output_tokens, cost_usd, latency_ms, outcome, budget_violations) "
                "VALUES (%(run_id)s, %(tenant_id)s, %(agent)s, %(channel)s, %(prompt_ref)s, "
                "%(model)s, %(input_tokens)s, %(output_tokens)s, %(cost_usd)s, %(latency_ms)s, "
                "%(outcome)s, %(budget_violations)s) ON CONFLICT (run_id) DO NOTHING", row)

    def summary(self, tenant_id: str) -> dict[str, Any]:
        rows = [r for r in self.rows if r["tenant_id"] == tenant_id]
        if not rows:
            return {"tenant_id": tenant_id, "runs": 0}
        lat = sorted(r["latency_ms"] for r in rows)
        by_channel: dict[str, list[int]] = {}
        for r in rows:
            by_channel.setdefault(r["channel"], []).append(r["latency_ms"])
        return {
            "tenant_id": tenant_id,
            "runs": len(rows),
            "cost_usd_total": round(sum(r["cost_usd"] for r in rows), 6),
            "cost_usd_per_run": round(statistics.fmean(r["cost_usd"] for r in rows), 6),
            "tokens": {"input": sum(r["input_tokens"] for r in rows),
                       "output": sum(r["output_tokens"] for r in rows)},
            "latency_ms": {"p50": lat[len(lat) // 2], "p95": lat[max(0, int(len(lat) * .95) - 1)]},
            "latency_p95_by_channel": {
                ch: sorted(v)[max(0, int(len(v) * .95) - 1)] for ch, v in by_channel.items()},
            "budget_violations": sum(1 for r in rows if r["budget_violations"]),
            "outcomes": {o: sum(1 for r in rows if r["outcome"] == o)
                         for o in sorted({str(r["outcome"]) for r in rows})},
        }
