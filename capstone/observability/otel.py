"""OpenTelemetry setup: traces and metrics to Azure Application Insights (or any OTLP backend).

Two kinds of telemetry, deliberately separate:
- **LangSmith** for LLM-level traces: prompts, tool calls, token usage, feedback, datasets.
  Enabled with LANGSMITH_TRACING=true; nothing in this file.
- **OpenTelemetry** for service-level telemetry: HTTP spans, queue latency, DB calls, error
  rates, custom metrics (cost, budget violations) in the same place as the rest of the
  company's services (Joblogic runs on Azure, so App Insights).

They are joined by IDs: every agent span carries `fieldops.thread_id` and the LangSmith run
metadata carries the OTel trace id, so you can jump from an App Insights alert to the trace.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import metrics, trace

log = logging.getLogger(__name__)
_configured = False

tracer = trace.get_tracer("fieldops")
meter = metrics.get_meter("fieldops")
RUN_COST = meter.create_counter("fieldops.agent.cost_usd", unit="USD",
                                description="LLM cost per agent run")
RUN_LATENCY = meter.create_histogram("fieldops.agent.latency_ms", unit="ms",
                                     description="End-to-end agent run latency")
BUDGET_VIOLATIONS = meter.create_counter("fieldops.agent.budget_violations",
                                         description="Runs over their latency or cost budget")


def setup_tracing(service_name: str = "fieldops-api", connection_string: str = "",
                  app: Any = None) -> str:
    """Configure exporters once. Returns which backend is active."""
    global _configured
    if _configured:
        return "already-configured"
    _configured = True
    if connection_string:
        try:
            from azure.monitor.opentelemetry import configure_azure_monitor

            configure_azure_monitor(connection_string=connection_string,
                                    resource_attributes={"service.name": service_name})
            backend = "azure-monitor"
        except ImportError:
            log.warning("azure-monitor-opentelemetry not installed: uv sync --extra observability")
            backend = "none"
    else:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider

        trace.set_tracer_provider(TracerProvider(
            resource=Resource.create({"service.name": service_name})))
        backend = "local"  # spans are created (and testable) but not exported
    if app is not None:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

            FastAPIInstrumentor.instrument_app(app, excluded_urls="healthz")
        except ImportError:
            pass
    return backend


@contextmanager
def agent_span(name: str, *, tenant_id: str, channel: str, thread_id: str,
               **attrs: Any) -> Iterator[trace.Span]:
    """Span around one agent run. Tenant/channel attributes make App Insights queries useful:
    `dependencies | where customDimensions["fieldops.tenant_id"] == "t2"`."""
    with tracer.start_as_current_span(name) as span:
        span.set_attribute("fieldops.tenant_id", tenant_id)
        span.set_attribute("fieldops.channel", channel)
        span.set_attribute("fieldops.thread_id", thread_id)
        for k, v in attrs.items():
            if v is not None:
                span.set_attribute(f"fieldops.{k}", v)
        yield span


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def record_run_metrics(row: dict[str, Any]) -> None:
    attrs = {"tenant_id": row["tenant_id"], "channel": row["channel"], "agent": row["agent"]}
    RUN_COST.add(row["cost_usd"], attrs)
    RUN_LATENCY.record(row["latency_ms"], attrs)
    if row["budget_violations"]:
        BUDGET_VIOLATIONS.add(1, attrs)
