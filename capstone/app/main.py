"""FieldOps API.

    uv run uvicorn capstone.app.main:app --reload
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel

from capstone.agents.job_intake.schemas import ReviewDecision
from capstone.app.runtime import Runtime, get_runtime
from capstone.core.config import get_settings
from capstone.observability.otel import setup_tracing

app = FastAPI(title="FieldOps Agent Platform", version="0.1.0")
setup_tracing("fieldops-api", get_settings().applicationinsights_connection_string, app)


def tenant_from_key(x_api_key: Annotated[str, Header()]) -> str:
    tenant = get_settings().api_key_map().get(x_api_key)
    if not tenant:
        raise HTTPException(401, "invalid API key")
    return tenant


Tenant = Annotated[str, Depends(tenant_from_key)]
Rt = Annotated[Runtime, Depends(get_runtime)]


class InboundMessage(BaseModel):
    channel: Literal["sms", "whatsapp", "email", "voice", "web"] = "web"
    sender: str
    text: str
    message_id: str | None = None


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness: the process is up. Cheap; never touches dependencies."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz(rt: Rt) -> dict[str, Any]:
    """Readiness: dependencies are usable. Container Apps stops routing traffic if this fails."""
    checks: dict[str, Any] = {"graph": rt.intake is not None, "prompt": rt.prompt_ref}
    if get_settings().database_url:
        try:
            rt.store.search(("healthcheck",), limit=1)
            checks["postgres"] = True
        except Exception as e:  # report, don't crash the probe
            checks["postgres"] = f"error: {type(e).__name__}"
    failed = [k for k, v in checks.items() if v is False or str(v).startswith("error")]
    if failed:
        raise HTTPException(503, checks)
    return {"status": "ready", **checks}


@app.get("/v1/usage")
def usage(tenant: Tenant, rt: Rt) -> dict[str, Any]:
    """Cost, tokens, latency percentiles and budget violations for the caller's tenant."""
    return rt.ledger.summary(tenant)


@app.post("/v1/agents/job-intake/messages")
def post_message(body: InboundMessage, tenant: Tenant, rt: Rt) -> dict[str, Any]:
    return rt.handle_message(tenant, body.channel, body.sender, body.text, body.message_id)


@app.get("/v1/reviews")
def list_reviews(tenant: Tenant, rt: Rt) -> list[dict[str, Any]]:
    return rt.reviews.pending(tenant)


@app.post("/v1/reviews/{thread_id}")
def decide_review(thread_id: str, decision: ReviewDecision, tenant: Tenant,
                  rt: Rt) -> dict[str, Any]:
    try:
        return rt.resume(tenant, thread_id, decision.model_dump())
    except PermissionError as e:
        raise HTTPException(403, str(e)) from e
    except KeyError as e:
        raise HTTPException(404, "no pending review for this thread") from e


def _include_optional_routers() -> None:
    from capstone.app import webhooks

    app.include_router(webhooks.router)


_include_optional_routers()
