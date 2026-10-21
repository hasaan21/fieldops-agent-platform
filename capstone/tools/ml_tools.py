"""ML models exposed as agent tools.

The LLM passes IDs and plain fields; the server resolves the asset through the tenant-scoped
CRM (so a model call can never read another tenant's asset), builds features, and returns a
small JSON answer with the model version for traceability.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from capstone.core.crm import CRMRepository
from capstone.ml.serving import ModelService
from capstone.tools.crm_tools import CATEGORY_ENUM, URGENCY_ENUM
from capstone.tools.registry import ToolRegistry, ToolSpec

ML_TOOL_NAMES = ("estimate_job_duration", "asset_failure_risk")


def register_ml_tools(reg: ToolRegistry, crm: CRMRepository, models: ModelService) -> None:
    def duration(tenant_id: str, a: dict[str, Any]) -> dict[str, Any]:
        asset = crm.get_asset(tenant_id, a["asset_id"]) if a.get("asset_id") else None
        when = datetime.fromisoformat(a["slot_start"]) if a.get("slot_start") else crm.now
        X = models.duration_features(tenant_id=tenant_id, category=a["category"],
                                     urgency=a["urgency"], description=a["description"],
                                     asset=asset, when=when, error_code=a.get("error_code"))
        return models.estimate_duration(X)

    def failure(tenant_id: str, a: dict[str, Any]) -> dict[str, Any]:
        crm.get_asset(tenant_id, a["asset_id"])  # raises TenantIsolationError across tenants
        X = models.latest_snapshot(a["asset_id"])
        if X is None:
            return {"error": "no telemetry for this asset"}
        return {"asset_id": a["asset_id"], **models.failure_risk(X)}

    reg.register(ToolSpec(
        name="estimate_job_duration",
        description="Estimate on-site hours for a job (median and p10-p90 range) to size the "
                    "booking slot. Use after classifying the job, before proposing a slot.",
        kind="python",
        input_schema={"type": "object", "required": ["category", "urgency", "description"],
                      "additionalProperties": False,
                      "properties": {"category": {"type": "string", "enum": CATEGORY_ENUM},
                                     "urgency": {"type": "string", "enum": URGENCY_ENUM},
                                     "description": {"type": "string"},
                                     "asset_id": {"type": "string"},
                                     "error_code": {"type": "string"},
                                     "slot_start": {"type": "string"}}}),
        duration)
    reg.register(ToolSpec(
        name="asset_failure_risk",
        description="Probability that a customer's asset fails in the next 30 days, with the "
                    "main drivers. Use to suggest a maintenance visit, never to diagnose.",
        kind="python",
        input_schema={"type": "object", "required": ["asset_id"], "additionalProperties": False,
                      "properties": {"asset_id": {"type": "string"}}}),
        failure)
