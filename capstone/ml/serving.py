"""Online inference for the FieldOps models, used by agent tools.

Loads `models:/<name>@champion` from the MLflow registry. On Databricks the same URIs point at
Unity Catalog (`models:/main.fieldops.duration@champion`), or you call the Model Serving endpoint
over HTTP instead (see databricks/notebooks/04_serving.py).

Design rules:
- Never fail the agent because a model is unavailable. Fall back to a transparent heuristic
  and say so (`source: "fallback"`), so traces and evals can tell the two paths apart.
- Every response carries the model name and version, so a bad answer in a trace can be tied
  to the exact registry version.
- Feature construction lives here, next to the model, not in the agent prompt.
"""

from __future__ import annotations

import logging
import math
import threading
from datetime import datetime
from functools import cached_property
from typing import Any

import pandas as pd

from capstone.core.config import get_settings
from capstone.data.loader import load_frame
from capstone.ml.features import JOB_CATEGORICAL, JOB_NUMERIC, SNAPSHOT_NUMERIC

log = logging.getLogger(__name__)

DURATION_MODEL = "fieldops-duration"
FAILURE_MODEL = "fieldops-asset-failure"
URGENCY_MODEL = "fieldops-urgency"
DEFAULT_ENGINEER_EXPERIENCE = 5.0  # not known until an engineer is assigned


class _Loaded:
    def __init__(self, model: Any, version: str, run_id: str, extras: dict[str, Any]) -> None:
        self.model, self.version, self.run_id, self.extras = model, version, run_id, extras


class ModelService:
    def __init__(self, tracking_uri: str | None = None, alias: str = "champion",
                 enabled: bool = True) -> None:
        self.tracking_uri = tracking_uri or get_settings().mlflow_tracking_uri
        self.alias = alias
        self.enabled = enabled
        self._cache: dict[str, _Loaded | None] = {}
        self._lock = threading.Lock()

    # -- loading
    def _load(self, name: str) -> _Loaded | None:
        if name in self._cache:
            return self._cache[name]
        with self._lock:
            if name not in self._cache:
                self._cache[name] = self._load_uncached(name)
        return self._cache[name]

    def _load_uncached(self, name: str) -> _Loaded | None:
        loaded: _Loaded | None = None
        if self.enabled:
            try:
                import mlflow
                from mlflow import MlflowClient

                mlflow.set_tracking_uri(self.tracking_uri)
                mlflow.set_registry_uri(self.tracking_uri)
                client = MlflowClient()
                mv = client.get_model_version_by_alias(name, self.alias)
                model = mlflow.sklearn.load_model(f"models:/{name}@{self.alias}")
                extras: dict[str, Any] = {}
                if name == DURATION_MODEL:
                    run = client.get_run(str(mv.run_id))
                    extras["p10"] = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/model_p10")
                    extras["p90"] = mlflow.sklearn.load_model(f"runs:/{mv.run_id}/model_p90")
                    extras["conformal_q"] = float(run.data.params.get("conformal_offset_h", 0))
                loaded = _Loaded(model, str(mv.version), str(mv.run_id), extras)
            except Exception as exc:  # registry down, model missing, mlflow not installed
                log.warning("model %s unavailable, using fallback: %s", name, exc)
        return loaded

    def status(self) -> dict[str, str | None]:
        return {n: (m.version if (m := self._load(n)) else None)
                for n in (DURATION_MODEL, FAILURE_MODEL, URGENCY_MODEL)}

    # -- duration
    @cached_property
    def _category_medians(self) -> dict[str, float]:
        jobs = load_frame("jobs")
        return jobs.groupby("category")["duration_hours"].median().to_dict()

    def duration_features(self, *, tenant_id: str, category: str, urgency: str,
                          description: str, asset: dict[str, Any] | None = None,
                          when: datetime | None = None, error_code: str | None = None,
                          engineer_experience_years: float | None = None) -> pd.DataFrame:
        when = when or datetime.now()
        age = 0.0
        if asset and asset.get("installed_at"):
            age = (when - datetime.fromisoformat(asset["installed_at"])).days / 365
        row = {
            "category": category, "urgency": urgency,
            "asset_type": (asset or {}).get("asset_type", "unknown"), "tenant_id": tenant_id,
            "asset_age_years": round(age, 2),
            "engineer_experience_years": engineer_experience_years or DEFAULT_ENGINEER_EXPERIENCE,
            "hour": when.hour, "dayofweek": when.weekday(), "month": when.month,
            "has_error_code": int(bool(error_code)), "desc_len": len(description),
        }
        return pd.DataFrame([row])[JOB_CATEGORICAL + JOB_NUMERIC]

    def estimate_duration(self, X: pd.DataFrame) -> dict[str, Any]:
        m = self._load(DURATION_MODEL)
        if m is None:
            med = float(self._category_medians.get(str(X["category"].iloc[0]), 2.0))
            return {"p50_hours": round(med, 2), "p10_hours": round(med * 0.6, 2),
                    "p90_hours": round(med * 1.6, 2), "source": "fallback",
                    "method": "category median"}
        q = m.extras["conformal_q"]
        p50 = float(m.model.predict(X)[0])
        p10 = float(m.extras["p10"].predict(X)[0]) - q
        p90 = float(m.extras["p90"].predict(X)[0]) + q
        p10, p90 = max(0.25, min(p10, p50)), max(p90, p50)  # no quantile crossing
        return {"p50_hours": round(p50, 2), "p10_hours": round(p10, 2),
                "p90_hours": round(p90, 2), "source": "model", "model": DURATION_MODEL,
                "version": m.version}

    # -- failure risk
    def latest_snapshot(self, asset_id: str) -> pd.DataFrame | None:
        snaps = load_frame("asset_snapshots")
        rows = snaps[snaps["asset_id"] == asset_id]
        if rows.empty:
            return None
        row = rows.sort_values("snapshot_date").iloc[[-1]].copy()
        row["overdue_ratio"] = (row["days_since_service"] / row["service_interval_days"]).clip(
            upper=5)
        return row[SNAPSHOT_NUMERIC + ["asset_type"]].reset_index(drop=True)

    def failure_risk(self, X: pd.DataFrame) -> dict[str, Any]:
        m = self._load(FAILURE_MODEL)
        drivers = {"overdue_ratio": round(float(X["overdue_ratio"].iloc[0]), 2),
                   "fault_count_90d": int(X["fault_count_90d"].iloc[0]),
                   "age_years": round(float(X["age_years"].iloc[0]), 1)}
        if m is None:
            z = -3.0 + 0.8 * drivers["overdue_ratio"] + 0.5 * drivers["fault_count_90d"] \
                + 0.05 * drivers["age_years"]
            p, source, extra = 1 / (1 + math.exp(-z)), "fallback", {}
        else:
            p = float(m.model.predict_proba(X)[0, 1])
            source, extra = "model", {"model": FAILURE_MODEL, "version": m.version}
        band = "high" if p >= 0.3 else "medium" if p >= 0.12 else "low"
        return {"p_fail_30d": round(p, 3), "band": band, "drivers": drivers,
                "source": source, **extra}

    # -- urgency (second opinion to the LLM classifier)
    def urgency(self, description: str) -> dict[str, Any] | None:
        m = self._load(URGENCY_MODEL)
        if m is None:
            return None
        X = pd.DataFrame({"description": [description]})
        label = str(m.model.predict(X)[0])
        out: dict[str, Any] = {"urgency": label, "model": URGENCY_MODEL, "version": m.version}
        if hasattr(m.model, "predict_proba"):
            probs = m.model.predict_proba(X)[0]
            out["confidence"] = round(float(max(probs)), 3)
        return out


_service: ModelService | None = None


def get_model_service() -> ModelService:
    global _service
    if _service is None:
        _service = ModelService()
    return _service
