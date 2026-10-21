import json

import numpy as np
import pandas as pd
import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

from capstone.agents.job_intake.graph import IntakeContext, IntakeDeps, build_graph
from capstone.core.tenancy import TenantIsolationError
from capstone.data.loader import load_table
from capstone.ml.features import job_frame, snapshot_frame, time_split, to_category, to_text
from capstone.ml.serving import ModelService
from capstone.ml.train_duration import cqr_offset
from capstone.ml.train_urgency import expected_calibration_error, select
from capstone.tools.crm_tools import build_registry
from capstone.tools.ml_tools import ML_TOOL_NAMES


@pytest.fixture
def offline_models():
    return ModelService(enabled=False)


def test_pipeline_input_steps_accept_serving_shapes():
    df = pd.DataFrame({"asset_type": ["combi_boiler"], "age": [3.0]})
    assert str(to_category(df)["asset_type"].dtype) == "category"
    assert str(to_category(df)["age"].dtype) == "float64"
    texts = ["boiler leaking", "no power"]
    assert to_text(pd.DataFrame({"description": texts})) == texts
    assert to_text(pd.Series(texts)) == texts


def test_time_split_never_trains_on_the_future():
    df = job_frame()
    train, test = time_split(df, "created_at", 0.2)
    assert train["created_at"].max() <= test["created_at"].min()
    snaps = snapshot_frame()
    assert set(snaps["failed_next_30d"].unique()) <= {0, 1}


def test_conformal_offset_restores_coverage():
    rng = np.random.default_rng(0)
    y = rng.normal(0, 1, 4000)
    lo, hi = np.full_like(y, -0.5), np.full_like(y, 0.5)  # badly under-covering interval
    q = cqr_offset(lo[:2000], hi[:2000], y[:2000], alpha=0.2)
    coverage = np.mean((y[2000:] >= lo[2000:] - q) & (y[2000:] <= hi[2000:] + q))
    assert 0.77 <= coverage <= 0.84


def test_ece_and_calibration_aware_selection():
    classes = np.array(["a", "b"])
    y = np.array(["a"] * 50 + ["b"] * 50)
    perfect = np.array([[1.0, 0.0]] * 50 + [[0.0, 1.0]] * 50)
    overconfident = np.array([[0.99, 0.01]] * 100)  # always "a", 99% sure, right half the time
    assert expected_calibration_error(y, perfect, classes) < 1e-9
    assert expected_calibration_error(y, overconfident, classes) > 0.4
    results = {"nb": {"macro_f1": 0.90, "cv_macro_f1_std": 0.02, "log_loss": 1.8},
               "lr": {"macro_f1": 0.89, "cv_macro_f1_std": 0.02, "log_loss": 0.4},
               "bad": {"macro_f1": 0.70, "cv_macro_f1_std": 0.02, "log_loss": 0.1}}
    assert select(results) == "lr"  # within noise on F1, far better calibrated


def test_model_service_falls_back_transparently(crm, offline_models):
    asset = next(a for a in load_table("assets") if a["tenant_id"] == "t1")
    X = offline_models.duration_features(tenant_id="t1", category="plumbing", urgency="normal",
                                         description="leaking tap", asset=asset)
    est = offline_models.estimate_duration(X)
    assert est["source"] == "fallback"
    assert 0 < est["p10_hours"] <= est["p50_hours"] <= est["p90_hours"]
    snap = offline_models.latest_snapshot(asset["asset_id"])
    assert snap is not None
    risk = offline_models.failure_risk(snap)
    assert risk["source"] == "fallback" and 0 <= risk["p_fail_30d"] <= 1
    assert offline_models.urgency("boiler leaking") is None  # no model, no opinion


def test_ml_tools_are_tenant_scoped(crm, offline_models):
    reg = build_registry(crm, offline_models)
    reg.enable("t1", *ML_TOOL_NAMES)
    own = next(a for a in load_table("assets") if a["tenant_id"] == "t1")
    other = next(a for a in load_table("assets") if a["tenant_id"] == "t2")
    assert reg.call("t1", "asset_failure_risk", {"asset_id": own["asset_id"]})["asset_id"] == \
        own["asset_id"]
    with pytest.raises(TenantIsolationError):
        reg.call("t1", "asset_failure_risk", {"asset_id": other["asset_id"]})
    with pytest.raises(TenantIsolationError):
        reg.call("t1", "estimate_job_duration",
                 {"category": "plumbing", "urgency": "normal", "description": "leak",
                  "asset_id": other["asset_id"]})


def test_agent_attaches_ml_estimates_to_booking_proposal(crm, known_customer, offline_models):
    app = build_graph(IntakeDeps(crm=crm, models=offline_models)).compile(
        checkpointer=InMemorySaver(), store=InMemoryStore())
    ctx = IntakeContext(tenant_id="t1", channel="sms", sender=known_customer["phone"])
    out = app.invoke({"messages": [HumanMessage(
        "The Thermacore X200 boiler is showing error code F22, can you send someone?")]},
        {"configurable": {"thread_id": "ml-1"}}, context=ctx)
    proposal = out["__interrupt__"][0].value["proposal"]
    assert proposal["estimated_hours"]["source"] == "fallback"
    assert "estimate_job_duration" in out["trajectory"]


def test_train_register_and_serve_champion(tmp_path, monkeypatch):
    pytest.importorskip("mlflow")
    from mlflow import MlflowClient

    from capstone.core.config import get_settings
    from capstone.ml import train_failure

    uri = f"sqlite:///{tmp_path}/mlflow.db"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    monkeypatch.setenv("FIELDOPS_MODEL_CARD_DIR", str(tmp_path / "cards"))
    get_settings.cache_clear()
    try:
        MlflowClient(tracking_uri=uri).create_experiment(
            "fieldops/asset_failure", artifact_location=(tmp_path / "artifacts").as_uri())
        out = train_failure.main(register=True)
        assert "champion" in out["registry"]
        assert (tmp_path / "cards" / "fieldops-asset-failure.md").exists()
        svc = ModelService(tracking_uri=uri)
        snap = svc.latest_snapshot(load_table("assets")[0]["asset_id"])
        risk = svc.failure_risk(snap)
        assert risk["source"] == "model" and risk["version"] == "1"
    finally:
        get_settings.cache_clear()


def test_failure_mode_analysis_ranks_by_severity(tmp_path):
    from capstone.analysis.failure_modes import analyse

    def row(i, msg, key, score, fm="none"):
        return {"id": f"x{i}", "inputs": {"tenant_id": "t1", "channel": "sms", "message": msg},
                "metadata": {"failure_mode": fm}, "outputs": {"outcome": "booked"},
                "scores": [{"key": key, "score": score, "comment": ""}]}

    rows = [row(i, f"gas smell in kitchen {i}", "safety_escalation", 0.0, "gas") for i in range(3)]
    rows += [row(10 + i, f"reply too long about filters {i}", "reply_length_ok", 0.0)
             for i in range(6)]
    rows += [row(20 + i, f"all good {i}", "outcome_correct", 1.0) for i in range(5)]
    path = tmp_path / "results.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    result = analyse([str(path)])
    assert result["runs"] == 14 and result["failures"] == 9
    top = result["clusters"].iloc[0]
    assert top["top_metric"] == "safety_escalation"  # 3 safety misses outrank 6 long replies


def test_stale_eval_dataset_is_detected(tmp_path, monkeypatch):
    from capstone.evals import datasets

    (tmp_path / "dev.jsonl").write_text("{}\n")
    (tmp_path / "_meta.json").write_text(json.dumps({"world_fingerprint": "not-this-world"}))
    monkeypatch.setattr(datasets, "DATA_DIR", tmp_path)
    with pytest.raises(datasets.StaleDatasetError):
        datasets.load_split("dev")
