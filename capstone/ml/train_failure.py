"""Predictive maintenance: probability that an asset fails within 30 days.

    uv run python -m capstone.ml.train_failure

Framing: discrete-time hazard. Each monthly snapshot is a row; the label is "failure in the next
30 days". A survival model (Cox, or a discrete-time hazard with time-varying covariates) is the
stretch goal.

What matters operationally: maintenance teams can inspect maybe the top 5% riskiest assets per
month, so precision@5% and lift matter more than ROC-AUC. Failures are rare (imbalanced), so
report PR-AUC, not accuracy.

Guardrails on the model itself: monotone constraints. Risk must not *decrease* as overdue
servicing, recent faults or age increase. This keeps explanations sane for customers and
regulators, and usually costs little accuracy.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from capstone.ml.common import log_sklearn, promote_if_better, setup, write_model_card
from capstone.ml.features import SNAPSHOT_NUMERIC, cast_step, snapshot_frame, time_split

MODEL_NAME = "fieldops-asset-failure"
MONOTONE = {"age_years": 1, "overdue_ratio": 1, "fault_count_90d": 1, "days_since_service": 1}


def precision_at_k(y: np.ndarray, p: np.ndarray, frac: float = 0.05) -> float:
    k = max(1, int(len(p) * frac))
    top = np.argsort(-p)[:k]
    return float(y[top].mean())


def calibration_table(y: np.ndarray, p: np.ndarray, bins: int = 5) -> pd.DataFrame:
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(
        predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))


def candidates(feats: list[str]) -> dict[str, Pipeline]:
    """Both take the same raw frame (numeric + asset_type strings), so either can be served."""
    numeric = [f for f in feats if f in SNAPSHOT_NUMERIC]
    linear = ColumnTransformer([
        ("num", StandardScaler(), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["asset_type"]),
    ])
    return {
        "logreg": make_pipeline(linear, LogisticRegression(max_iter=1000)),
        "hgb_monotone": make_pipeline(cast_step(), HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=40,
            monotonic_cst={f: c for f, c in MONOTONE.items() if f in feats},
            categorical_features="from_dtype", random_state=0)),
    }


def main(register: bool = True) -> dict[str, Any]:
    setup("asset_failure")
    df = snapshot_frame()
    train, test = time_split(df, "snapshot_date", 0.3)
    # grouped: hold out 20% of assets entirely, so the test set has never-seen machines too
    rng = np.random.default_rng(0)
    assets = df["asset_id"].unique()
    held = set(rng.choice(assets, size=len(assets) // 5, replace=False))
    train = train[~train["asset_id"].isin(held)]
    feats = SNAPSHOT_NUMERIC + ["asset_type"]
    Xtr, ytr = train[feats], train["failed_next_30d"].to_numpy()
    Xte, yte = test[feats], test["failed_next_30d"].to_numpy()
    base_rate = float(ytr.mean())

    unseen = test["asset_id"].isin(held).to_numpy()
    results: dict[str, dict[str, float]] = {}
    models: dict[str, Pipeline] = {}
    runs: dict[str, str] = {}
    for name, model in candidates(feats).items():
        with mlflow.start_run(run_name=name) as run:
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            m = {"pr_auc": average_precision_score(yte, p), "roc_auc": roc_auc_score(yte, p),
                 "brier": brier_score_loss(yte, p), "precision_at_5pct": precision_at_k(yte, p),
                 "base_rate": float(yte.mean())}
            m["lift_at_5pct"] = m["precision_at_5pct"] / max(m["base_rate"], 1e-9)
            m["pr_auc_unseen_assets"] = average_precision_score(yte[unseen], p[unseen])
            mlflow.log_params({"model": name, "monotone": str(MONOTONE),
                               "train_base_rate": base_rate})
            mlflow.log_metrics(m)
            mlflow.log_text(calibration_table(yte, p).to_string(), "calibration.txt")
            log_sklearn(model, "model", Xte.head(3))
            results[name], models[name], runs[name] = m, model, run.info.run_id

    best = max(results, key=lambda k: results[k]["pr_auc"])
    msg = "not registered"
    if register:
        mv = mlflow.register_model(f"runs:/{runs[best]}/model", MODEL_NAME)
        msg = promote_if_better(MODEL_NAME, mv.version, "pr_auc", results[best]["pr_auc"])
    perm = permutation_importance(models[best], Xte, yte, n_repeats=5, random_state=0,
                                  scoring="average_precision")
    importance = pd.Series(perm.importances_mean, index=feats).sort_values(ascending=False)
    write_model_card(MODEL_NAME, {
        "Intended use": "Rank assets by 30-day failure risk to prioritise proactive "
                        "maintenance visits; exposed to agents as the `asset_failure_risk` tool.",
        "Data": f"{len(df)} monthly snapshots; time split + 20% of assets held out entirely. "
                f"Train base rate {base_rate:.3f}.",
        **{f"Test metrics: {k}": v for k, v in results.items()},
        "Selected": f"{best} (higher PR-AUC). The synthetic hazard is logistic in its "
                    "features, so a linear model matching the generating process is expected "
                    "to compete; on real telemetry with interactions, boosting usually wins.",
        "Top features": importance.head(6).to_dict(),
        "Registry": msg,
        "Limitations": ["Synthetic hazard: real telemetry (pressure, temperature, cycles) is "
                        "far noisier.",
                        "Snapshots of the same asset are correlated; CIs need cluster "
                        "bootstrap.",
                        "Survival framing (time-to-failure) would use censored data better."],
    })
    return {"results": results, "registry": msg}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
