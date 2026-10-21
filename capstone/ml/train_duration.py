"""Job duration forecast (hours on site) with prediction intervals.

    uv run python -m capstone.ml.train_duration

Used by the booking flow to size the slot and by scheduling to plan the day. Point estimates
aren't enough for scheduling, so we also train p10/p90 quantile models and report interval
coverage (it should be about 80%).
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline

from capstone.ml.common import log_sklearn, promote_if_better, setup, write_model_card
from capstone.ml.features import JOB_CATEGORICAL, JOB_NUMERIC, cast_step, job_frame, time_split

MODEL_NAME = "fieldops-duration"
FEATURES = JOB_CATEGORICAL + JOB_NUMERIC


def cqr_offset(lo: np.ndarray, hi: np.ndarray, y: np.ndarray, alpha: float) -> float:
    """Conformalized quantile regression: widen [lo, hi] by q to get 1-alpha coverage.

    Holds without distributional assumptions if calibration and test data are exchangeable,
    which time drift weakens. Recalibrate on recent data.
    """
    scores = np.maximum(lo - y, y - hi)
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


def main(register: bool = True) -> dict[str, Any]:
    setup("duration")
    df = job_frame()
    train, test = time_split(df, "created_at")
    Xtr, ytr = train[FEATURES], train["duration_hours"]
    Xte, yte = test[FEATURES], test["duration_hours"]

    baseline = test["category"].map(train.groupby("category")["duration_hours"].median())
    metrics: dict[str, float] = {"baseline_category_median_mae": mean_absolute_error(yte,
                                                                                     baseline)}
    params = {"max_iter": 400, "learning_rate": 0.04, "max_leaf_nodes": 15,
              "min_samples_leaf": 20, "categorical_features": "from_dtype", "random_state": 0}
    with mlflow.start_run(run_name="hgb_duration") as run:
        def gbm(**kw: Any) -> Pipeline:
            return make_pipeline(cast_step(), HistGradientBoostingRegressor(**params, **kw))

        model = gbm(loss="absolute_error").fit(Xtr, ytr)
        # Quantile models are fit on the older part of train; the newest 20% calibrates them
        # (conformalized quantile regression), because raw quantile GBMs under-cover.
        fit_part, cal_part = time_split(train, "created_at", 0.2)
        lo = gbm(loss="quantile", quantile=0.1).fit(fit_part[FEATURES], fit_part["duration_hours"])
        hi = gbm(loss="quantile", quantile=0.9).fit(fit_part[FEATURES], fit_part["duration_hours"])
        conformal_q = cqr_offset(lo.predict(cal_part[FEATURES]), hi.predict(cal_part[FEATURES]),
                                 cal_part["duration_hours"].to_numpy(), alpha=0.2)
        pred = model.predict(Xte)
        p10, p90 = lo.predict(Xte), hi.predict(Xte)
        metrics |= {
            "mae": mean_absolute_error(yte, pred),
            "rmse": float(np.sqrt(mean_squared_error(yte, pred))),
            "mape": float(np.mean(np.abs((yte - pred) / yte))),
            "raw_p10_p90_coverage": float(np.mean((yte >= p10) & (yte <= p90))),
            "conformal_offset_h": conformal_q,
            "p10_p90_coverage": float(np.mean((yte >= p10 - conformal_q)
                                              & (yte <= p90 + conformal_q))),
            "mean_interval_width_h": float(np.mean(p90 - p10 + 2 * conformal_q)),
        }
        metrics["mae_improvement_vs_baseline"] = 1 - metrics["mae"] / \
            metrics["baseline_category_median_mae"]
        perm = permutation_importance(model, Xte, yte, n_repeats=5, random_state=0,
                                      scoring="neg_mean_absolute_error")
        importance = pd.Series(perm.importances_mean, index=FEATURES).sort_values(
            ascending=False)
        mlflow.log_params({**params, "features": ",".join(FEATURES),
                           "conformal_offset_h": conformal_q})
        mlflow.log_metrics(metrics)
        mlflow.log_text(importance.to_string(), "feature_importance.txt")
        log_sklearn(model, "model", Xte.head(3))
        log_sklearn(lo, "model_p10", Xte.head(3))
        log_sklearn(hi, "model_p90", Xte.head(3))
        run_id = run.info.run_id

    msg = "not registered"
    if register:
        mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
        msg = promote_if_better(MODEL_NAME, mv.version, "mae", metrics["mae"],
                                higher_is_better=False)
    write_model_card(MODEL_NAME, {
        "Intended use": "Estimate on-site hours for a new job to size slots; p10/p90 for "
                        "schedule buffers.",
        "Data": f"Synthetic jobs, time split ({len(Xtr)} train / {len(Xte)} test). Features "
                "known at intake only: " + ", ".join(FEATURES),
        "Test metrics": metrics,
        "Top features": importance.head(6).to_dict(),
        "Registry": msg,
        "Limitations": ["Engineer experience is assumed known at booking (assigned engineer).",
                        "No travel time; add route-distance features for real scheduling.",
                        "Quantile crossing is possible; clip p10 <= p50 <= p90 at serve time.",
                        f"Serve intervals as [p10 - {conformal_q:.2f}, p90 + {conformal_q:.2f}] "
                        "(conformal offset); recalibrate monthly."],
    })
    return {"metrics": metrics, "registry": msg}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
