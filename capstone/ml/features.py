"""Feature engineering. The same functions run locally (pandas) and on Databricks.

Leakage rules:
- Job features only use information available at intake time (no duration, no first-time-fix).
- Snapshot features only use data up to snapshot_date. The label looks 30 days ahead.
- Splits are by time (train on the past, test on the future), and for assets grouped by
  asset_id, so the same machine never appears in both train and test.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from capstone.data.loader import load_frame

JOB_CATEGORICAL = ["category", "urgency", "asset_type", "tenant_id"]
JOB_NUMERIC = ["asset_age_years", "engineer_experience_years", "hour", "dayofweek", "month",
               "has_error_code", "desc_len"]
SNAPSHOT_NUMERIC = ["age_years", "days_since_service", "service_interval_days",
                    "overdue_ratio", "fault_count_90d", "avg_daily_runtime_hours"]


def to_category(X: pd.DataFrame) -> pd.DataFrame:
    """First pipeline step: raw strings in, pandas categoricals out.

    The served model then accepts plain JSON strings, and the gradient-boosting step's
    `categorical_features="from_dtype"` handles encoding, including unseen categories.
    """
    cols = [c for c in X.columns
            if pd.api.types.is_string_dtype(X[c]) or pd.api.types.is_object_dtype(X[c])]
    return X.astype({c: "category" for c in cols})


def to_text(X: Any) -> list[str]:
    """Accept a DataFrame (what MLflow pyfunc serving sends), a Series or a list of strings."""
    if isinstance(X, pd.DataFrame):
        X = X.iloc[:, 0]
    return [str(x) for x in X]


def text_step():  # -> sklearn FunctionTransformer
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(to_text)


def cast_step():  # -> sklearn FunctionTransformer
    from sklearn.preprocessing import FunctionTransformer

    return FunctionTransformer(to_category, feature_names_out="one-to-one")


def job_frame() -> pd.DataFrame:
    df = load_frame("jobs")
    ts = pd.to_datetime(df["created_at"])
    df = df.assign(created_at=ts, hour=ts.dt.hour, dayofweek=ts.dt.dayofweek, month=ts.dt.month,
                   has_error_code=df["error_code"].notna().astype(int),
                   desc_len=df["description"].str.len())
    return df.sort_values("created_at").reset_index(drop=True)


def snapshot_frame() -> pd.DataFrame:
    df = load_frame("asset_snapshots")
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"])
    df["overdue_ratio"] = (df["days_since_service"] / df["service_interval_days"]).clip(upper=5)
    df["failed_next_30d"] = df["failed_next_30d"].astype(int)
    return df.sort_values("snapshot_date").reset_index(drop=True)


def time_split(df: pd.DataFrame, ts_col: str, test_fraction: float = 0.25
               ) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = df[ts_col].quantile(1 - test_fraction)
    return df[df[ts_col] < cutoff], df[df[ts_col] >= cutoff]
