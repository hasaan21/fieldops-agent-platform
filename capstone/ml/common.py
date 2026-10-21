"""Shared MLflow plumbing: experiments, champion/challenger promotion, model cards."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import mlflow  # noqa: E402
from mlflow import MlflowClient  # noqa: E402

from capstone.core.config import get_settings  # noqa: E402

MODEL_CARD_DIR = Path("docs/model_cards")

# MLflow 3 saves sklearn models with skops, which refuses to load types it hasn't been told to
# trust (pickle-style deserialisation is a code-execution risk). We list exactly the types our
# own pipelines use, never "trust everything".
SKOPS_TRUSTED = [
    "capstone.ml.features.to_category",
    "capstone.ml.features.to_text",
    "functools.partial",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor",
    "sklearn.ensemble._hist_gradient_boosting.binning._BinMapper",
    "sklearn.utils.validation.check_array",
]


def log_sklearn(model: Any, name: str = "model", input_example: Any = None) -> None:
    mlflow.sklearn.log_model(model, name=name, input_example=input_example,
                             skops_trusted_types=SKOPS_TRUSTED)


def setup(experiment: str) -> None:
    mlflow.set_tracking_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_registry_uri(get_settings().mlflow_tracking_uri)
    mlflow.set_experiment(f"fieldops/{experiment}")


def promote_if_better(model_name: str, version: str, metric: str, value: float,
                      higher_is_better: bool = True) -> str:
    """Set `challenger` on the new version; move `champion` only if it beats the current one.

    In Databricks Unity Catalog this is the same API (aliases on UC models), and serving
    endpoints pinned to `@champion` pick up the new version without a redeploy.
    """
    client = MlflowClient()
    client.set_registered_model_alias(model_name, "challenger", version)
    try:
        champ = client.get_model_version_by_alias(model_name, "champion")
        champ_value = client.get_run(str(champ.run_id)).data.metrics.get(metric)
    except Exception:
        champ_value = None
    better = champ_value is None or (value > champ_value if higher_is_better
                                     else value < champ_value)
    if better:
        client.set_registered_model_alias(model_name, "champion", version)
        return f"promoted v{version} to champion ({metric}={value:.4f} vs {champ_value})"
    return f"kept champion ({metric}={champ_value:.4f}); v{version} is challenger ({value:.4f})"


def write_model_card(name: str, card: dict[str, Any]) -> Path:
    card_dir = Path(os.environ.get("FIELDOPS_MODEL_CARD_DIR", MODEL_CARD_DIR))
    card_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# Model card: {name}", ""]
    for section, body in card.items():
        lines += [f"## {section}", ""]
        if isinstance(body, dict):
            lines += ["| metric | value |", "|---|---|"]
            lines += [f"| {k} | {v:.4f} |" if isinstance(v, float) else f"| {k} | {v} |"
                      for k, v in body.items()]
        elif isinstance(body, list):
            lines += [f"- {b}" for b in body]
        else:
            lines.append(str(body))
        lines.append("")
    path = card_dir / f"{name}.md"
    path.write_text("\n".join(lines))
    (card_dir / f"{name}.json").write_text(json.dumps(card, indent=2, default=str))
    return path
