"""Did the ML tools make the agent better? With vs without, on the same data.

    uv run python -m capstone.ml.train_duration && ... (train the models first)
    uv run python -m capstone.evals.ml_ablation

Two comparisons, because "smarter" must mean a business metric, not a vibe:

1. Agent eval (dev split): the standard heuristic metrics with and without ML tools, plus what
   the tools added to bookings (duration estimates, maintenance flags). Guards against the ML
   path *hurting* triage.
2. Slot sizing (historical test jobs): book slots using the category median (today's policy)
   vs the model's conformal p90. Reports overrun rate (job runs past its slot, so the next
   customer waits) against booked hours (capacity cost). This is where the model earns its keep.

Writes docs/eval-reports/ml-ablation.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from capstone.agents.job_intake.graph import default_deps
from capstone.core.crm import CRMRepository
from capstone.evals.runner import run
from capstone.evals.target import IntakeTarget
from capstone.ml.features import JOB_CATEGORICAL, JOB_NUMERIC, job_frame, time_split
from capstone.ml.serving import DURATION_MODEL, ModelService

REPORT = Path("docs/eval-reports/ml-ablation.md")
FEATURES = JOB_CATEGORICAL + JOB_NUMERIC


def agent_ablation(models: ModelService, split: str = "dev",
                   limit: int | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, svc in [("without_ml", None), ("with_ml", models)]:
        deps = default_deps(CRMRepository(), offline=True)
        deps.models = svc
        summary, out_dir = run(split, IntakeTarget(deps), limit=limit)
        rows = [json.loads(line)
                for line in (out_dir / "results.jsonl").read_text().splitlines()]
        proposals = [r["outputs"]["proposal"] for r in rows if r["outputs"].get("proposal")]
        flags = [f for r in rows for f in r["outputs"].get("flags", [])]
        out[label] = {
            "metrics": summary["metrics"],
            "error_rate": summary["error_rate"],
            "latency_p50_ms": summary["latency_ms"]["p50"],
            "bookings": len(proposals),
            "with_duration_estimate": sum(1 for p in proposals if p.get("estimated_hours")),
            "with_asset_risk": sum(1 for p in proposals if p.get("asset_risk")),
            "maintenance_suggested": flags.count("maintenance_suggested"),
            "ml_urgency_upgrades": flags.count("ml_urgency_upgrade"),
            "ml_urgency_disagreements": sum(1 for f in flags
                                            if f.startswith("ml_urgency_disagree")),
        }
    return out


def slot_sizing(models: ModelService) -> dict[str, Any]:
    df = job_frame()
    train, test = time_split(df, "created_at")
    y = test["duration_hours"].to_numpy()
    medians = train.groupby("category")["duration_hours"].median()
    policies: dict[str, np.ndarray] = {
        "fixed_2h": np.full(len(test), 2.0),
        "category_median": test["category"].map(medians).to_numpy(dtype=float),
        "category_p90": test["category"].map(
            train.groupby("category")["duration_hours"].quantile(0.9)).to_numpy(dtype=float),
    }
    m = models._load(DURATION_MODEL)
    if m is not None:
        X = test[FEATURES]
        policies["model_p50"] = m.model.predict(X)
        policies["model_conformal_p90"] = m.extras["p90"].predict(X) + m.extras["conformal_q"]
    return {name: {"overrun_rate": float(np.mean(y > slot)),
                   "mean_booked_hours": float(np.mean(slot)),
                   "mean_idle_hours": float(np.mean(np.clip(slot - y, 0, None)))}
            for name, slot in policies.items()}


def render(agent: dict[str, Any], sizing: dict[str, Any], status: dict[str, Any]) -> str:
    keys = sorted(set(agent["without_ml"]["metrics"]) | set(agent["with_ml"]["metrics"]))
    lines = ["# ML tools ablation: does the agent get smarter?", "",
             f"Registry versions used: `{status}`", "",
             "## 1. Agent eval (dev split, offline classifier)", "",
             "| metric | without ML | with ML | delta |", "|---|---|---|---|"]
    for k in keys:
        a = agent["without_ml"]["metrics"].get(k, float("nan"))
        b = agent["with_ml"]["metrics"].get(k, float("nan"))
        lines.append(f"| {k} | {a:.3f} | {b:.3f} | {b - a:+.3f} |")
    lines += ["", "| what the tools added | without ML | with ML |", "|---|---|---|"]
    for k in ["bookings", "with_duration_estimate", "with_asset_risk", "maintenance_suggested",
              "ml_urgency_upgrades", "ml_urgency_disagreements", "latency_p50_ms"]:
        lines.append(f"| {k} | {agent['without_ml'][k]} | {agent['with_ml'][k]} |")
    lines += ["", "## 2. Slot sizing on historical test jobs", "",
              "| policy | overrun rate | mean booked h | mean idle h |", "|---|---|---|---|"]
    for name, r in sizing.items():
        lines.append(f"| {name} | {r['overrun_rate']:.1%} | {r['mean_booked_hours']:.2f} | "
                     f"{r['mean_idle_hours']:.2f} |")
    lines += ["", "## Reading this", "",
              "- Triage metrics should be flat: the ML second opinion only upgrades to "
              "emergency at high confidence. A drop here would block the change.",
              "- The win is in slot sizing: compare policies at similar booked hours. A "
              "model that cuts overruns without booking more hours frees real capacity.",
              "- Synthetic data caveat: durations come from a known generator, so the model "
              "has an easier job than on real engineer timesheets."]
    return "\n".join(lines) + "\n"


def main(limit: int | None = None) -> dict[str, Any]:
    models = ModelService()
    agent = agent_ablation(models, limit=limit)
    sizing = slot_sizing(models)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(render(agent, sizing, models.status()))
    return {"agent": agent, "slot_sizing": sizing}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2, default=str))
