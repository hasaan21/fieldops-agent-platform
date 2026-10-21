"""Offline evaluation runner.

    uv run python -m capstone.evals.runner --split dev --offline          # rule-based baseline
    uv run python -m capstone.evals.runner --split dev --prompt v1        # LLM, prompt v1
    uv run python -m capstone.evals.runner --split test --prompt v2 --judge
    uv run python -m capstone.evals.runner --split dev --prompt v2 --langsmith  # LangSmith

Writes artifacts/evals/<run_id>/{results.jsonl, summary.json, report.md}.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from capstone.evals.datasets import load_split
from capstone.evals.evaluators import heuristic
from capstone.evals.target import IntakeTarget

ARTIFACTS = Path("artifacts/evals")


def wilson_interval(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% CI for a proportion. With n=50, 80% accuracy is really 67-89%: say so in reports."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def score_example(ex: dict[str, Any], out: dict[str, Any], judge: Any = None) -> list[dict]:
    ref = ex["reference"]
    results = [r for e in heuristic.ALL if (r := e(out, ref)) is not None]
    if judge is not None and out.get("reply"):
        results += judge.evaluate(ex["inputs"], out)
    return results


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_key: dict[str, list[float]] = defaultdict(list)
    by_slice: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    critical: list[dict[str, str]] = []
    for row in rows:
        fm = row["metadata"].get("failure_mode", "none")
        for r in row["scores"]:
            by_key[r["key"]].append(r["score"])
            by_slice[fm][r["key"]].append(r["score"])
            if r["key"] in {"safety_escalation", "no_sensitive_leak"} and r["score"] < 1:
                critical.append({"id": row["id"], "key": r["key"], "comment": r["comment"]})
    metrics = {k: statistics.fmean(v) for k, v in sorted(by_key.items())}
    n_outcome = len(by_key.get("outcome_correct", []))
    ci = wilson_interval(sum(by_key.get("outcome_correct", [])), n_outcome)
    lat = sorted(r["outputs"]["latency_ms"] for r in rows)
    return {
        "n": len(rows),
        "metrics": metrics,
        "outcome_correct_ci95": ci,
        "slices": {fm: {k: round(statistics.fmean(v), 3) for k, v in d.items()}
                   for fm, d in sorted(by_slice.items())},
        "critical_failures": critical,
        "error_rate": sum(1 for r in rows if r["outputs"].get("error")) / max(1, len(rows)),
        "latency_ms": {"p50": lat[len(lat) // 2] if lat else 0,
                       "p95": lat[int(len(lat) * 0.95) - 1] if lat else 0},
    }


def render_report(summary: dict[str, Any], meta: dict[str, Any]) -> str:
    lines = [f"# Eval report: {meta['config']} on `{meta['split']}`", "",
             f"- run: `{meta['run_id']}` | n={summary['n']} | error rate "
             f"{summary['error_rate']:.1%} | latency p50 {summary['latency_ms']['p50']}ms, "
             f"p95 {summary['latency_ms']['p95']}ms",
             f"- outcome_correct 95% CI: {summary['outcome_correct_ci95'][0]:.1%} - "
             f"{summary['outcome_correct_ci95'][1]:.1%}", "",
             "| metric | score |", "|---|---|"]
    lines += [f"| {k} | {v:.3f} |" for k, v in summary["metrics"].items()]
    keys = ["outcome_correct", "intent_correct", "trajectory_match", "no_sensitive_leak"]
    lines += ["", "## By failure mode", "", "| failure mode | " + " | ".join(keys) + " |",
              "|---" * (len(keys) + 1) + "|"]
    for fm, d in summary["slices"].items():
        lines.append(f"| {fm} | " + " | ".join(f"{d.get(k, float('nan')):.2f}" for k in keys)
                     + " |")
    if summary["critical_failures"]:
        lines += ["", "## Critical failures", ""]
        lines += [f"- `{c['id']}` **{c['key']}**: {c['comment']}"
                  for c in summary["critical_failures"]]
    return "\n".join(lines) + "\n"


def run(split: str, target: IntakeTarget, judge: Any = None, limit: int | None = None,
        workers: int = 4) -> tuple[dict[str, Any], Path]:
    examples = load_split(split)[:limit]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        outputs = list(pool.map(lambda ex: target(ex["inputs"]), examples))
    rows = [{"id": ex["id"], "inputs": ex["inputs"], "metadata": ex["metadata"],
             "reference": ex["reference"], "outputs": out,
             "scores": score_example(ex, out, judge)} for ex, out in zip(examples, outputs,
                                                                         strict=True)]
    run_id = f"{datetime.now(UTC):%Y%m%dT%H%M%S}-{split}-{target.config_id}".replace(":", "_")
    out_dir = ARTIFACTS / run_id
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.jsonl").write_text("".join(json.dumps(r, default=str) + "\n"
                                                   for r in rows))
    summary = summarize(rows)
    meta = {"run_id": run_id, "split": split, "config": target.config_id}
    (out_dir / "summary.json").write_text(json.dumps({**meta, **summary}, indent=2))
    (out_dir / "report.md").write_text(render_report(summary, meta))
    return {**meta, **summary}, out_dir


def run_langsmith(dataset: str, target: IntakeTarget, prefix: str) -> None:
    from langsmith import evaluate

    def wrap(fn):
        def ev(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
            return fn(outputs, reference_outputs) or {"key": fn.__name__, "score": None}
        ev.__name__ = fn.__name__
        return ev

    evaluate(target, data=dataset, evaluators=[wrap(e) for e in heuristic.ALL],
             experiment_prefix=prefix, max_concurrency=4,
             metadata={"config": target.config_id})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev")
    ap.add_argument("--prompt", help="classify prompt version or tag")
    ap.add_argument("--offline", action="store_true", help="rule-based classifier, no LLM")
    ap.add_argument("--judge", action="store_true", help="add LLM-as-judge rubric scores")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--langsmith", action="store_true")
    args = ap.parse_args()
    target = IntakeTarget(prompt_version=args.prompt, offline=args.offline or None)
    if args.langsmith:
        run_langsmith(f"fieldops-intake-{args.split}", target, f"intake-{target.config_id}")
        return
    judge = None
    if args.judge:
        from capstone.core.llm import get_chat_model
        from capstone.evals.evaluators.judge import RubricJudge

        judge = RubricJudge(get_chat_model("judge"))
    summary, out_dir = run(args.split, target, judge, args.limit)
    print((out_dir / "report.md").read_text())
    print(f"artifacts: {out_dir}")


if __name__ == "__main__":
    main()
