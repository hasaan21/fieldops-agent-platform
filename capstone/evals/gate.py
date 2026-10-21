"""CI eval gate: compare a run summary to absolute thresholds and the committed baseline.

    uv run python -m capstone.evals.gate artifacts/evals/<run>/summary.json \
        --baseline capstone/evals/baselines/job_intake_offline.json [--update-baseline]

Exit code 1 blocks the merge / prompt promotion. Writes gate.md next to the summary for the
PR comment.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

THRESHOLDS = Path(__file__).parent / "thresholds.yaml"


def check(summary: dict[str, Any], baseline: dict[str, Any] | None,
          rules: dict[str, Any]) -> list[tuple[str, bool, str]]:
    m = summary["metrics"]
    results: list[tuple[str, bool, str]] = []
    for key, minimum in rules.get("absolute", {}).items():
        if key not in m:
            continue
        results.append((f"{key} >= {minimum}", m[key] >= minimum - 1e-9, f"{m[key]:.3f}"))
    if baseline:
        bm = baseline["metrics"]
        for key, tol in rules.get("max_regression", {}).items():
            if key in m and key in bm:
                drop = bm[key] - m[key]
                results.append((f"{key} drop <= {tol}", drop <= tol + 1e-9,
                                f"{bm[key]:.3f} -> {m[key]:.3f} ({-drop:+.3f})"))
    if "max_error_rate" in rules:
        results.append((f"error_rate <= {rules['max_error_rate']}",
                        summary["error_rate"] <= rules["max_error_rate"],
                        f"{summary['error_rate']:.3f}"))
    if "max_p95_latency_ms" in rules:
        p95 = summary["latency_ms"]["p95"]
        results.append((f"p95 latency <= {rules['max_p95_latency_ms']}ms",
                        p95 <= rules["max_p95_latency_ms"], f"{p95}ms"))
    return results


def render(results: list[tuple[str, bool, str]], summary: dict[str, Any]) -> str:
    ok = all(r[1] for r in results)
    lines = [f"## Eval gate: {'PASSED' if ok else 'FAILED'}",
             f"`{summary.get('config')}` on `{summary.get('split')}` (n={summary['n']})", "",
             "| rule | result | value |", "|---|---|---|"]
    lines += [f"| {name} | {'pass' if passed else '**FAIL**'} | {val} |"
              for name, passed, val in results]
    if summary.get("critical_failures"):
        lines += ["", "Critical failures:"] + [f"- `{c['id']}` {c['key']}: {c['comment']}"
                                              for c in summary["critical_failures"][:10]]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary", type=Path)
    ap.add_argument("--baseline", type=Path)
    ap.add_argument("--agent", default="job_intake")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args()
    summary = json.loads(args.summary.read_text())
    rules = yaml.safe_load(THRESHOLDS.read_text())[args.agent]
    baseline = json.loads(args.baseline.read_text()) if args.baseline and \
        args.baseline.exists() else None
    results = check(summary, baseline, rules)
    md = render(results, summary)
    (args.summary.parent / "gate.md").write_text(md)
    print(md)
    passed = all(r[1] for r in results)
    if passed and args.update_baseline and args.baseline:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(json.dumps(summary, indent=2))
        print(f"baseline updated: {args.baseline}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
