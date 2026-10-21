"""Online scoring and drift detection on live traffic.

    uv run python -m capstone.evals.online score --hours 1 --sample 0.2
    uv run python -m capstone.evals.online drift --baseline-days 7 --current-hours 24

`score`: pulls recent root runs from the LangSmith project, applies reference-free evaluators
(and optionally the LLM judge on a sample) and writes them back as feedback, so they show up on
dashboards and can be filtered ("judge_safety < 0.5 in the last 24h").

`drift`: compares the distribution of outcomes/intents/categories now vs a baseline window using
the Population Stability Index, and the pass rate of each feedback key using a two-proportion
z-test. PSI > 0.2 or |z| > 3 raises an alert.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from typing import Any

from capstone.evals.evaluators import heuristic


def psi(expected: dict[str, float], actual: dict[str, float], eps: float = 1e-4) -> float:
    """Population Stability Index over categorical distributions (given as counts or shares).

    < 0.1 stable, 0.1-0.2 moderate shift, > 0.2 significant shift.
    """
    keys = set(expected) | set(actual)
    te, ta = sum(expected.values()) or 1, sum(actual.values()) or 1
    total = 0.0
    for k in keys:
        e = max(expected.get(k, 0) / te, eps)
        a = max(actual.get(k, 0) / ta, eps)
        total += (a - e) * math.log(a / e)
    return total


def two_proportion_z(p1: float, n1: int, p2: float, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (p2 - p1) / se


def distribution(values: Iterable[str | None]) -> dict[str, float]:
    return dict(Counter(v or "none" for v in values))


def drift_report(baseline_runs: list[dict[str, Any]], current_runs: list[dict[str, Any]]
                 ) -> dict[str, Any]:
    report: dict[str, Any] = {"psi": {}, "alerts": []}
    for field in ["outcome", "intent", "category", "channel"]:
        b = distribution(r.get(field) for r in baseline_runs)
        c = distribution(r.get(field) for r in current_runs)
        value = psi(b, c)
        report["psi"][field] = round(value, 4)
        if value > 0.2:
            report["alerts"].append(f"PSI {field}={value:.2f}: distribution shift")
    keys = {k for r in baseline_runs + current_runs for k in r.get("scores", {})}
    report["pass_rate"] = {}
    for k in sorted(keys):
        bs = [r["scores"][k] for r in baseline_runs if k in r.get("scores", {})]
        cs = [r["scores"][k] for r in current_runs if k in r.get("scores", {})]
        if not bs or not cs:
            continue
        pb, pc = sum(bs) / len(bs), sum(cs) / len(cs)
        z = two_proportion_z(pb, len(bs), pc, len(cs))
        report["pass_rate"][k] = {"baseline": round(pb, 3), "current": round(pc, 3),
                                  "z": round(z, 2)}
        if z < -3:
            report["alerts"].append(f"{k} pass rate dropped {pb:.2f} -> {pc:.2f} (z={z:.1f})")
    return report


def _run_to_record(run: Any) -> dict[str, Any]:
    out = run.outputs or {}
    cls = out.get("classification") or {}
    return {"run_id": str(run.id), "outcome": out.get("outcome"), "intent": cls.get("intent"),
            "category": cls.get("category"), "channel": (run.extra or {}).get("metadata", {})
            .get("channel"), "reply": out.get("reply"), "classification": cls,
            "scores": {f.key: f.score for f in getattr(run, "feedback_stats", None) or []}}


def fetch_runs(project: str, since: datetime, until: datetime | None = None) -> list[Any]:
    from langsmith import Client

    return list(Client().list_runs(project_name=project, is_root=True, start_time=since,
                                   end_time=until, filter='has(tags, "channel:sms") or '
                                   'has(tags, "channel:email") or has(tags, "channel:whatsapp") '
                                   'or has(tags, "channel:voice") or has(tags, "channel:web")'))


def score(project: str, hours: int, sample: float, use_judge: bool) -> int:
    from langsmith import Client

    client = Client()
    runs = fetch_runs(project, datetime.now(UTC) - timedelta(hours=hours))
    judge = None
    if use_judge:
        from capstone.core.llm import get_chat_model
        from capstone.evals.evaluators.judge import RubricJudge

        judge = RubricJudge(get_chat_model("judge"))
    n = 0
    for run in runs:
        if random.random() > sample:
            continue
        rec = _run_to_record(run)
        outputs = {"reply": rec["reply"], "classification": rec["classification"],
                   "channel": rec["channel"] or "web", "outcome": rec["outcome"]}
        results = [r for e in heuristic.REFERENCE_FREE if (r := e(outputs, {})) is not None]
        if judge and rec["reply"]:
            inputs = {"message": str((run.inputs or {}).get("messages", [""])[-1]),
                      "channel": outputs["channel"]}
            results += judge.evaluate(inputs, outputs)
        for r in results:
            client.create_feedback(run.id, key=r["key"], score=r["score"],
                                   comment=r.get("comment"), source_info={"scorer": "online"})
        n += 1
    return n


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score")
    s.add_argument("--project", default="fieldops-prod")
    s.add_argument("--hours", type=int, default=1)
    s.add_argument("--sample", type=float, default=0.2)
    s.add_argument("--judge", action="store_true")
    d = sub.add_parser("drift")
    d.add_argument("--project", default="fieldops-prod")
    d.add_argument("--baseline-days", type=int, default=7)
    d.add_argument("--current-hours", type=int, default=24)
    args = ap.parse_args()
    if args.cmd == "score":
        print(f"scored {score(args.project, args.hours, args.sample, args.judge)} runs")
    else:
        now = datetime.now(UTC)
        cur_start = now - timedelta(hours=args.current_hours)
        base = [_run_to_record(r) for r in fetch_runs(
            args.project, cur_start - timedelta(days=args.baseline_days), cur_start)]
        cur = [_run_to_record(r) for r in fetch_runs(args.project, cur_start)]
        rep = drift_report(base, cur)
        print(rep)
        raise SystemExit(1 if rep["alerts"] else 0)


if __name__ == "__main__":
    main()
