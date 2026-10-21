"""Close the loop: failing production runs become regression tests.

    # 1. collect failing runs (LangSmith online feedback, or a local eval/results file)
    uv run python -m capstone.evals.feedback_loop collect --project fieldops-prod --hours 24
    uv run python -m capstone.evals.feedback_loop collect \
        --results artifacts/evals/<run>/results.jsonl
    # 2. label them (a human confirms the expected behaviour; the model's output is the default)
    uv run python -m capstone.evals.feedback_loop review
    # 3. fix the prompt/code, then promote labelled cases into the regressions split
    uv run python -m capstone.evals.feedback_loop promote
    # 4. CI runs the regressions split with "fixed stays fixed" thresholds
    uv run python -m capstone.evals.runner --split regressions --offline

The loop in one sentence: online scorer flags a run -> candidate row -> human label ->
prompt/code fix -> CI gate (dev + regressions) -> prompt promoted via registry tag.

Rules that keep the dataset healthy:
- **Dedupe** near-duplicates (embedding cosine) against existing splits; 50 copies of the same
  failure skew every metric.
- **PII**: messages are redacted again before they enter a dataset that lives in git.
- **Provenance**: every row keeps its source run id, tenant and the failing feedback keys.
- **Human in the loop**: the model's own output is never the label without review.
"""

from __future__ import annotations

import argparse
import contextlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from capstone.evals.datasets import DATA_DIR, load_split
from capstone.guardrails import pii
from capstone.rag.embeddings import HashingEmbedder

CANDIDATES = DATA_DIR / "candidates.jsonl"
REGRESSIONS = DATA_DIR / "regressions.jsonl"
FAIL_THRESHOLD = 0.5
DUP_COSINE = 0.92
LABEL_KEYS = ("intent", "category", "urgency", "expected_outcome", "should_escalate")


def _read(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def candidate_from_eval_row(row: dict[str, Any]) -> dict[str, Any] | None:
    failing = [s["key"] for s in row["scores"]
               if s["score"] is not None and s["score"] < FAIL_THRESHOLD]
    if not failing:
        return None
    return {
        "id": f"fb-{row['id']}",
        "inputs": {**row["inputs"], "message": pii.redact(row["inputs"]["message"]).text},
        "reference": row["reference"],   # offline evals already carry a human label
        "metadata": {"source": "eval", "source_id": row["id"], "failing_keys": failing,
                     "failure_mode": row["metadata"].get("failure_mode", "none"),
                     "observed": {"outcome": row["outputs"].get("outcome"),
                                  "classification": row["outputs"].get("classification")},
                     "labelled": True, "collected_at": datetime.now(UTC).isoformat()},
    }


def candidate_from_langsmith_run(run: Any, failing: list[str]) -> dict[str, Any]:
    meta = (run.extra or {}).get("metadata", {})
    msgs = (run.inputs or {}).get("messages") or [""]
    last = msgs[-1]
    text = last.get("content", "") if isinstance(last, dict) else str(last)
    out = run.outputs or {}
    cls = out.get("classification") or {}
    return {
        "id": f"fb-{str(run.id)[:12]}",
        "inputs": {"tenant_id": meta.get("tenant_id", "unknown"),
                   "channel": meta.get("channel", "web"), "sender": "",
                   "message": pii.redact(text).text},
        # proposed label = what the agent did; a reviewer must confirm or correct it
        "reference": {"intent": cls.get("intent"), "category": cls.get("category"),
                      "urgency": cls.get("urgency"), "expected_outcome": out.get("outcome"),
                      "should_escalate": out.get("outcome") == "escalated"},
        "metadata": {"source": "online", "source_id": str(run.id), "failing_keys": failing,
                     "failure_mode": "unlabelled", "labelled": False,
                     "collected_at": datetime.now(UTC).isoformat()},
    }


def fetch_failing_runs(project: str, hours: int, keys: list[str]) -> list[dict[str, Any]]:
    from langsmith import Client

    client = Client()
    since = datetime.now(UTC) - timedelta(hours=hours)
    by_id: dict[str, tuple[Any, list[str]]] = {}
    for key in keys:
        flt = f'and(eq(feedback_key, "{key}"), lt(feedback_score, {FAIL_THRESHOLD}))'
        for run in client.list_runs(project_name=project, is_root=True, start_time=since,
                                    filter=flt):
            by_id.setdefault(str(run.id), (run, []))[1].append(key)
    return [candidate_from_langsmith_run(run, failing) for run, failing in by_id.values()]


def dedupe(new: list[dict[str, Any]], existing: list[dict[str, Any]],
           threshold: float = DUP_COSINE) -> tuple[list[dict[str, Any]], int]:
    """Drop candidates whose message is a near-duplicate of an existing row or of each other."""
    if not new:
        return [], 0
    emb = HashingEmbedder(dim=512)
    pool = emb.embed([r["inputs"]["message"] for r in existing]) if existing else \
        np.zeros((0, 512))
    ids = {r["id"] for r in existing}
    kept: list[dict[str, Any]] = []
    dropped = 0
    for row in new:
        v = emb.embed([row["inputs"]["message"]])[0]
        if row["id"] in ids or (len(pool) and float((pool @ v).max()) >= threshold):
            dropped += 1
            continue
        kept.append(row)
        ids.add(row["id"])
        pool = np.vstack([pool, v[None, :]])
    return kept, dropped


def collect(rows: list[dict[str, Any]], against_splits: bool = True) -> dict[str, int]:
    """`against_splits=False` when the rows come *from* dev/test (offline eval results)."""
    existing = _read(CANDIDATES) + _read(REGRESSIONS)
    for split in ("dev", "test") if against_splits else ():
        with contextlib.suppress(Exception):  # missing/stale splits shouldn't block collection
            existing += load_split(split)
    kept, dropped = dedupe(rows, existing)
    _write(CANDIDATES, _read(CANDIDATES) + kept)
    return {"found": len(rows), "added": len(kept), "duplicates": dropped}


def review(auto_accept: bool = False) -> int:
    """Terminal labelling loop. Enter keeps the proposed value; type a new value to correct."""
    rows = _read(CANDIDATES)
    n = 0
    for row in rows:
        if row["metadata"].get("labelled"):
            continue
        if not auto_accept:
            print("\n" + "=" * 72)
            print(f"[{row['inputs']['tenant_id']}/{row['inputs']['channel']}] "
                  f"{row['inputs']['message']}")
            print(f"failing: {row['metadata']['failing_keys']}")
            for k in LABEL_KEYS:
                ans = input(f"  {k} [{row['reference'].get(k)}]: ").strip()
                if ans:
                    row["reference"][k] = ans.lower() == "true" if k == "should_escalate" \
                        else ans
            if input("  keep this case? [Y/n] ").strip().lower() == "n":
                row["metadata"]["discarded"] = True
        row["metadata"]["labelled"] = True
        row["metadata"]["labelled_at"] = datetime.now(UTC).isoformat()
        n += 1
    _write(CANDIDATES, rows)
    return n


def promote() -> dict[str, int]:
    rows = _read(CANDIDATES)
    ready = [r for r in rows if r["metadata"].get("labelled") and
             not r["metadata"].get("discarded")]
    waiting = [r for r in rows if not r["metadata"].get("labelled")]
    regressions = _read(REGRESSIONS)
    have = {r["id"] for r in regressions}
    for r in ready:
        if r["id"] not in have:
            r["metadata"] |= {"split": "regressions", "promoted_at": datetime.now(UTC).isoformat()}
            regressions.append(r)
    _write(REGRESSIONS, regressions)
    _write(CANDIDATES, waiting)
    return {"promoted": len(ready), "regressions_total": len(regressions),
            "still_unlabelled": len(waiting)}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--results", type=Path, nargs="*", default=[])
    c.add_argument("--project", default="fieldops-prod")
    c.add_argument("--hours", type=int, default=24)
    c.add_argument("--keys", nargs="*", default=["user_score", "judge_safety", "judge_overall",
                                                   "reply_length_ok", "no_sensitive_leak"])
    r = sub.add_parser("review")
    r.add_argument("--auto-accept", action="store_true", help="accept proposed labels (demo)")
    sub.add_parser("promote")
    args = ap.parse_args(argv)
    if args.cmd == "collect":
        if args.results:
            rows = [cand for p in args.results for line in p.read_text().splitlines()
                    if (cand := candidate_from_eval_row(json.loads(line))) is not None]
            print(collect(rows, against_splits=False))
        else:
            print(collect(fetch_failing_runs(args.project, args.hours, args.keys)))
    elif args.cmd == "review":
        print(f"labelled {review(args.auto_accept)} candidates")
    else:
        print(promote())


if __name__ == "__main__":
    main()
