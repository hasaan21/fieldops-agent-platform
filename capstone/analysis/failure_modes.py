"""Failure-mode analysis: which problems should we fix first?

    uv run python -m capstone.analysis.failure_modes                      # latest eval runs
    uv run python -m capstone.analysis.failure_modes artifacts/evals/<run>/results.jsonl ...

The weekly loop on an agent platform: pull failing runs (offline evals, or online runs that got
bad feedback), group them, size each group, fix the biggest. Three lenses:

1. SQL over the labelled failures: which metric, channel, tenant and failure mode, and what
   share of traffic. Known slices answer known questions.
2. Embeddings + KMeans over the failing messages: finds groups nobody labelled ("vague
   messages with no asset named", "two requests in one message").
3. Impact = frequency x severity, where severity encodes business cost (a missed safety
   escalation is worth far more than an over-long reply). Ranked list -> backlog.

Same code runs as a Databricks notebook over the traces Delta table (see
databricks/notebooks/05_failure_modes.py); only the loader changes.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

from capstone.rag.embeddings import Embedder, HashingEmbedder

REPORT = Path("docs/eval-reports/failure-modes.md")

# Business cost of one failure on each metric. Argue about these numbers with the product
# owner; the point is to make the trade-off explicit, not to be precise.
SEVERITY = {"safety_escalation": 20.0, "no_sensitive_leak": 15.0, "outcome_correct": 5.0,
            "trajectory_match": 2.0, "urgency_score": 3.0, "intent_correct": 2.0,
            "category_correct": 2.0, "schema_valid": 5.0, "reply_length_ok": 0.5}
FIX_HINTS = {
    "safety_escalation": "Extend the deterministic safety net patterns; add the cases to golden.",
    "no_sensitive_leak": "Tighten PII redaction on output; add a leak test per channel.",
    "outcome_correct": "Check routing rules and the classifier prompt few-shots for this group.",
    "urgency_score": "Add urgency few-shots; consider the calibrated ML second opinion.",
    "intent_correct": "Multi-intent handling: classify the dominant intent, then follow up.",
    "category_correct": "Add asset-to-trade mapping hints (model names) to the prompt.",
    "trajectory_match": "Tool-use guidance in the system prompt; check tool descriptions.",
    "reply_length_ok": "Channel-specific length limits in the responder prompt.",
}


def latest_results(pattern: str = "artifacts/evals/*/results.jsonl", n: int = 3) -> list[str]:
    paths = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return paths[:n]


def load_failures(paths: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (all_runs, failures). One failure row per (example, failing metric)."""
    runs, fails = [], []
    for p in paths:
        for line in Path(p).read_text().splitlines():
            r = json.loads(line)
            base = {"run": Path(p).parent.name, "id": r["id"],
                    "tenant_id": r["inputs"]["tenant_id"],
                    "channel": r["inputs"]["channel"], "message": r["inputs"]["message"],
                    "failure_mode": r["metadata"].get("failure_mode", "none"),
                    "outcome": r["outputs"].get("outcome")}
            runs.append(base)
            for s in r["scores"]:
                if s["score"] is not None and s["score"] < 1:
                    fails.append({**base, "metric": s["key"], "score": float(s["score"]),
                                  "comment": s.get("comment", "")})
    runs_df = pd.DataFrame(runs).drop_duplicates(["run", "id"])
    fails_df = pd.DataFrame(fails)
    if not fails_df.empty:
        fails_df["cost"] = fails_df["metric"].map(SEVERITY).fillna(1.0) * (1 - fails_df["score"])
    return runs_df, fails_df


def sql_slices(runs: pd.DataFrame, fails: pd.DataFrame) -> dict[str, pd.DataFrame]:
    con = sqlite3.connect(":memory:")
    runs.to_sql("runs", con, index=False)
    fails.to_sql("failures", con, index=False)
    q = {
        "by_metric": """
            SELECT metric, COUNT(*) AS n, ROUND(SUM(cost), 1) AS cost,
                   ROUND(100.0 * COUNT(DISTINCT run || id) / (SELECT COUNT(*) FROM runs), 1)
                     AS pct_of_runs
            FROM failures GROUP BY metric ORDER BY cost DESC""",
        "by_failure_mode": """
            SELECT r.failure_mode, COUNT(DISTINCT r.run || r.id) AS runs,
                   COUNT(DISTINCT f.run || f.id) AS failing,
                   ROUND(100.0 * COUNT(DISTINCT f.run || f.id)
                         / COUNT(DISTINCT r.run || r.id), 1) AS fail_pct,
                   ROUND(COALESCE(SUM(f.cost), 0), 1) AS cost
            FROM runs r LEFT JOIN failures f ON f.run = r.run AND f.id = r.id
            GROUP BY r.failure_mode ORDER BY cost DESC""",
        "by_channel_tenant": """
            SELECT channel, tenant_id, COUNT(*) AS failures, ROUND(SUM(cost), 1) AS cost
            FROM failures GROUP BY channel, tenant_id ORDER BY cost DESC LIMIT 10""",
    }
    out = {name: pd.read_sql_query(sql, con) for name, sql in q.items()}
    con.close()
    return out


def cluster(fails: pd.DataFrame, embedder: Embedder | None = None, max_k: int = 8,
            seed: int = 0) -> pd.DataFrame:
    """Cluster unique failing messages; label clusters by their most distinctive terms."""
    msgs = fails.groupby("message").agg(cost=("cost", "sum"), n=("id", "size"),
                                        metrics=("metric", lambda s: s.value_counts().to_dict()),
                                        example_id=("id", "first")).reset_index()
    if len(msgs) < 3:
        msgs["cluster"] = 0
        return msgs
    X = (embedder or HashingEmbedder(dim=512)).embed(msgs["message"].tolist())
    best_k, best_s, best_labels = 1, -1.0, np.zeros(len(msgs), dtype=int)
    for k in range(2, min(max_k, len(msgs) - 1) + 1):
        labels = KMeans(n_clusters=k, n_init=10, random_state=seed).fit_predict(X)
        s = silhouette_score(X, labels)
        if s > best_s:
            best_k, best_s, best_labels = k, s, labels
    msgs["cluster"] = best_labels
    msgs.attrs["k"], msgs.attrs["silhouette"] = best_k, best_s
    return msgs


def describe_clusters(msgs: pd.DataFrame, top_terms: int = 5) -> pd.DataFrame:
    tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
    M = tfidf.fit_transform(msgs["message"])
    vocab = np.array(tfidf.get_feature_names_out())
    overall = np.asarray(M.mean(axis=0)).ravel()
    rows = []
    for c, grp in msgs.groupby("cluster"):
        idx = grp.index.to_numpy()
        lift = np.asarray(M[idx].mean(axis=0)).ravel() - overall  # distinctive, not just common
        metric_counts: dict[str, int] = {}
        for d in grp["metrics"]:
            for k, v in d.items():
                metric_counts[k] = metric_counts.get(k, 0) + v
        top_metric = max(metric_counts, key=lambda k: metric_counts[k])
        rows.append({
            "cluster": int(c), "messages": len(grp), "failures": int(grp["n"].sum()),
            "impact": round(float(grp["cost"].sum()), 1),
            "terms": ", ".join(vocab[np.argsort(-lift)[:top_terms]]),
            "top_metric": top_metric, "fix": FIX_HINTS.get(top_metric, "Investigate."),
            "example": grp.sort_values("cost", ascending=False)["message"].iloc[0][:140],
        })
    return pd.DataFrame(rows).sort_values("impact", ascending=False).reset_index(drop=True)


def _md(df: pd.DataFrame) -> str:
    if df.empty:
        return "_none_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]).replace("|", "/") for c in cols) + " |")
    return "\n".join(lines)


def analyse(paths: list[str], embedder: Embedder | None = None) -> dict[str, Any]:
    runs, fails = load_failures(paths)
    if fails.empty:
        return {"runs": len(runs), "failures": 0, "slices": {}, "clusters": pd.DataFrame()}
    slices = sql_slices(runs, fails)
    msgs = cluster(fails, embedder)
    clusters = describe_clusters(msgs)
    return {"runs": len(runs), "failures": len(fails), "slices": slices, "clusters": clusters,
            "k": msgs.attrs.get("k", 1), "silhouette": msgs.attrs.get("silhouette")}


def render(result: dict[str, Any], paths: list[str]) -> str:
    lines = ["# Failure-mode analysis", "",
             f"Sources: {', '.join(f'`{Path(p).parent.name}`' for p in paths)}", "",
             f"{result['runs']} runs, {result['failures']} metric failures "
             f"(a run can fail several metrics).", ""]
    if not result["failures"]:
        return "\n".join(lines + ["No failures."]) + "\n"
    for name, df in result["slices"].items():
        lines += [f"## {name.replace('_', ' ')}", "", _md(df), ""]
    sil = result.get("silhouette")
    lines += ["## Ranked fix backlog (embedding clusters)", "",
              f"KMeans with k={result['k']} chosen by silhouette"
              + (f" ({sil:.2f})" if sil is not None else "")
              + ". Impact = sum of severity x (1 - score).", "",
              _md(result["clusters"][["cluster", "impact", "failures", "top_metric", "terms",
                                      "fix", "example"]]), ""]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*")
    ap.add_argument("--out", type=Path, default=REPORT)
    args = ap.parse_args(argv)
    paths = args.paths or latest_results()
    result = analyse(paths)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(result, paths))
    print(f"wrote {args.out}")
    return result


if __name__ == "__main__":
    main()
