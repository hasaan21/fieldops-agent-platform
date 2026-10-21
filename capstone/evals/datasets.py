"""Eval datasets for the Job Intake agent.

    uv run python -m capstone.evals.datasets                 # write local JSONL splits
    uv run python -m capstone.evals.datasets --push          # also create LangSmith datasets

Splits are assigned by a stable hash of the example id, stratified by failure mode, so the same
example always lands in the same split. Dev is for iterating on prompts; test is touched only to
report results (otherwise you overfit the prompt to it).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from capstone.data.loader import load_table

DATA_DIR = Path(__file__).parent / "data"

OUTCOME_FOR_LABELS = [
    (lambda lab: lab["should_escalate"], "escalated"),
    (lambda lab: lab["intent"] == "out_of_scope", "declined"),
    (lambda lab: lab["intent"] == "question", "answered"),
    (lambda lab: lab["intent"] == "reschedule", "reschedule_handoff"),
    (lambda lab: lab["needs_clarification"], "clarify"),
    (lambda lab: lab["should_book"], "booked"),
]


def expected_outcome(labels: dict[str, Any]) -> str:
    return next((o for pred, o in OUTCOME_FOR_LABELS if pred(labels)), "clarify")


def _split(example_id: str, test_fraction: float = 0.3) -> str:
    h = int(hashlib.sha256(example_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "test" if h < test_fraction else "dev"


def to_example(tx: dict[str, Any], source: str = "synthetic") -> dict[str, Any]:
    labels = dict(tx["labels"])
    labels["expected_outcome"] = labels.get("expected_outcome") or expected_outcome(labels)
    return {
        "id": tx["transcript_id"],
        "inputs": {"tenant_id": tx["tenant_id"], "channel": tx["channel"], "sender": tx["from"],
                   "message": tx["message"]},
        "reference": labels,
        "metadata": {"source": source, "failure_mode": labels.get("failure_mode") or "none",
                     "split": _split(tx["transcript_id"])},
    }


def build() -> dict[str, list[dict[str, Any]]]:
    examples = [to_example(t) for t in load_table("transcripts")]
    golden_path = DATA_DIR / "golden.jsonl"
    golden = [json.loads(line) for line in golden_path.read_text().splitlines() if line.strip()]
    for g in golden:
        g["reference"]["expected_outcome"] = g["reference"].get("expected_outcome") or \
            expected_outcome(g["reference"])
        g.setdefault("metadata", {}).update({"source": "golden", "split": "test"})
    splits: dict[str, list[dict[str, Any]]] = {"dev": [], "test": [], "golden": golden}
    for ex in examples:
        splits[ex["metadata"]["split"]].append(ex)
    splits["adversarial"] = [e for e in examples + golden
                             if e["metadata"]["failure_mode"] in
                             {"prompt_injection", "contains_pii", "angry_customer"}]
    return splits


class StaleDatasetError(RuntimeError):
    pass


def world_fingerprint() -> str:
    """Hash of the inputs the frozen splits depend on (senders, messages, labels)."""
    h = hashlib.sha256()
    for t in sorted(load_table("transcripts"), key=lambda t: t["transcript_id"]):
        h.update(json.dumps([t["transcript_id"], t["from"], t["message"], t["labels"]],
                            sort_keys=True).encode())
    return h.hexdigest()[:16]


def load_split(name: str) -> list[dict[str, Any]]:
    path = DATA_DIR / f"{name}.jsonl"
    if not path.exists() or name == "golden":
        return build()[name]
    meta_path = DATA_DIR / "_meta.json"
    if meta_path.exists():
        expected = json.loads(meta_path.read_text()).get("world_fingerprint")
        if expected and expected != world_fingerprint():
            # Frozen splits reference customers/phones from the world they were built on.
            # Silently evaluating against a different world shows up as a fake regression.
            raise StaleDatasetError(
                f"{path.name} was built from a different synthetic world; run "
                "`uv run python -m capstone.evals.datasets` and re-baseline")
    return [json.loads(line) for line in path.read_text().splitlines()]


def push_to_langsmith(splits: dict[str, list[dict[str, Any]]], prefix: str = "fieldops-intake"
                      ) -> None:
    from langsmith import Client

    client = Client()
    for name, rows in splits.items():
        ds_name = f"{prefix}-{name}"
        if client.has_dataset(dataset_name=ds_name):
            client.delete_dataset(dataset_name=ds_name)
        ds = client.create_dataset(ds_name, description=f"Job intake eval set: {name}")
        client.create_examples(
            dataset_id=ds.id,
            inputs=[r["inputs"] for r in rows],
            outputs=[r["reference"] for r in rows],
            metadata=[{**r["metadata"], "example_id": r["id"]} for r in rows],
        )
        print(f"pushed {len(rows)} examples to {ds_name}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    args = ap.parse_args()
    splits = build()
    for name, rows in splits.items():
        if name == "golden":
            continue
        (DATA_DIR / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (DATA_DIR / "_meta.json").write_text(json.dumps(
        {"world_fingerprint": world_fingerprint(),
         "counts": {k: len(v) for k, v in splits.items()}}, indent=2) + "\n")
    for name, rows in splits.items():
        print(f"{name:>12}: {len(rows)}")
    if args.push:
        push_to_langsmith(splits)


if __name__ == "__main__":
    main()
