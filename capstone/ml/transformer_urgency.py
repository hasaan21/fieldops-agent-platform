"""Fine-tune a small transformer for urgency and compare it with the TF-IDF champion.

    uv sync --extra deep --extra ml
    uv run python -m capstone.ml.transformer_urgency --epochs 3          # CPU: ~10-15 min
    uv run python -m capstone.ml.transformer_urgency --model distilroberta-base

Plain PyTorch training loop on purpose (no Trainer), so every step is visible: tokenise,
batch, forward, loss, backward, clip, step, schedule, evaluate. Uses the same time split and
metrics as train_urgency.py, so the numbers are directly comparable.

What to expect on the synthetic data: about the same accuracy as TF-IDF (both hit the
label-noise ceiling), at around 100x the inference cost. That *is* the lesson: reach for the
transformer when text is varied (real customer messages, paraphrases, typos), and prove it
with the eval, not by default. Rerun the comparison on the paraphrased golden set.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import mlflow
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, recall_score

from capstone.ml.common import setup
from capstone.ml.features import job_frame, time_split
from capstone.ml.train_urgency import LABELS, expected_calibration_error


def main(model_name: str = "distilbert-base-uncased", epochs: int = 3, batch_size: int = 32,
         lr: float = 5e-5, max_len: int = 96, seed: int = 0) -> dict[str, Any]:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        get_linear_schedule_with_warmup,
    )

    torch.manual_seed(seed)
    device = "cuda" if torch.cuda.is_available() else \
        "mps" if torch.backends.mps.is_available() else "cpu"
    setup("urgency")
    df = job_frame()
    train, test = time_split(df, "created_at")
    label_id = {lab: i for i, lab in enumerate(LABELS)}

    tok = AutoTokenizer.from_pretrained(model_name)

    def encode(texts: Any, labels: Any) -> TensorDataset:
        enc = tok(list(texts), truncation=True, padding="max_length", max_length=max_len,
                  return_tensors="pt")
        y = torch.tensor([label_id[str(v)] for v in labels])
        return TensorDataset(enc["input_ids"], enc["attention_mask"], y)

    train_dl = DataLoader(encode(train["description"], train["urgency"]),
                          batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(encode(test["description"], test["urgency"]), batch_size=128)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
        label2id=label_id).to(device)
    # class weights: emergencies are rare and under-triage is the costly error
    counts = train["urgency"].astype(str).value_counts()
    weights = torch.tensor([len(train) / (len(LABELS) * counts.get(lab, 1)) for lab in LABELS],
                           dtype=torch.float, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    steps = epochs * len(train_dl)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    with mlflow.start_run(run_name=f"transformer_{model_name.split('/')[-1]}"):
        mlflow.log_params({"model": model_name, "epochs": epochs, "lr": lr,
                           "batch_size": batch_size, "max_len": max_len, "device": device})
        for epoch in range(epochs):
            model.train()
            total = 0.0
            for ids, mask, y in train_dl:
                ids, mask, y = ids.to(device), mask.to(device), y.to(device)
                opt.zero_grad()
                loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sched.step()
                total += loss.item()
            mlflow.log_metric("train_loss", total / len(train_dl), step=epoch)

        model.eval()
        probs: list[np.ndarray] = []
        t0 = time.perf_counter()
        with torch.no_grad():
            for ids, mask, _ in test_dl:
                logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits
                probs.append(torch.softmax(logits, -1).cpu().numpy())
        latency_ms = (time.perf_counter() - t0) * 1000 / len(test)
        proba = np.concatenate(probs)
        pred = np.array(LABELS)[proba.argmax(1)]
        y, y_true = test["urgency"].astype(str), test["true_urgency"].astype(str)
        metrics = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "emergency_recall": recall_score(y, pred, labels=["emergency"], average="macro"),
            "accuracy_vs_true_label": accuracy_score(y_true, pred),
            "ece": expected_calibration_error(y, proba, LABELS),
            "latency_ms_per_example": latency_ms,
        }
        mlflow.log_metrics(metrics)
        from transformers import pipeline

        clf = pipeline("text-classification", model=model.to("cpu"), tokenizer=tok)
        mlflow.transformers.log_model(clf, name="model",
                                      input_example=test["description"].head(2).tolist())
    return metrics


if __name__ == "__main__":
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="distilbert-base-uncased")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--lr", type=float, default=5e-5)
    a = ap.parse_args()
    print(json.dumps(main(a.model, a.epochs, lr=a.lr), indent=2))
