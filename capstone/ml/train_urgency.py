"""Job urgency classifier from free-text descriptions.

    uv run python -m capstone.ml.train_urgency

Compares TF-IDF + calibrated logistic regression against TF-IDF + Complement Naive Bayes, logs
both to MLflow, and registers the better one. The labels are *dispatcher* labels with about
12% disagreement, so we also score against `true_urgency` to show the label-noise ceiling.

Why have a classical model when the LLM classifies too? It costs about 1/1000th per call, adds
no latency, works as a fallback when the provider is down, and gives a second opinion: big
disagreement with the LLM sends the case to a human.
"""

from __future__ import annotations

from typing import Any

import mlflow
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import FeatureUnion, Pipeline

from capstone.ml.common import log_sklearn, promote_if_better, setup, write_model_card
from capstone.ml.features import job_frame, text_step, time_split

LABELS = ["emergency", "high", "normal", "low"]
MODEL_NAME = "fieldops-urgency"


def tfidf() -> FeatureUnion:
    return FeatureUnion([
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                                 sublinear_tf=True)),
    ])


def candidates() -> dict[str, Pipeline]:
    return {
        "tfidf_logreg_calibrated": Pipeline([
            ("text", text_step()),
            ("tfidf", tfidf()),
            ("clf", CalibratedClassifierCV(
                LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced"),
                method="isotonic", cv=3)),
        ]),
        "tfidf_complement_nb": Pipeline([
            ("text", text_step()),
            ("tfidf", tfidf()),
            ("clf", ComplementNB(alpha=0.3)),
        ]),
    }


def expected_calibration_error(y: Any, proba: np.ndarray, classes: Any, bins: int = 10) -> float:
    """Top-label ECE: how far stated confidence is from actual accuracy, weighted by bin size."""
    conf = proba.max(axis=1)
    correct = np.asarray(classes)[proba.argmax(axis=1)] == np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def evaluate(model: Pipeline, X: Any, y: Any, y_true: Any) -> dict[str, float]:
    pred = model.predict(X)
    proba = model.predict_proba(X)
    return {
        "accuracy": accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro"),
        "emergency_recall": recall_score(y, pred, labels=["emergency"], average="macro"),
        "accuracy_vs_true_label": accuracy_score(y_true, pred),
        "log_loss": log_loss(y, proba, labels=model.classes_),
        "ece": expected_calibration_error(y, proba, model.classes_),
    }


def select(results: dict[str, dict[str, float]]) -> str:
    """Best macro-F1, but among models within one CV std of it, prefer the best calibrated.

    The agent acts on this model's *confidence* (upgrade to emergency above a threshold), so
    a model with slightly higher F1 and unreliable probabilities is the worse choice.
    """
    top = max(results, key=lambda k: results[k]["macro_f1"])
    band = results[top]["macro_f1"] - results[top]["cv_macro_f1_std"]
    close = [k for k, m in results.items() if m["macro_f1"] >= band]
    return min(close, key=lambda k: results[k]["log_loss"])


def main(register: bool = True) -> dict[str, Any]:
    setup("urgency")
    df = job_frame()
    train, test = time_split(df, "created_at")
    Xtr, ytr = train["description"], train["urgency"].astype(str)
    Xte, yte, yte_true = test["description"], test["urgency"].astype(str), \
        test["true_urgency"].astype(str)
    label_agreement = float((test["urgency"].astype(str) == yte_true).mean())

    results: dict[str, dict[str, float]] = {}
    runs: dict[str, str] = {}
    for name, model in candidates().items():
        with mlflow.start_run(run_name=name) as run:
            cv = cross_val_score(model, Xtr, ytr, scoring="f1_macro",
                                 cv=StratifiedKFold(5, shuffle=True, random_state=0))
            model.fit(Xtr, ytr)
            m = evaluate(model, Xte, yte, yte_true) | {"cv_macro_f1_mean": float(cv.mean()),
                                                       "cv_macro_f1_std": float(cv.std())}
            mlflow.log_params({"model": name, "n_train": len(Xtr), "n_test": len(Xte)})
            mlflow.log_metrics(m)
            cm = confusion_matrix(yte, model.predict(Xte), labels=LABELS)
            mlflow.log_text(np.array2string(cm), "confusion_matrix.txt")
            mlflow.log_text(classification_report(yte, model.predict(Xte)), "report.txt")
            log_sklearn(model, input_example=Xte.head(3).to_frame())
            results[name], runs[name] = m, run.info.run_id

    best_name = select(results)
    msg = "not registered"
    if register:
        mv = mlflow.register_model(f"runs:/{runs[best_name]}/model", MODEL_NAME)
        msg = promote_if_better(MODEL_NAME, mv.version, "log_loss",
                                results[best_name]["log_loss"], higher_is_better=False)
    write_model_card(MODEL_NAME, {
        "Intended use": "Second-opinion urgency triage for inbound job descriptions; fallback "
                        "when the LLM is unavailable. Not for life-safety decisions (the "
                        "deterministic safety net and the LLM handle those).",
        "Data": f"{len(df)} synthetic historical jobs; time-based split ({len(Xtr)} train / "
                f"{len(Xte)} test). Labels are dispatcher-assigned; agreement with true "
                f"urgency on test = {label_agreement:.3f} (the practical accuracy ceiling).",
        **{f"Test metrics: {k}": v for k, v in results.items()},
        "Selected": f"{best_name} ({msg}). Rule: best macro-F1, ties within one CV std go "
                    "to the lower log loss, because the agent acts on the confidence.",
        "Limitations": ["Templated synthetic text inflates accuracy vs real messages.",
                        "Retrain/evaluate on paraphrased and production-derived data.",
                        "Emergency recall is the metric to watch; tune thresholds for it.",
                        "Trained on job descriptions, applied to customer messages: domain "
                        "shift. Vague messages ('something is broken') get overconfident "
                        "scores, so the agent may only upgrade urgency, never downgrade."],
    })
    return {"results": results, "selected": best_name, "registry": msg,
            "label_ceiling": label_agreement}


if __name__ == "__main__":
    import json

    print(json.dumps(main(), indent=2, default=str))
