# Model card: fieldops-urgency

## Intended use

Second-opinion urgency triage for inbound job descriptions; fallback when the LLM is unavailable. Not for life-safety decisions (the deterministic safety net and the LLM handle those).

## Data

1800 synthetic historical jobs; time-based split (1350 train / 450 test). Labels are dispatcher-assigned; agreement with true urgency on test = 0.893 (the practical accuracy ceiling).

## Test metrics: tfidf_logreg_calibrated

| metric | value |
|---|---|
| accuracy | 0.8911 |
| macro_f1 | 0.8927 |
| emergency_recall | 0.9322 |
| accuracy_vs_true_label | 0.9978 |
| log_loss | 0.4133 |
| ece | 0.0146 |
| cv_macro_f1_mean | 0.8976 |
| cv_macro_f1_std | 0.0175 |

## Test metrics: tfidf_complement_nb

| metric | value |
|---|---|
| accuracy | 0.8933 |
| macro_f1 | 0.8950 |
| emergency_recall | 0.9322 |
| accuracy_vs_true_label | 1.0000 |
| log_loss | 1.8014 |
| ece | 0.1067 |
| cv_macro_f1_mean | 0.9014 |
| cv_macro_f1_std | 0.0184 |

## Selected

tfidf_logreg_calibrated (promoted v2 to champion (log_loss=0.4133 vs None)). Rule: best macro-F1, ties within one CV std go to the lower log loss, because the agent acts on the confidence.

## Limitations

- Templated synthetic text inflates accuracy vs real messages.
- Retrain/evaluate on paraphrased and production-derived data.
- Emergency recall is the metric to watch; tune thresholds for it.
- Trained on job descriptions, applied to customer messages: domain shift. Vague messages ('something is broken') get overconfident scores, so the agent may only upgrade urgency, never downgrade.
