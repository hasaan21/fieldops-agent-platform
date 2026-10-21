"""Calibrate an LLM judge against human labels.

Workflow:
  1. `export`: sample N runs into a CSV with an empty `human_score` column per criterion.
  2. You label them (blind to the judge's score).
  3. `report`: compute agreement, Cohen's kappa (binary pass/fail) and quadratic-weighted kappa
     (1-5 ordinal), plus the confusion matrix. Iterate on the rubric until kappa >= 0.6.

Kappa corrects raw agreement for chance: kappa = (p_o - p_e) / (1 - p_e).
If 90% of replies pass, a judge that always says "pass" gets 90% agreement but kappa = 0.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from collections.abc import Sequence
from pathlib import Path


def cohen_kappa(a: Sequence[object], b: Sequence[object]) -> float:
    if len(a) != len(b) or not a:
        raise ValueError("need two equal-length, non-empty label lists")
    n = len(a)
    p_o = sum(x == y for x, y in zip(a, b, strict=True)) / n
    ca, cb = Counter(a), Counter(b)
    p_e = sum(ca[k] * cb.get(k, 0) for k in ca) / (n * n)
    return 1.0 if p_e == 1 else (p_o - p_e) / (1 - p_e)


def weighted_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int] = (1, 2, 3, 4, 5)
                   ) -> float:
    """Quadratic-weighted kappa: disagreeing by 1 point costs far less than by 4."""
    k = len(labels)
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(a)
    obs = [[0.0] * k for _ in range(k)]
    for x, y in zip(a, b, strict=True):
        obs[idx[x]][idx[y]] += 1
    ra = [sum(r) for r in obs]
    rb = [sum(obs[i][j] for i in range(k)) for j in range(k)]
    num = den = 0.0
    for i in range(k):
        for j in range(k):
            w = ((i - j) ** 2) / ((k - 1) ** 2)
            num += w * obs[i][j]
            den += w * ra[i] * rb[j] / n
    return 1.0 - num / den if den else 1.0


def confusion(a: Sequence[object], b: Sequence[object]) -> dict[tuple[object, object], int]:
    return dict(Counter(zip(a, b, strict=True)))


def report(path: Path, criterion: str, pass_at: int = 4) -> dict[str, float]:
    rows = list(csv.DictReader(path.open()))
    rows = [r for r in rows if r.get(f"human_{criterion}") and r.get(f"judge_{criterion}")]
    human = [int(r[f"human_{criterion}"]) for r in rows]
    judge = [int(r[f"judge_{criterion}"]) for r in rows]
    hb = [h >= pass_at for h in human]
    jb = [j >= pass_at for j in judge]
    out = {
        "n": float(len(rows)),
        "exact_agreement": sum(h == j for h, j in zip(human, judge, strict=True)) / len(rows),
        "pass_fail_agreement": sum(h == j for h, j in zip(hb, jb, strict=True)) / len(rows),
        "cohen_kappa_pass_fail": cohen_kappa(hb, jb),
        "weighted_kappa_1to5": weighted_kappa(human, judge),
        "judge_more_lenient_rate": sum(j > h for h, j in zip(human, judge, strict=True))
        / len(rows),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path)
    ap.add_argument("--criterion", required=True)
    args = ap.parse_args()
    for k, v in report(args.csv, args.criterion).items():
        print(f"{k:>26}: {v:.3f}")


if __name__ == "__main__":
    main()
