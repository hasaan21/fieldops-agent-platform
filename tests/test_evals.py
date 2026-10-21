import pytest

from capstone.evals.calibration import cohen_kappa, weighted_kappa
from capstone.evals.datasets import build, expected_outcome
from capstone.evals.evaluators import heuristic as h
from capstone.evals.gate import check
from capstone.evals.online import drift_report, psi, two_proportion_z
from capstone.evals.runner import run, wilson_interval
from capstone.evals.target import IntakeTarget

REF = {"intent": "book_job", "category": "gas", "urgency": "emergency", "should_escalate": True,
       "needs_clarification": False, "should_book": False, "expected_tools": [],
       "must_not_contain": ["4111"], "expected_outcome": "escalated"}


def test_kappa_matches_sklearn():
    from sklearn.metrics import cohen_kappa_score

    a = [1, 1, 0, 1, 0, 0, 1, 1, 1, 0]
    b = [1, 0, 0, 1, 0, 1, 1, 1, 0, 0]
    assert cohen_kappa(a, b) == pytest.approx(cohen_kappa_score(a, b))
    x = [1, 2, 3, 4, 5, 5, 4, 3]
    y = [1, 2, 4, 4, 5, 4, 4, 2]
    assert weighted_kappa(x, y) == pytest.approx(cohen_kappa_score(x, y, weights="quadratic"))


def test_always_pass_judge_has_zero_kappa():
    human = [True] * 9 + [False]
    judge = [True] * 10
    assert cohen_kappa(human, judge) == pytest.approx(0.0)


def test_under_triaged_emergency_scores_zero():
    out = {"classification": {"urgency": "high"}, "outcome": "booked",
           "trajectory": ["create_job"]}
    assert h.urgency_score(out, REF)["score"] == 0
    assert h.safety_escalation(out, REF)["score"] == 0
    assert h.trajectory_match(out, REF)["score"] == 0  # booked when it must not


def test_leak_detection():
    assert h.no_sensitive_leak({"reply": "card 4111 noted"}, REF)["score"] == 0
    assert h.no_sensitive_leak({"reply": "thanks"}, REF)["score"] == 1


def test_trajectory_allows_interleaving_but_checks_order():
    ref = {"expected_tools": ["lookup_customer", "find_available_slots", "create_job"]}
    ok = {"trajectory": ["lookup_customer", "x", "find_available_slots", "create_job"]}
    bad = {"trajectory": ["lookup_customer", "create_job", "find_available_slots"]}
    assert h.trajectory_match(ok, ref)["score"] == 1
    assert h.trajectory_match(bad, ref)["score"] == 0


def test_expected_outcome_priority():
    assert expected_outcome({**REF}) == "escalated"
    assert expected_outcome({**REF, "should_escalate": False, "should_book": True}) == "booked"


def test_splits_are_disjoint_and_stable():
    a, b = build(), build()
    ids = lambda s: {e["id"] for e in s}  # noqa: E731
    assert ids(a["dev"]) == ids(b["dev"])
    assert not ids(a["dev"]) & ids(a["test"])


def test_wilson_interval_is_wide_for_small_n():
    lo, hi = wilson_interval(8, 10)
    assert lo < 0.5 and hi > 0.9


def test_gate_blocks_safety_regression():
    rules = {"absolute": {"safety_escalation": 1.0},
             "max_regression": {"outcome_correct": 0.03}}
    base = {"metrics": {"outcome_correct": 0.90, "safety_escalation": 1.0}}
    cur = {"metrics": {"outcome_correct": 0.85, "safety_escalation": 0.95}, "error_rate": 0,
           "latency_ms": {"p95": 1}}
    results = check(cur, base, rules)
    assert [r[1] for r in results] == [False, False]


def test_psi_and_z():
    same = {"a": 50, "b": 50}
    assert psi(same, same) == pytest.approx(0.0, abs=1e-9)
    assert psi(same, {"a": 90, "b": 10}) > 0.2
    assert two_proportion_z(0.95, 500, 0.85, 500) < -3


def test_drift_report_alerts_on_outcome_shift():
    base = [{"outcome": "booked", "scores": {"k": 1}}] * 80 + \
           [{"outcome": "clarify", "scores": {"k": 1}}] * 20
    cur = [{"outcome": "booked", "scores": {"k": 0}}] * 30 + \
          [{"outcome": "clarify", "scores": {"k": 0}}] * 70
    rep = drift_report(base, cur)
    assert any("outcome" in a for a in rep["alerts"])
    assert any("k pass rate" in a for a in rep["alerts"])


def test_offline_eval_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("capstone.evals.runner.ARTIFACTS", tmp_path)
    summary, out_dir = run("golden", IntakeTarget(offline=True))
    assert summary["n"] == 12
    assert summary["metrics"]["safety_escalation"] == 1.0
    assert summary["metrics"]["no_sensitive_leak"] == 1.0
    assert (out_dir / "report.md").exists()
