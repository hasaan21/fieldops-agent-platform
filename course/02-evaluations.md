# Week 2: Evaluations (the core skill)

**Time:** about 30 hours. Take extra from Week 6 if you need it.
**Outcome:** an eval suite that can say with numbers "prompt v2 is better than v1, and it is safe to ship", and a CI gate that enforces it.

## Why it matters

Evaluation is what separates an agent demo from an agent product, and it's the skill AI engineering roles test hardest: datasets, heuristic and LLM-judge rubrics, offline evaluations and online scoring of live runs, pass/fail thresholds, and regression and drift tracking over time. Knowing an eval *tool* (LangSmith, PromptFoo) isn't enough. You need to *design* evals: pick metrics that match business risk, know when a judge can be trusted, and read results statistically.

## Concepts

### 1. What to measure: a pyramid
1. **Deterministic checks** (fast, free, run on every commit): schema validity, routing/outcome correctness, trajectory (did it call the right tools in the right order, and *not* call `create_job` when it shouldn't), leakage, length limits.
2. **Reference-based quality**: classification accuracy and ordinal urgency scoring against gold labels.
3. **LLM-as-judge** (slower, costs money, must be calibrated): tone, groundedness, outcome fidelity, safety of the wording.
4. **Human review**: a small sample weekly, used to calibrate the judge and to find new failure modes.

### 2. Metrics follow risk
- A missed gas leak is not "1 of 255 errors"; it is a release blocker. Hence `safety_escalation` and `no_sensitive_leak` have an absolute threshold of 1.0 in `thresholds.yaml`.
- `urgency_score` gives partial credit off-by-one, but **0 for under-triaging an emergency**. Asymmetric costs belong in the metric.
- Slice everything by failure mode. Overall accuracy hides that `out_of_scope` handling might be 46%.

### 3. Datasets
- Sources: synthetic (with failure modes), handwritten golden cases (`evals/data/golden.jsonl`), and reviewed production traces (Week 6).
- **Splits**: `dev` for iterating, `test` only for reporting, assigned by a stable hash. If you tune prompts on `test`, its numbers mean nothing.
- **Warning, seen in this repo:** the rule-based baseline scores about 97% on synthetic dev because the rules and the templates share vocabulary. Synthetic data is too easy. Your LLM must be judged mainly on golden + paraphrased + production-derived sets. Write this up; interviewers love candidates who distrust their own numbers.

### 4. LLM-as-judge done properly
- One criterion per call, with anchored 1-5 descriptions (`rubrics/job_intake_reply.yaml`); reasoning before score; a stronger, different model than the agent.
- **Calibrate**: label 50-100 replies yourself, blind. Compute Cohen's kappa for pass/fail and quadratic-weighted kappa for 1-5 (`calibration.py`). Aim for kappa >= 0.6 before you use the judge as a gate; below that, use it for exploration only.
- Known biases: position (fix by swapping A/B, see `PairwiseJudge`), verbosity (longer looks better), self-preference (the same model family rates its own outputs higher).
- Reading: [Zheng et al. "Judging LLM-as-a-Judge"](https://arxiv.org/abs/2306.05685), [Hamel Husain: "Your AI product needs evals"](https://hamel.dev/blog/posts/evals/), [Eugene Yan: LLM evaluators](https://eugeneyan.com/writing/llm-evaluators/).

### 5. Statistics you must be able to say out loud
- With n=100 and 80% accuracy, the 95% Wilson interval is roughly 71-87%. A 2-point "improvement" is noise. The report prints the interval.
- To compare two prompts on the *same* examples, use a paired test (McNemar on per-example pass/fail), or bootstrap the per-example differences.
- Thresholds must sit above the noise floor, which is why the regression tolerance is 0.03, not 0.

### 6. Offline vs online
- **Offline**: datasets plus a target function, in LangSmith experiments or `runner.py`. Repeatable and comparable.
- **Online**: score sampled live runs with *reference-free* evaluators and the judge, and write the results back as LangSmith feedback (`online.py score`).
- **Drift**: PSI on outcome/intent/channel distributions, plus a two-proportion z-test on feedback pass rates (`online.py drift`). Typical causes: new seasonality (boiler breakdowns in winter), a provider model update, a prompt tag moved, or a new tenant with a different customer base.

### 7. Tools
- LangSmith: [evaluation concepts](https://docs.smith.langchain.com/evaluation/concepts), `evaluate()`, experiments comparison view, pairwise experiments, online evaluators/rules, annotation queues.
- PromptFoo: [assertions](https://www.promptfoo.dev/docs/configuration/expected-outputs/), [Python providers](https://www.promptfoo.dev/docs/providers/python/), [red teaming](https://www.promptfoo.dev/docs/red-team/).

## Code tour
| File | Purpose |
|---|---|
| `capstone/evals/datasets.py` | Splits, golden merge, LangSmith push |
| `capstone/evals/evaluators/heuristic.py` | 10 deterministic evaluators in LangSmith feedback shape |
| `capstone/evals/evaluators/judge.py` + `rubrics/` | Rubric judge, pairwise judge with position swap |
| `capstone/evals/calibration.py` | Cohen's / weighted kappa, agreement report |
| `capstone/evals/runner.py` | Parallel offline runs, slices, Wilson CI, markdown report |
| `capstone/evals/gate.py` + `thresholds.yaml` | Absolute + regression rules; exit 1 blocks CI |
| `capstone/evals/online.py` | Online feedback scoring, PSI/z-test drift |
| `capstone/evals/promptfoo/` | Behavioural suite and red-team config |
| `.github/workflows/eval-gate.yml` | Tests, then offline gate, then LLM gate, PR comment, PromptFoo |

## Lab

1. Baseline: `uv run python -m capstone.evals.runner --split dev --offline`. Read `report.md`.
2. LLM runs: evaluate classify prompt `v1` and `v2` on `dev`, then `golden`:
   ```bash
   uv run python -m capstone.evals.runner --split dev --prompt v1
   uv run python -m capstone.evals.runner --split dev --prompt v2
   ```
   Then push datasets (`python -m capstone.evals.datasets --push`) and repeat with `--langsmith` to see the experiment comparison view.
3. **Judge calibration (required):** run with `--judge` on 60 examples. Export `reply`, `outcome` and the judge scores to CSV, add your own `human_<criterion>` columns *without looking at the judge column*, then run `python -m capstone.evals.calibration file.csv --criterion groundedness`. Iterate on the rubric anchors until kappa >= 0.6, and record each iteration.
4. **Write a new evaluator (required):** `clarification_quality`. When the outcome is `clarify`, does the reply ask for exactly the missing information (address vs asset) and nothing else? Implement it as a heuristic first, then as a judge criterion, and compare the two.
5. **Paired comparison (required):** add `capstone/evals/compare.py` that loads two `results.jsonl` files, aligns them by example id and runs McNemar's test on `outcome_correct`. Report: "v2 fixes 14 examples and breaks 3 (p=0.006)".
6. **PromptFoo:** `cd capstone/evals/promptfoo && npx promptfoo@latest eval`. Add 5 tests for failure modes you found in step 2. Then run the red team and triage the findings into: fix in prompt, fix in code (guardrail), or accept with a reason.
7. **Online loop:** run the API with tracing for 30 messages (use transcripts as traffic), run `online.py score --judge`, and build a LangSmith dashboard showing pass rate per tenant.
8. **Write the report:** copy `docs/eval-reports/TEMPLATE.md` to `docs/eval-reports/001-classify-v1-vs-v2.md` and fill it in with real numbers, CIs, the paired test, slices, cost per 1k messages and your ship/no-ship recommendation.

### Acceptance criteria
- [ ] `tests/test_evals.py` passes; your new evaluator has tests.
- [ ] A calibrated judge (kappa recorded) for at least two criteria.
- [ ] The eval report has CIs and a paired significance test.
- [ ] A CI run on a PR shows the gate comment. Prove it fails by deliberately breaking the safety net.

## Production bar
- Evaluators are pure functions with tests; the judge prompt and rubric are versioned like any prompt.
- Gates are on safety-critical metrics *absolutely* and on quality metrics *relative to baseline*.
- Every eval run records config (prompt ref, model, dataset version), cost and latency.
- The baseline only updates on main, after a passing gate (`--update-baseline`).

## Interview questions
1. *"How would you evaluate a new booking agent before launch?"* Pyramid; golden set from domain experts; failure-mode taxonomy; metric per risk; thresholds; shadow mode against human agents; canary with online scoring.
2. *"How do you know your LLM judge is right?"* Human labels, kappa, bias mitigations, a judge model different from the agent, periodic re-calibration, and tracking judge drift too.
3. *"Your eval improved 2 points. Ship?"* Check the CI and a paired test; look at slices for regressions; check cost/latency; check the safety metrics are unchanged; then canary.
4. *"How do you detect quality drift in production with no labels?"* Reference-free evaluators, a judge on a sample, distribution drift (PSI) on inputs and outputs, proxy business signals (reviewer rejection rate, customer re-contact within 24h), alerts, and annotation queues.
5. *"PromptFoo vs LangSmith?"* PromptFoo: declarative, CI-friendly, assertions and red teaming, provider-agnostic. LangSmith: tracing-native datasets, experiments UI, online evaluators, annotation queues, prompt registry. Use both: PromptFoo for behavioural/red-team gates, LangSmith for experiment tracking and production loops.
