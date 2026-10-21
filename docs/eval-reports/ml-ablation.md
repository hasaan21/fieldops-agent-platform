# ML tools ablation: does the agent get smarter?

Registry versions used: `{'fieldops-duration': '1', 'fieldops-asset-failure': '1', 'fieldops-urgency': '2'}`

## 1. Agent eval (dev split, offline classifier)

| metric | without ML | with ML | delta |
|---|---|---|---|
| category_correct | 0.994 | 0.994 | +0.000 |
| intent_correct | 1.000 | 1.000 | +0.000 |
| no_sensitive_leak | 1.000 | 1.000 | +0.000 |
| outcome_correct | 1.000 | 1.000 | +0.000 |
| reply_length_ok | 1.000 | 1.000 | +0.000 |
| safety_escalation | 1.000 | 1.000 | +0.000 |
| schema_valid | 1.000 | 1.000 | +0.000 |
| trajectory_match | 1.000 | 1.000 | +0.000 |
| urgency_score | 0.951 | 0.951 | +0.000 |

| what the tools added | without ML | with ML |
|---|---|---|
| bookings | 118 | 118 |
| with_duration_estimate | 0 | 118 |
| with_asset_risk | 0 | 47 |
| maintenance_suggested | 0 | 3 |
| ml_urgency_upgrades | 0 | 0 |
| ml_urgency_disagreements | 0 | 2 |
| latency_p50_ms | 7 | 23 |

## 2. Slot sizing on historical test jobs

| policy | overrun rate | mean booked h | mean idle h |
|---|---|---|---|
| fixed_2h | 81.6% | 2.00 | 0.06 |
| category_median | 51.3% | 2.75 | 0.28 |
| category_p90 | 12.4% | 4.13 | 1.28 |
| model_p50 | 50.0% | 2.84 | 0.26 |
| model_conformal_p90 | 12.4% | 3.92 | 1.07 |

## Reading this

- Triage metrics should be flat: the ML second opinion only upgrades to emergency at high confidence. A drop here would block the change.
- The win is in slot sizing: compare policies at similar booked hours. A model that cuts overruns without booking more hours frees real capacity.
- Synthetic data caveat: durations come from a known generator, so the model has an easier job than on real engineer timesheets.
