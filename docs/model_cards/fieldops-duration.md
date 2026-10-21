# Model card: fieldops-duration

## Intended use

Estimate on-site hours for a new job to size slots; p10/p90 for schedule buffers.

## Data

Synthetic jobs, time split (1350 train / 450 test). Features known at intake only: category, urgency, asset_type, tenant_id, asset_age_years, engineer_experience_years, hour, dayofweek, month, has_error_code, desc_len

## Test metrics

| metric | value |
|---|---|
| baseline_category_median_mae | 0.7512 |
| mae | 0.6202 |
| rmse | 0.8236 |
| mape | 0.2165 |
| raw_p10_p90_coverage | 0.6667 |
| conformal_offset_h | 0.2583 |
| p10_p90_coverage | 0.8044 |
| mean_interval_width_h | 2.0002 |
| mae_improvement_vs_baseline | 0.1744 |

## Top features

| metric | value |
|---|---|
| category | 0.1857 |
| urgency | 0.1452 |
| engineer_experience_years | 0.0768 |
| asset_age_years | 0.0559 |
| desc_len | 0.0067 |
| asset_type | 0.0014 |

## Registry

promoted v1 to champion (mae=0.6202 vs None)

## Limitations

- Engineer experience is assumed known at booking (assigned engineer).
- No travel time; add route-distance features for real scheduling.
- Quantile crossing is possible; clip p10 <= p50 <= p90 at serve time.
- Serve intervals as [p10 - 0.26, p90 + 0.26] (conformal offset); recalibrate monthly.
