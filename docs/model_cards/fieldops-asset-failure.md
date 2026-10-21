# Model card: fieldops-asset-failure

## Intended use

Rank assets by 30-day failure risk to prioritise proactive maintenance visits; exposed to agents as the `asset_failure_risk` tool.

## Data

5652 monthly snapshots; time split + 20% of assets held out entirely. Train base rate 0.109.

## Test metrics: logreg

| metric | value |
|---|---|
| pr_auc | 0.3268 |
| roc_auc | 0.7137 |
| brier | 0.0733 |
| precision_at_5pct | 0.4681 |
| base_rate | 0.0913 |
| lift_at_5pct | 5.1272 |
| pr_auc_unseen_assets | 0.5184 |

## Test metrics: hgb_monotone

| metric | value |
|---|---|
| pr_auc | 0.2660 |
| roc_auc | 0.6853 |
| brier | 0.0773 |
| precision_at_5pct | 0.3723 |
| base_rate | 0.0913 |
| lift_at_5pct | 4.0784 |
| pr_auc_unseen_assets | 0.4182 |

## Selected

logreg (higher PR-AUC). The synthetic hazard is logistic in its features, so a linear model matching the generating process is expected to compete; on real telemetry with interactions, boosting usually wins.

## Top features

| metric | value |
|---|---|
| overdue_ratio | 0.1950 |
| fault_count_90d | 0.1103 |
| age_years | 0.0324 |
| service_interval_days | 0.0181 |
| avg_daily_runtime_hours | 0.0144 |
| asset_type | 0.0107 |

## Registry

promoted v1 to champion (pr_auc=0.3268 vs None)

## Limitations

- Synthetic hazard: real telemetry (pressure, temperature, cycles) is far noisier.
- Snapshots of the same asset are correlated; CIs need cluster bootstrap.
- Survival framing (time-to-failure) would use censored data better.
