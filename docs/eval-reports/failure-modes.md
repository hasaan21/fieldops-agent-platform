# Failure-mode analysis

Sources: `20261020T072122-dev-RuleBasedClassifier`, `20261020T072100-test-RuleBasedClassifier`, `20261020T072102-adversarial-RuleBasedClassifier`

426 runs, 33 metric failures (a run can fail several metrics).

## by metric

| metric | n | cost | pct_of_runs |
|---|---|---|---|
| urgency_score | 32 | 51.0 | 7.5 |
| category_correct | 1 | 2.0 | 0.2 |

## by failure mode

| failure_mode | runs | failing | fail_pct | cost |
|---|---|---|---|---|
| none | 174 | 12 | 6.9 | 18.0 |
| ambiguous_asset | 26 | 8 | 30.8 | 12.0 |
| multi_intent | 23 | 3 | 13.0 | 9.5 |
| missing_address | 34 | 4 | 11.8 | 6.0 |
| angry_customer | 41 | 4 | 9.8 | 6.0 |
| prompt_injection | 52 | 1 | 1.9 | 1.5 |
| reschedule_request | 24 | 0 | 0.0 | 0.0 |
| out_of_scope | 17 | 0 | 0.0 | 0.0 |
| contains_pii | 35 | 0 | 0.0 | 0.0 |

## by channel tenant

| channel | tenant_id | failures | cost |
|---|---|---|---|
| email | t2 | 5 | 9.5 |
| sms | t2 | 6 | 9.0 |
| sms | t3 | 6 | 9.0 |
| email | t3 | 4 | 6.0 |
| email | t1 | 3 | 4.5 |
| whatsapp | t1 | 3 | 4.5 |
| whatsapp | t2 | 3 | 4.5 |
| sms | t1 | 1 | 3.0 |
| voice | t2 | 1 | 1.5 |
| voice | t3 | 1 | 1.5 |

## Ranked fix backlog (embedding clusters)

KMeans with k=3 chosen by silhouette (0.31). Impact = sum of severity x (1 - score).

| cluster | impact | failures | top_metric | terms | fix | example |
|---|---|---|---|---|---|---|
| 0 | 31.5 | 21 | urgency_score | sentinel fp, fp, sentinel, fault, light code | Add urgency few-shots; consider the calibrated ML second opinion. | The Sentinel FP-4 is showing a fault light, code Z1. |
| 1 | 12.0 | 8 | urgency_score | stopped working, help, stopped, working help, working | Add urgency few-shots; consider the calibrated ML second opinion. | It's stopped working. Help. |
| 2 | 9.5 | 4 | urgency_score | thermacore x200, thermacore, x200, db, 12 | Add urgency few-shots; consider the calibrated ML second opinion. | We need the EICR inspection done on our Voltix DB-12 this month. Also, the voltix db-12 is in alarm and we can't silence it, no fire visible |
