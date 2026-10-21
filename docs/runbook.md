# FieldOps runbook

For whoever is on call. Each section starts with the symptom, then the steps.

## Where the signals live

| Question | Look in | Joined by |
|---|---|---|
| Is the service up, how fast, what errors? | App Insights (OpenTelemetry): requests, dependencies, `fieldops.agent.*` metrics | `operation_Id` = OTel trace id |
| What did the agent see, decide and say? | LangSmith project `fieldops-<env>`: the run tree, prompts, tool calls, tokens, feedback | run metadata `otel_trace_id`, `thread_id`, `tenant_id` |
| What did it cost, was it over budget? | `run_ledger` table in Postgres, `GET /v1/usage` per tenant | `run_id`, `tenant_id` |
| Is a human waiting on it? | `GET /v1/reviews` for the tenant | `thread_id` |
| Did quality drift? | Online scorer feedback in LangSmith, PSI report from `capstone.evals.online` | feedback keys |

Thread ids are `<tenant>:<channel>:<sender>`, so a customer complaint ("I texted you at 9am")
gives you the thread directly.

## 1. "The agent said something wrong" (debug a single bad answer)

1. **Find the run.** From an alert, take the App Insights `operation_Id` and search LangSmith
   metadata `otel_trace_id = <id>`. From a complaint, filter LangSmith by `thread_id`.
2. **Read the run tree top-down** and stop at the first node whose output is wrong:
   - `guard_input`: was PII redacted or an injection flagged? A flag that should have fired and
     didn't is a guardrail bug.
   - `classify`: compare `classification` to what a dispatcher would say. Check `prompt_ref`
     (prompt name and version): is it the version you think is deployed?
   - `route`: the decision is deterministic code. If the classification was right and the route
     was wrong, it's a code bug; write a unit test first.
   - `answer` / tools: did it call the right tool with the tenant's data? Tool output is
     spotlighted; did the model follow instructions found in it?
   - `respond`: did the output policy fire (`policy:*` entries in `flags`)? A template fallback
     here means the LLM reply was blocked.
3. **Reproduce locally** with the same prompt version, and confirm you see the same failure
   before changing anything:
   ```python
   from capstone.evals.target import IntakeTarget
   IntakeTarget(prompt_version="<version from prompt_ref>")(
       {"tenant_id": "t2", "channel": "sms", "message": "<redacted message>"})
   ```
4. **Fix and lock it in** through the feedback loop (section 5), so it can never silently return.

Common root causes, most frequent first: stale prompt tag in an environment, a vague message
outside the few-shot distribution, a KB document missing for one tenant, the ML second opinion
upgrading urgency (look for `ml_urgency_upgrade` in `flags`), and eval fixtures out of date
with the data (`StaleDatasetError`).

## 2. Budgets: latency and cost

Budgets are enforced in the request path by `capstone/observability/ledger.py` and every run
records its violations.

| Channel | p95 latency budget | Why |
|---|---|---|
| voice | 1.5 s | callers hang up; this is per turn |
| web | 6 s | interactive |
| sms / whatsapp | 8 s | the provider times out at 15 s |
| email | 30 s | asynchronous |

Cost cap: **$0.02 per run** (`MAX_COST_PER_RUN_USD`).

**Alert: `fieldops.agent.budget_violations` rising.**

1. `GET /v1/usage` for the affected tenant: is it latency or cost, and which channel?
2. Latency, App Insights: split `fieldops.agent.latency_ms` by channel, then open slow traces.
   Usual suspects are a slow LLM provider (check the provider status page and the per-call
   `latency_ms` in the ledger), a ReAct loop taking extra turns (count tool calls in
   LangSmith), or a cold start (min replicas should be 2 in prod for exactly this reason).
3. Cost: query the ledger for the expensive runs.
   ```sql
   SELECT run_id, channel, model, input_tokens, output_tokens, cost_usd, llm_calls
   FROM run_ledger WHERE tenant_id = 't2' AND created_at > now() - interval '1 day'
   ORDER BY cost_usd DESC LIMIT 20;
   ```
   Large input tokens usually means a long email thread that wasn't stripped, or memory
   growing without trimming. Many LLM calls means a loop.
4. Mitigations, least invasive first: switch the channel to a smaller model via config, cap
   ReAct iterations, tighten message trimming, or temporarily route the tenant to offline mode
   (rule-based classifier and templated replies) while you fix it.

Prices live in `PRICES` and cost is computed when the row is written, so a price change never
rewrites history. Update the table when a provider changes prices.

## 3. Roll back

Pick the smallest rollback that removes the problem.

| What changed | Roll back by | Time |
|---|---|---|
| Prompt | Move the LangSmith tag (`prod`) back to the previous commit. Runtimes pull by tag | seconds, no deploy |
| ML model | `client.set_registered_model_alias(name, "champion", <previous version>)`. Serving loads `@champion` | seconds, no deploy |
| ML tools misbehaving | Set `FIELDOPS_ML_TOOLS=false`. The agent falls back and labels it `source: "fallback"` | one revision |
| Code or image | Shift traffic to the previous revision (below) | about 30 s |

```bash
az containerapp revision list -g fieldops-prod -n fieldops-prod-api \
  --query "[].{name:name, active:properties.active, traffic:properties.trafficWeight}" -o table
az containerapp ingress traffic set -g fieldops-prod -n fieldops-prod-api \
  --revision-weight fieldops-prod-api--<previous-suffix>=100
```

The deploy workflow already does this automatically if the smoke test or canary step fails.
After any rollback, open an issue with the trace links and add the failing case to the
regressions split before re-deploying.

## 4. Readiness failing

`/readyz` returns 503 with the failing check in the body (`graph`, `prompt`, `postgres`).
Container Apps stops routing traffic to that replica but doesn't restart it; `/healthz`
(liveness) stays green unless the process is wedged.

- `postgres: error: OperationalError`: check the Flexible Server status and connection count.
  Each replica holds a pool; scaling to max replicas can exhaust connections.
- `prompt` missing: LangSmith was unreachable at startup and there was no local fallback for
  the requested tag. Pin `FIELDOPS_PROMPT_SOURCE=local` to unblock.

## 5. Feedback loop: from a bad run to a permanent test

```bash
# collect failing runs from the last day (online feedback < 0.5 on any key)
uv run python -m capstone.evals.feedback_loop collect --project fieldops-prod --hours 24
# a human confirms or corrects each proposed label
uv run python -m capstone.evals.feedback_loop review
# fix the prompt or code, confirm the fix on the dev split, then promote
uv run python -m capstone.evals.feedback_loop promote
uv run python -m capstone.evals.runner --split regressions --offline
```

Collected messages are redacted again before they're written, near-duplicates (cosine 0.92)
are dropped, and every row keeps its source run id. The CI gate runs the regressions split
with "fixed stays fixed" thresholds (`job_intake_regressions` in `thresholds.yaml`): outcome
0.95, safety and leaks 1.0, no errors.

## 6. Model retraining

The Databricks job `fieldops_ml_pipeline` runs weekly (`databricks/databricks.yml`). A new
version is registered as `@challenger` and promoted to `@champion` only if it wins on the same
time window. If a retrain is skipped or fails, the champion keeps serving; nothing breaks.
Before turning a new model on for the agent, run `uv run python -m capstone.evals.ml_ablation`
and check that triage metrics are flat or better.

## Deploy

Pipeline: push to `main` runs the eval gate, then builds and pushes the image to ACR, rolls out
to staging and smoke-tests it, then waits for a production approval. In production it creates
a new revision at 0% traffic, smoke-tests that revision directly, sends 10% for five minutes,
then 100%. Any failure shifts traffic back to the old revision.

### One-time setup

1. **Azure resources** (per environment):
   ```bash
   az group create -n fieldops-staging -l uksouth
   az acr create -g fieldops-shared -n <acr> --sku Basic
   az deployment group create -g fieldops-staging -f infra/azure/main.bicep \
     -p env=staging acrName=<acr> image=<acr>.azurecr.io/fieldops:<sha> \
        databaseUrl=<secret> redisHost=<host>:6380 redisPassword=<secret> \
        openaiApiKey=<secret> langsmithApiKey=<secret> apiKeys='<key>:t1,<smoke-key>:smoke'
   ```
   Postgres needs the `vector` extension and `infra/postgres/init.sql` applied.
2. **GitHub OIDC** (no stored cloud credentials): create an Entra app registration, add a
   federated credential for `repo:<owner>/<repo>:environment:staging` and one for
   `environment:production`, and grant it `Contributor` on both resource groups and `AcrPush`
   on the registry.
3. **Repo variables:** `AZURE_CLIENT_ID`, `AZURE_TENANT_ID`, `AZURE_SUBSCRIPTION_ID`, `ACR_NAME`.
   **Repo secrets:** `SMOKE_API_KEY` (a key mapped to a dedicated smoke-test tenant), plus the
   eval-gate secrets (`OPENAI_API_KEY`, `LANGSMITH_API_KEY`).
4. **Environments:** create `staging` and `production` in GitHub; add a required reviewer on
   `production`.

### Verify locally before the first deploy

```bash
docker build -f infra/Dockerfile -t fieldops:dev .
docker run -d -p 8000:8000 -e FIELDOPS_OFFLINE=true fieldops:dev
scripts/smoke_test.sh http://localhost:8000 dev-key-t1
az bicep build --file infra/azure/main.bicep --stdout > /dev/null
```
