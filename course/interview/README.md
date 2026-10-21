# Interview track

Runs alongside Week 6 (about 8 of that week's hours) and in the days before each interview round.

| File | What | When |
|---|---|---|
| [`star-stories.md`](star-stories.md) | The 8 behavioural stories to prepare, a template and a worked example | Draft in week 5, rehearse in week 6 |
| [`system-design.md`](system-design.md) | 3 design drills: eval platform, RAG for 10k tenants, drift detection | One per day in week 6 |
| [`take-home.md`](take-home.md) | A 4-hour timed take-home (Quote Assistant) with a self-review rubric | End of week 6 |
| [`prompt-debugging-drill.md`](prompt-debugging-drill.md) | A 30-minute live drill on a broken prompt, with an answer key | Week 6, then again before the technical round |

Keep personal drafts (real stories, company notes) in `private/`, which is git-ignored.

## Common requirements to evidence

What AI/ML engineer postings for agent platforms typically ask for, and where this course gives
you something to *show* rather than claim. In an interview, answer with the evidence column:
"I did X, here's the number, here's the code." Before each application, copy this table into
`private/`, map it to that posting's wording, and add a column for your own work experience.

| Requirement | Capstone evidence | Where |
|---|---|---|
| Build agents on LangGraph: tools, knowledge bases, human review, draft to production | Job Intake agent: explicit graph, typed context, checkpointer, interrupt before side effects | `capstone/agents/job_intake/`, Week 1 |
| Prompt engineering with a registry; few-shot, structured output, context management | Versioned prompts with tags and promotion via the gate; v1 vs v2 eval report | `capstone/prompts/`, Weeks 1-2 |
| **Evaluations: datasets, heuristic and judge rubrics, offline and online, thresholds, drift** | 10 heuristic evaluators, calibrated judge (kappa), Wilson CIs, gate, PSI drift, feedback loop, regressions split | `capstone/evals/`, Weeks 2 and 6 |
| RAG: chunking, embeddings, hybrid search, faithfulness | Chunking ablation (recall@5 0.66 to 1.00), BM25 + vector + RRF + rerank, claim-level faithfulness | `capstone/rag/`, Week 3 |
| Tools: HTTP, MCP, recipes | Tool registry, SSRF-safe HTTP tools, recipe sub-graphs, tenant-scoped MCP server | `capstone/tools/`, Week 4 |
| Data analysis of traces and transcripts | Failure-mode clustering ranked by impact; the out-of-scope root cause | `capstone/analysis/`, Weeks 5-6 |
| Applied ML: classification, forecasting, predictive maintenance | Urgency (calibrated), duration (conformal intervals), failure risk (PR-AUC, lift); agent ablation | `capstone/ml/`, Week 5 |
| MLOps on Databricks or SageMaker | Asset bundle, UC feature table with point-in-time lookups, MLflow aliases, serving | `databricks/`, Week 5 |
| Observability (LangSmith, an APM such as App Insights); close the loop | OTel + LangSmith joined by trace id, ledger with budgets, runbook, feedback loop | `capstone/observability/`, `docs/runbook.md` |
| Strong Python, production services, tested code | FastAPI service, typed (mypy clean), tested, CI | whole repo |
| LLM APIs, tool calling, ReAct loops | ReAct answerer with tenant-bound tools | Week 1 |
| ML foundation: framing, features, validation, overfitting | Time and grouped splits, leakage rules, calibration, model selection under noise | Week 5 |
| Pandas, NumPy, SQL | Failure-mode SQL slices, feature engineering, ledger queries | Weeks 5-6 |
| APIs, JSON schemas, third-party services | Twilio, Nylas, ElevenLabs adapters, JSON-schema tools | Week 4 |
| SQL and NoSQL | Postgres (pgvector, RLS, checkpoints), Redis (idempotency, queues) | Weeks 0, 4 |
| Nice-to-haves: PromptFoo, vector DBs, Celery, cloud deploy, Docker, CI/CD, telephony/email, guardrails, feature stores | All in the capstone | Weeks 2-6 |

## Gaps to be honest about

Say them before they're found, with what you did about them. Common ones:
- **No production LangGraph (or agent framework) at work.** "I built a production-shaped agent
  with checkpointing, interrupts and evals, and here's how it relates to what I did at work."
- **Applied ML is recent.** "I rebuilt it on three real framings, with honest validation and an
  agent-level ablation." Lead with the Naive Bayes story.
- **The ML platform was learned on a free tier**, not at production scale. Know what changes at
  scale: cluster policies, catalog permissions, job compute costs, serving autoscaling.

## The week before an interview

- Re-read the company's engineering blog and product pages; map one of your stories to one of
  their features.
- Rehearse all eight STAR stories out loud; time them.
- Do one system-design drill against the clock.
- Re-run the capstone from a clean clone (`uv sync`, tests, eval, docker) so a live demo works.
- Prepare three questions for them, for example: how they evaluate customer-customised agents,
  how prompt changes reach production today, and what the first 90 days would look like.
