# Course: production AI agents, evals and applied ML

Six intensive weeks (about 25 hours each) plus a setup weekend. Every week adds one part of the
FieldOps capstone, so you finish with a working, deployed, evaluated platform rather than a folder
of exercises. It's aimed at engineers moving into AI/ML engineering on agent platforms: LangGraph,
LangSmith, PromptFoo, RAG, tools and MCP, applied ML and MLOps. The domain is field service, but
the skills transfer to any multi-tenant agent product.

| Week | Module | You build | Key result |
|---|---|---|---|
| 0 | [Setup and the synthetic world](00-setup.md) | Dev environment, a seeded generator of tenants, customers, assets, jobs and transcripts with labelled failure modes | A world you can evaluate against |
| 1 | [LangGraph and prompts](01-langgraph-and-prompts.md) | Job Intake agent: graph, checkpointer, human review interrupt, tenant memory, prompt registry | A booking needs a human's approval |
| 2 | [Evaluations](02-evaluations.md) | Datasets, heuristic evaluators, calibrated judge, runner, gate, online scoring, PromptFoo | CI blocks a regressing prompt |
| 3 | [RAG and knowledge](03-rag-and-knowledge.md) | Engineer Copilot: chunking ablation, hybrid search, RRF, reranking, faithfulness | Recall@5 0.66 to 1.00 from chunking |
| 4 | [Tools, channels and guardrails](04-tools-channels-guardrails.md) | Tool registry, MCP server, Celery workers, SMS/email/voice, layered guardrails, red team | Exactly-once effects, tenant-safe tools |
| 5 | [Applied ML and Databricks](05-applied-ml-databricks.md) | Urgency, duration and failure models, MLflow registry, asset bundle, failure-mode analysis | Agent ablation catches a "better" model that hurts |
| 6 | [Production hardening](06-capstone-hardening.md) | Tracing, cost ledger and budgets, feedback loop, Docker, Bicep, canary deploy, runbook | Bad run to permanent test; safe deploys |
| + | [Interview track](interview/README.md) | STAR stories, system design drills, take-home, prompt-debugging drill | Evidence for common requirements |

## How each module is laid out

- **Why it matters**: where the skill shows up in real AI engineering work.
- **Concepts**: what to understand, with links to primary docs and papers.
- **Code tour**: which capstone files implement it.
- **Results**: real numbers from running the code, and the war stories behind them.
- **Lab**: what you build or extend, with acceptance criteria. Items marked *required* are the
  ones interviewers will probe.
- **Production bar**: the checklist that makes it more than a demo.
- **Interview questions**: with answer outlines.

## Suggested weekly rhythm (25 hours)

| Block | Hours | What |
|---|---|---|
| Read | 4 | Concepts and the primary docs linked in the module |
| Code tour | 3 | Read the capstone code for the week, run the tests, break something on purpose |
| Lab | 14 | The required labs; stretch goals only if the required ones are done |
| Write | 2 | An eval report or design note in `docs/` with real numbers |
| Rehearse | 2 | Answer the week's interview questions out loud |

## Rules that make this "production-level"

1. Every agent change ships behind an eval gate in CI.
2. Tenant isolation is enforced in memory, RAG and tools, and tests prove it.
3. Code is typed (mypy), tested (pytest), traced, and has documented cost and latency.
4. Every number you'd quote in an interview is reproducible from a command in the repo.
5. Every bug you fix gets a regression case.

## Getting started

Start with [Week 0](00-setup.md). If you're short of time, the minimum path is Weeks 1, 2, 5 and
6: the agent, its evals, the ML, and running it in production. Those are the parts interviews
weigh most.
