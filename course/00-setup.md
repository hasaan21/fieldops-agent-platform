# Week 0: Setup and the synthetic world

**Time:** one weekend (about 8 hours).
**Outcome:** a reproducible dev environment, plus a synthetic field-service dataset that every later lab uses.

## Why it matters

The capstone is a multi-tenant FSM (field service management) platform: tenants are field-service businesses, and they have customers, sites, assets, jobs and engineers. Real customer data isn't available for a portfolio project (and shouldn't be), so build a world shaped like it. That lets you show (a) eval datasets with known gold labels, (b) tenant isolation you can actually test, and (c) ML models with real signal to learn. A generator you can tune ("inject 20% prompt injections") is also a strong interview story: it's how you test agents before you have production traffic.

## Concepts

- **Synthetic data for evals.** Templates plus controlled noise give you exact labels. LLM-paraphrased data gives you variety but noisy labels. Start with templates, add an LLM paraphrase pass later, and keep the labels from the template. See [LangSmith: synthetic data](https://docs.smith.langchain.com/evaluation/how_to_guides) and the [Anthropic eval guide](https://docs.anthropic.com/en/docs/test-and-evaluate/develop-tests).
- **Injected failure modes.** Every dataset row records *which* failure mode it tests (`labels.failure_mode`). That lets you slice metrics later ("injection resistance is 92%, but missing-address handling is 61%").
- **Data with a causal story.** In `asset_snapshots`, failure hazard is a known function of age, overdue servicing, recent faults and runtime. A good model should recover those effects, and you can check that it does (Week 5).
- **Deliberately per-tenant facts.** Call-out fees, warranties and hours differ by tenant. If tenant `t1` ever answers with `t2`'s fee, your isolation is broken, and the test suite will catch it.

## Lab

1. Install the tools: `uv`, Docker, Node 20+ (for PromptFoo). Sign up for free tiers of LangSmith and [Databricks Free Edition](https://www.databricks.com/learn/free-edition).
2. Install the project:
   ```bash
   uv sync --extra ml
   cp .env.example .env              # fill in OPENAI_API_KEY and LANGSMITH_API_KEY at minimum
   docker compose -f infra/docker-compose.yml up -d postgres redis
   ```
3. Generate the world:
   ```bash
   uv run python -m capstone.data.synthetic.generator
   ```
4. Read `capstone/data/synthetic/generator.py` end to end. Then explore the data in a notebook or the REPL:
   ```python
   from capstone.data.loader import load_frame
   tx = load_frame("transcripts")
   tx["labels"].apply(lambda l: l["failure_mode"]).value_counts(dropna=False)
   ```
5. **Extend the generator (required):** add one new failure mode, `wrong_tenant`, where a customer of `t1` messages `t2`'s number about a `t1` job. Give it the gold label `should_book=False`, a new label `needs_clarification=True`, and a sensible `expected_tools`.
6. **Stretch:** add `--paraphrase`, which rewrites `message` with an LLM (temperature 0.9) while keeping the labels. Record a label-drift check: re-classify 50 paraphrases by hand and see how many labels no longer hold.

### Acceptance criteria
- [ ] `docker compose ps` shows postgres (with pgvector) and redis healthy.
- [ ] `uv run pytest tests/test_synthetic.py` passes.
- [ ] Your `wrong_tenant` failure mode appears in `transcripts.jsonl` and has a test.

## Production bar
- Generation is deterministic (seeded); the same seed always produces the same bytes.
- No real PII anywhere; `example.com` emails and fictional brands only.
- Generated output is git-ignored. The generator is the source of truth.

## Interview questions
1. *"We have no labelled data for a new agent. How do you start evaluating it?"* Outline: write 20-50 golden cases by hand from domain experts; add synthetic templates with failure modes; deploy with tracing to an internal group; then convert reviewed production traces into dataset rows (the loop you build in Week 6).
2. *"What are the risks of LLM-generated eval data?"* Outline: label drift, mode collapse (everything sounds the same), contamination with the model under test, and over-optimistic difficulty. Mitigate with human spot checks, a different generator model than the one under test, and stratifying by failure mode.
