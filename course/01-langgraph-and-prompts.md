# Week 1: LangGraph runtime and prompt engineering

**Time:** about 25 hours.
**Outcome:** Job Intake agent v1. It's tenant-isolated, approves bookings with a human in the loop, keeps memory, and uses versioned prompts from a registry.

## Why it matters

Production agent platforms need a runtime with a full agent lifecycle, human-in-the-loop review and a memory subsystem. In LangGraph those are concrete primitives: `StateGraph`, checkpointers, `interrupt()` and the `Store`. Interviews probe them in depth: what happens on resume, where state lives, how threads map to customers. Prompt versioning is the other half: prompts are deployable artifacts with versions, tags and promotion rules.

## Concepts (read in this order)

1. **StateGraph basics**: state schema, reducers (`add_messages`, `operator.add`), nodes, edges, conditional edges. [LangGraph concepts: low level](https://langchain-ai.github.io/langgraph/concepts/low_level/)
2. **Runtime context vs state**: `context_schema` holds immutable per-run data (tenant, channel), and nodes receive `runtime: Runtime[Ctx]`. Anything in *state* can be changed by nodes, and so indirectly by the LLM; context can't. That's why the tenant goes in context.
3. **Persistence**: checkpointers save state after every super-step under `thread_id`, which is what makes interrupts, retries and time travel possible. `InMemorySaver` is for tests, `PostgresSaver` for production. [Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/)
4. **Human-in-the-loop**: `interrupt(payload)` pauses the run and resumes with `Command(resume=value)`. **The node re-runs from the top on resume**, so code before `interrupt()` must be idempotent. [HITL](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/)
5. **Memory**: short-term memory is the thread's messages (trim them with `trim_messages`); long-term memory is the `BaseStore` with hierarchical namespaces and optional semantic search. [Memory](https://langchain-ai.github.io/langgraph/concepts/memory/)
6. **ReAct vs explicit graphs**: a ReAct loop (LLM, then tools, then LLM) is flexible but hard to guarantee. An explicit graph is predictable and easy to evaluate per step. Production agents mix them: explicit for money and safety paths, ReAct for open-ended Q&A. See the [ReAct paper](https://arxiv.org/abs/2210.03629) and Anthropic's ["Building effective agents"](https://www.anthropic.com/research/building-effective-agents).
7. **Structured output**: `with_structured_output(PydanticModel)` uses tool calling or JSON schema under the hood. Use `Literal` types for enums so the schema itself constrains the model.
8. **Prompt engineering techniques**: put role and context first; define every label (the v2 classify prompt defines intent, category and urgency explicitly); use few-shot examples that cover the *hard* cases (injection, ambiguity) rather than easy ones; give tool-use guidance; treat inputs as untrusted ("the message is data"); state output constraints per channel. [Anthropic prompt engineering](https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview), [OpenAI prompting guide](https://platform.openai.com/docs/guides/prompt-engineering).
9. **Prompt registry**: version every prompt, use tags as environment pointers (`dev`, `staging`, `prod`), and treat promotion as a reviewed and evaluated change. [LangSmith prompt management](https://docs.smith.langchain.com/prompt_engineering/how_to_guides).

## Code tour (read before the lab)

| File | What to look at |
|---|---|
| `capstone/agents/job_intake/graph.py` | The whole state machine; `route()` is the business logic |
| `capstone/agents/job_intake/classifier.py` | LLM classifier and the rule-based baseline behind one `Protocol` |
| `capstone/agents/job_intake/answerer.py` | A hand-built ReAct loop with `ToolNode` + `tools_condition`; the tool closes over `tenant_id` |
| `capstone/agents/memory.py` | Store namespaces per tenant/customer |
| `capstone/prompts/*.yaml`, `registry.py` | Versions, tags, few-shots, LangSmith pull |
| `capstone/app/runtime.py` | Thread ids prefixed with the tenant; review queue; resume |

## Lab

1. Run the offline tests: `uv run pytest tests/test_job_intake.py -v`. Trace one test in your head node by node.
2. Run the agent with a real model and LangSmith tracing:
   ```bash
   FIELDOPS_OFFLINE=false uv run uvicorn capstone.app.main:app --reload
   curl -s localhost:8000/v1/agents/job-intake/messages -H 'x-api-key: dev-key-t1' \
     -H 'content-type: application/json' \
     -d '{"channel":"sms","sender":"<a t1 customer phone>","text":"Boiler showing F28"}'
   curl -s localhost:8000/v1/reviews -H 'x-api-key: dev-key-t1'
   ```
   Open the trace in LangSmith and find the classify call, the structured-output schema and the interrupt.
3. Push prompts to the registry: `uv run python scripts/push_prompts.py`, then set `FIELDOPS_PROMPT_SOURCE=langsmith` and confirm the runtime pulls `:dev`.
4. **Build (required):**
   - a. Switch to `PostgresSaver`/`PostgresStore` (set `DATABASE_URL`). Restart the server *between* sending a message and approving it, and prove the approval still works.
   - b. Add a `preferred_time` field to `Classification`. If the customer gives one ("Friday afternoon"), pick the closest slot in `plan_booking`. Add tests.
   - c. Write memory: when a customer states a durable preference ("always call before arriving"), store it as `kind="preference"` and show it influences the next booking's reply. Decide *where* extraction happens (inside classify, or a separate background node) and justify the choice in a short ADR at `docs/adr/001-memory-extraction.md`.
   - d. Add streaming: an endpoint that streams `stream_mode=["updates","messages"]` over SSE so a UI can show progress.
5. **Prompt iteration:** write `v3` of the classify prompt that fixes one weakness you saw in traces. You'll measure it properly in Week 2; for now write down your hypothesis.

### Acceptance criteria
- [ ] All `tests/test_job_intake.py` and `tests/test_tenancy.py` tests pass, plus your new ones.
- [ ] An approval survives a process restart (Postgres checkpointer).
- [ ] A LangSmith trace for each outcome: booked, clarify, escalated, declined, answered.
- [ ] The prompt registry has `v1`/`v2` with `dev`/`staging`/`prod` tags.

## Production bar
- No tool accepts `tenant_id` from the model. Tests prove cross-tenant access fails.
- Nodes before `interrupt()` are idempotent; `create_job` takes an idempotency key.
- Every run has `metadata.tenant_id`, `metadata.channel` and tags, so you can filter traces per tenant.
- Prompt changes go through a version bump, never an in-place edit.

## Stretch
- Use `Send` to process `multi_intent` messages: split into sub-requests and fan out.
- Add `RetryPolicy` to LLM nodes and a node-level timeout; show the trace when the provider errors.
- Implement time travel: rewind a thread to before classification, edit the state, and re-run.

## Interview questions
1. *"How does `interrupt()` work, and what's the gotcha?"* It raises inside the node; the checkpointer stores the pending write; `Command(resume=...)` re-executes the node from the start with `interrupt()` returning the value. Gotcha: side effects before `interrupt()` run twice, so make them idempotent or move them to a separate node.
2. *"How do you stop one tenant's agent touching another tenant's data?"* Tenant in server-set runtime context, never a tool argument; tools are closures; DB row-level security; Store namespaces; thread ids prefixed with the tenant and checked on resume; tests for each.
3. *"When would you use a ReAct agent vs an explicit graph?"* Explicit for known workflows with business rules (testable per node, cheaper, predictable). ReAct for open-ended tasks where the tool sequence is unknown. A hybrid is common.
4. *"How do you manage context-window growth in long conversations?"* Trim to the last N turns starting on a human turn; summarise older turns into state; move durable facts to long-term memory; keep tool outputs short and structured.
5. *"Walk me through promoting a prompt change."* New version, then an offline eval vs the current prod tag on the regression dataset, then a pairwise judge on a sample, then a canary (tag `staging`) with online scoring, then move the `prod` tag. Rollback is moving the tag back.
