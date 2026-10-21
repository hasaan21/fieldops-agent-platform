# Timed take-home: the Quote Assistant

**Time box: 4 hours, strictly.** Set a timer and stop when it rings. Take-homes are graded on
judgement under a constraint as much as on code. Do it in week 6, after the capstone is done,
on a fresh branch (`take-home/quote-assistant`). Don't look at your capstone code for the first
30 minutes: plan from scratch first, then reuse what fits.

## The brief (as a company would send it)

> Field-service customers often ask "how much will it cost?". Build a **Quote Assistant** agent
> that replies to a quote request with an itemised *estimate* the customer can trust.
>
> Inputs: a customer message, the tenant (which carries `callout_fee_gbp`, `hourly_rate_gbp`,
> `trades`, `tone`), and optionally a known customer and asset.
>
> The estimate must:
> - only use the tenant's own prices; never invent a price, discount or "free" anything;
> - give a duration *range* for the job, not a single number;
> - decline politely if the work is outside the tenant's trades;
> - escalate instead of quoting if the message describes an emergency (gas smell, sparks, flooding);
> - return structured output (`QuoteEstimate`) plus a customer-facing reply for the channel.
>
> Deliverables:
> 1. The agent (any framework; LangGraph preferred), runnable offline with a fake or rule-based model.
> 2. An evaluation: a dataset of at least 40 cases, evaluators, and a results table.
> 3. A `DECISIONS.md` (one page max): what you built, what you cut, what you'd do with another week.

Use the capstone's synthetic world for data (`load_table("tenants")`, `"jobs"`, `"assets"`).

## Suggested plan (do your own first, then compare)

| Time | Do |
|---|---|
| 0:00-0:20 | Read the brief twice. Write `DECISIONS.md` headings and the `QuoteEstimate` schema. List the 5 eval metrics before writing any agent code. |
| 0:20-1:00 | Dataset: 40+ cases from templates (in-scope, out-of-scope, emergency, injection "quote me £0", vague, multi-job) with gold labels. Prices in the gold labels are *computed*, not written by hand. |
| 1:00-2:15 | Agent: classify, then a deterministic price calculation tool, then compose. The LLM never does arithmetic. |
| 2:15-3:15 | Evaluators and a results table; fix the top failure. |
| 3:15-3:45 | Tests for the two invariants: tenant prices only, and emergencies are never quoted. |
| 3:45-4:00 | Finish `DECISIONS.md`. Commit. Stop. |

## What a strong submission looks like

- **The model doesn't do maths.** Price = call-out + hourly rate × hours, computed in a tool from
  tenant config; the duration range comes from history (the category's p10-p90, or the duration
  model with conformal intervals if you reuse Week 5). The LLM picks the category and writes the
  reply.
- **Hard invariants are code, not prompt.** An output check verifies every £ amount in the reply
  appears in the computed estimate, and blocks "free", "discount" and "waive" unless they're in
  tenant config. Emergencies route away before quoting.
- **Evals that match the risks:** `price_exact` (every amount traceable to config),
  `no_invented_price`, `scope_correct`, `emergency_not_quoted` (threshold 1.0), `range_contains_
  actual` against historical job durations, reply length per channel. Report n and a confidence
  interval, and slice by case type.
- **Honest `DECISIONS.md`:** "I cut the LLM judge for tone because I had no time to calibrate
  it; without calibration it would be noise" is a better answer than an uncalibrated judge.
- **Tests run offline** with no API keys.

## Self-review rubric (score yourself, 1-5)

| Criterion | 1 | 5 |
|---|---|---|
| Correctness of invariants | Prices could be invented | Every amount is computed and verified by code |
| Evaluation design | Accuracy on 10 hand-picked cases | Metrics mapped to risks, gold labels computed, slices, thresholds |
| Agent design | One big prompt | Deterministic steps where possible, LLM only where needed |
| Code quality | Script | Typed, tested, small modules, runs from the README |
| Communication | No write-up | One page that a reviewer can act on |
| Scope control | Unfinished | Smaller scope, fully working |

Afterwards, write down where the four hours actually went. If evals got less than 1h15, that's
the habit to fix before the real one, because evaluation is what reviewers weigh most.
