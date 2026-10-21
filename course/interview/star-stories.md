# STAR stories

Behavioural rounds for AI engineering roles keep asking the same eight questions in different
words. Prepare one story for each, from your own work and from this capstone. Keep your drafts
somewhere private (`course/interview/private/` is git-ignored): real stories contain employer
details that don't belong in a public repo.

## How to tell them

- About 2 minutes each. Situation and task in 20 seconds; most of the time on **your** actions
  (say "I", not "we"); end with a measurable result and what you'd do differently.
- Every number must be one you can defend. If an interviewer asks "how many?" and gets a shrug,
  they stop believing the rest. No number? Drop the claim.
- Rehearse out loud and record yourself at least once.

## The eight stories to prepare

| # | Prompt you'll hear | What it proves | Capstone material if you lack a work story |
|---|---|---|---|
| 1 | "Tell me about a production incident you debugged." | Calm, evidence-led debugging; communication | Stale eval fixtures (Week 5) |
| 2 | "Describe integrating a difficult third-party API or model provider." | APIs, schemas, tool calling | Twilio/Nylas webhooks and idempotency (Week 4) |
| 3 | "How have you measured cost or usage accurately?" | Data pipelines, observability | The cost ledger and budgets (Week 6) |
| 4 | "Tell me about building tools or integrations for an agent." | Tools, MCP, multi-tenancy | Tool registry and tenant-scoped MCP server (Week 4) |
| 5 | "How do you know your agent is good?" | Evaluation design, the core skill | Eval suite and CI gate; judge calibration (Week 2) |
| 6 | "Tell me about making a system more reliable." | Production engineering | Exactly-once message handling (Week 4) |
| 7 | "Describe working on security or tenant isolation." | Multi-tenant B2B thinking | Tenant isolation in memory, RAG and tools (Weeks 1-4) |
| 8 | "Tell me about a model that didn't behave as expected." | Applied ML judgement | The Naive Bayes champion (below) |

## Template

```markdown
## <title>
**Situation.** Where, when, what was at stake (one or two sentences).
**Task.** What *you* were responsible for.
**Action.** 3-5 bullets of what you did, in order, with the reasoning behind the key choice.
**Result.** A number, a shipped change, a customer outcome.
**Lesson.** What you'd do differently, tied to the role you're interviewing for.
*Likely follow-ups:* the two questions you'd ask if you were the interviewer, with answers.
```

## Worked example: the model that made the agent worse

**Situation.** In the FieldOps capstone I trained an urgency classifier as a second opinion for
the agent's triage. The Complement Naive Bayes model had the best macro-F1 (0.895 vs 0.893).

**Task.** Show that adding the model made the agent better, not just that it scored well.

**Action.**
- Ran an agent-level ablation instead of trusting the offline metric. With the model on, the
  agent's urgency score *dropped* by 0.031.
- Read the failures: vague messages ("something's broken in the plant room") upgraded to
  emergency at about 71% confidence. Naive Bayes probabilities are overconfident (ECE 0.107).
- Changed model selection to prefer the lowest log loss among models within one standard
  deviation on F1. That picked calibrated logistic regression (ECE 0.015).
- Made the policy asymmetric: the model can only upgrade to emergency, at high confidence.

**Result.** Zero wrong upgrades and flat triage metrics, while every booking gained a duration
estimate and slot sizing booked about 5% fewer hours at the same overrun rate.

**Lesson.** Evaluate a model where it's used, and choose metrics for the decision it drives.

*Likely follow-ups:* "Why not just raise the threshold?" (that treats the symptom; uncalibrated
probabilities make any threshold arbitrary). "How would you catch this in CI?" (run the agent
ablation as part of the eval gate whenever a model version changes).

## A strong incident story has this shape

Interviewers love incident stories because they show judgement under pressure. The best ones:
1. **Rule out your own change with evidence** (commits, deploy log, pinned dependency versions),
   not opinion.
2. **Reproduce at the lowest level** (a raw HTTP request with no SDK) to separate your bug from
   an upstream change.
3. **Find every place the pattern exists**, not just the one that was reported.
4. **Choose the robust fix** over the quick one, and say why.
5. **Communicate twice:** an internal write-up with evidence, and an honest customer-facing one.
6. **Close with prevention:** the synthetic probe, eval or alert that would have caught it first.
