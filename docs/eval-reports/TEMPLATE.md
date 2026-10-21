# Eval report NNN: <change under test>

**Author:** | **Date:** | **Decision:** ship / don't ship / ship behind canary

## Change
What changed (prompt version, model, code) and the hypothesis: which failure mode should improve, and why.

## Setup
| | Control | Candidate |
|---|---|---|
| Prompt ref | | |
| Model | | |
| Dataset (split, version, n) | | |
| Judge model + rubric version (+ kappa) | | |

## Headline results
| metric | control | candidate | delta | 95% CI / p-value |
|---|---|---|---|---|
| outcome_correct | | | | McNemar p = |
| safety_escalation | | | | (must be 1.0) |
| no_sensitive_leak | | | | (must be 1.0) |
| trajectory_match | | | | |
| judge_groundedness | | | | |
| cost per 1k messages (USD) | | | | |
| p95 latency (ms) | | | | |

## Slices (by failure mode)
Which slices moved, and which regressed?

## Fixed / broken examples
- Fixed (n=): 2-3 representative ids with a one-line explanation each.
- Broken (n=): every one, with a one-line root cause.

## Risks and follow-ups
- What this eval does *not* cover.
- New dataset rows added because of this analysis.
