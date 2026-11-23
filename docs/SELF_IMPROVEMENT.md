# Self-improvement architecture

Atom Zero is not claiming autonomous recursive self-modification. The concrete
goal is a **closed, human-gated improvement loop**: the system notices its own
weaknesses, proposes the next work, and accepts a change only after an explicit
evaluation gate.

## Loop

```text
local use
  -> feedback queue / knowledge gaps / claim audits / reliability / freshness
  -> weakness signals (counts, ids, no unnecessary content)
  -> prioritized improvement plan
  -> human approval
  -> allowlisted evaluation gate (pytest / AISC / manual)
  -> content-free decision ledger
  -> next plan
```

## Existing components

| Stage | Component |
|---|---|
| feedback | `feedback.py`, `/v1/learning/feedback`, review queue |
| gaps | `knowledge_gaps.py`, `/v1/learning/gaps/*` |
| curiosity | `curiosity.py` proposes research from gaps |
| claims | `claims.py`, `POST /v1/claims/audit` |
| consensus | `consensus.py`, `POST /v1/consensus` |
| reliability | `reliability.py`, content-free ledger |
| freshness | `freshness.py`, `POST /v1/freshness` |
| temporal validity | `temporal.py`, `POST /v1/temporal/filter` |
| evaluation | `evaluation.py`, `atomzero-gate` |
| planning | `improvement.py` (in progress) |

## What is allowed

- Propose a prioritized action from local signals.
- Require `requires_human: True` on every action.
- Run an allowlisted test/AISC gate and record the result.
- Record approve/reject/defer with an action id and a note hash.
- Improve retrieval, tools, prompts or tests after review.

## What is not allowed

- No automatic weight update.
- No silent prompt/tool/skill mutation.
- No plan is accepted because the model says it is good.
- No content-free claim means "no data": sample sizes and limits stay visible.
- No failed gate is hidden or converted into a pass.

## Why this is still self-improvement

The system can generate its own next work from its own failures instead of
waiting for a person to enumerate every task. It can rank what matters, write a
plan and prepare the evidence. A human still owns the irreversible acceptance
step. That is the difference between a self-improving system and an uncontrolled
one.

## First completed cycle

`docs/LOOP_CYCLE.md` records a real signal -> change -> gate -> gated approval
-> scoreboard cycle: a reproducible Windows HTTP flake was fixed, the focused
gate passed (`gate_id ab5fe9b0fa0f48bc`), the approval was recorded with
`gate_ok: true`, and the measured result moved from `792 passed / 1 failed` to
`793 passed / 0 failed` with no trend alerts.

`docs/LOOP_CYCLE_2.md` records a second cycle selected from a live claim audit:
date/code digits no longer double-count as numeric conflicts, the two-target
gate passed (`gate_id ac1eb8a7ee8b5271`), approval was recorded with
`gate_ok: true`, and the scoreboard moved from `793` to `794` passed with no
trend alerts.

## Near-term roadmap

1. Human-gated improvement planner (in progress).
2. Plan and decision HTTP routes.
3. Evaluation gate integration with plan acceptance.
4. Decision ledger in the weekly review.
5. Gap -> research -> plan links.
6. Feedback -> plan links.
7. Reliability -> source review links.
8. Proactive improvement reminders.
9. AISC coverage signals in the plan.
10. A self-improvement regression benchmark.

The detailed batch is in `docs/BATCH_0.8.0.md`.

## Honest limits

- The signals are lexical and local. They can rank the wrong thing.
- The planner cannot prove that an action will improve the product.
- The evaluation gate proves only that selected checks passed.
- The decision ledger is local and unsigned; `verify` checks shape and
  content-free rules, not authorship.
- Human approval is a feature, not a temporary limitation.
