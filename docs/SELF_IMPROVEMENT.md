# Self-improvement architecture

Atom Zero now defaults to a **closed, autonomous improvement loop** bounded by
an isolated worktree, focused and full test gates, a self-code diff and a
rollback. Per-change human approval is optional via `--require-approval`.

## Loop

```text
local use
  -> feedback queue / knowledge gaps / claim audits / reliability / freshness
  -> weakness signals (counts, ids, no unnecessary content)
  -> prioritized improvement plan
  -> bridge packet + opt-in
  -> external agent in a disposable worktree
  -> focused allowlisted gate
  -> self-code before/after diff
  -> full test suite in the sandbox
  -> local merge + post-merge full suite
  -> content-free decision ledger
  -> next plan
```

See `docs/AUTONOMOUS.md` for the sandbox boundary, rollback and the optional
human gate.

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
- Carry `requires_human: true` in the packet as a traceable label; autonomous
  mode auto-approves it after the automated gates pass.
- Run an allowlisted test/AISC gate and record the result.
- Record approve/reject/defer with an action id and a note hash.
- Improve retrieval, tools, prompts or tests after the gates pass.
- Merge a verified change locally and roll it back if the post-merge suite
  fails.

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

`docs/LOOP_CYCLE_3.md` records the first **agent-bridge consumed** cycle: a
live drift profile-format failure was packaged into a bounded task packet, an
external Implementer worked in `fix/drift-profile-format`, the gate passed
(`gate_id d38196191f97b81c`), approval recorded `gate_ok: true`, the real drift
report now works, and the scoreboard moved from `800` to `813` passed with no
trend alerts.

`docs/LOOP_CYCLE_8.md` records the first **fully autonomous** cycle: no
per-change human approval, a disposable sandbox worktree, focused gate,
self-code diff, 1066-test full suite, local auto-merge, 1066-test post-merge
suite, cleanup and decision (`gate_id da83c0302933d484`,
`1064/0 -> 1066/0`). The same cycle added the sandbox backend report and
`--require-approval` as an optional high-risk gate.

`docs/LOOP_CYCLE_7.md` records a **targeted audit** cycle: the loop was pinned
to `src/atomzero/aisc_coverage.py`, the external agent narrowed the broad
handler and added two tests, and the before/after audit gate caught its own
truncation false positive before the fixed run passed
(`gate_id 7f0d5be2bce36219`, `1033/0 -> 1035/0`). The gate now audits both
sides with a high per-kind cap.

`docs/LOOP_CYCLE_6.md` records the first **self-code fed** cycle: the AST
audit found a broad exception in `agent_runner.verify`, the planner emitted
`imp-01-self_code_medium`, the opt-in runner executed an external agent in an
isolated worktree, the gate passed (`gate_id cac82f022c879258`), approval
recorded `gate_ok: true`, and the scoreboard moved from `855` to `857` passed.
The same cycle exposed a constant decision-note id; the loop now binds the note
to the action id and gate id.

`docs/LOOP_CYCLE_5.md` records the first **autonomous opt-in runner loop**: a
live runner-scope signal was planned, packaged, executed by an actual nested
`opencode run` process, verified (`gate_id 7fb857e6552ae5cc`), approved with
`gate_ok: true`, and the scoreboard moved from `844` to `846` passed.

`docs/LOOP_CYCLE_4.md` records the first **opt-in runner** cycle: a live
policy/ECE drift signal was packaged, `atomzero-agent-run` prepared the
worktree, an external Implementer added policy-metric alerts, the runner
verified scope and gate (`gate_id 5134439e1f4e69df`), approval recorded
`gate_ok: true`, and the scoreboard moved from `821` to `839` passed with no
trend alerts.

## Near-term roadmap

1. Autonomous sandboxed loop (implemented; default).
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
- When a self-code audit is used, the loop compares before/after reports and
  blocks a decision on a new high/medium finding; it does not prove that every
  finding is a real bug.
- The decision ledger is local and unsigned; `verify` checks shape and
  content-free rules, not authorship.
- Human approval is optional for high-risk runs (`--require-approval`), not a
  per-change default.
- The worktree sandbox is a blast-radius reduction, not an OS-level security
  boundary. See `docs/AUTONOMOUS.md`.
