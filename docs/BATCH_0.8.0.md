# Batch 0.8.0 - self-improvement loop

`0.7.0` closed the ten-increment evidence/temporal batch. The next batch builds
the **planning layer of a self-improving system**: local weakness signals become
prioritized, human-gated improvement actions with explicit acceptance gates.
It does not train or apply changes automatically.

## The loop

```text
local signals
  -> prioritized improvement plan
  -> human approval
  -> evaluation gate (tests / AISC / benchmark)
  -> decision ledger (content-free)
  -> next plan
```

## Increments

| # | Increment | Status |
|---|---|---|
| 1 | Human-gated improvement loop (`atomzero-improve`) | merged on local main |
| 2 | `POST /v1/improvement/plan` + content-free decision route | merged on local main |
| 3 | Evaluation gate harness (pytest, AISC strict, focused benchmark) | merged on local main |
| 4 | Decision ledger integration into weekly review | planned |
| 5 | Gap-to-research approval link | planned |
| 6 | Feedback-to-improvement link | planned |
| 7 | Reliability-to-source-review link | planned |
| 8 | Proactive improvement reminders | planned |
| 9 | AISC coverage in the improvement plan | planned |
| 10 | Self-improvement regression benchmark | planned |

## Hard boundaries

- No automatic weight updates and no silent prompt/tool changes.
- Every action carries `requires_human: True`.
- Decision records are content-free: action id, kind, decision, timestamp and a
  note hash, never the note text.
- An improvement is "accepted" only after the acceptance gate passes; a failed
  gate stays visible.
- The system may propose its own next work, but a human approves the change.

## Research basis

The branch research note (`docs/BRANCH_RESEARCH.md`) covers claim-level
verification, source reliability, temporal RAG, conflicting evidence and
evidence sufficiency. This loop composes those signals; it does not claim they
are truth.
