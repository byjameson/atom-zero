# Second measured self-improvement cycle

This cycle starts from a live claim-audit signal, not a synthetic scoreboard
regression.

## 1. Signal

A local claim audit on two dated policy claims reported both `date_mismatch`
and `numeric_mismatch` for the same pair:

```text
The policy starts on 2020-01-01.
The policy starts on 2021-01-01.
reasons: ["numeric_mismatch", "date_mismatch"]
```

The date digits were being counted again as a numeric difference. The
improvement planner selected:

```text
imp-01-conflicts [conflicts]
requires_human: True
suggested_checks: pytest:tests/test_claims.py,
                  pytest:tests/test_consensus.py
```

## 2. Change

In the isolated `fix/structured-token-conflict-noise` worktree, numbers inside
full dates and code tokens are removed before deciding whether an independent
numeric mismatch remains.

Files: `src/atomzero/contradictions.py`, `tests/test_claims.py`.
Commit: `fbe89e0`; merge: `6213fdf`.

The same audit after the fix reports only:

```text
reasons: ["date_mismatch"]
counts: numeric_mismatch 0, date_mismatch 1
```

Independent numeric conflicts are preserved: a full date plus a different
price still reports both reasons, and a code plus a different part count still
reports both.

## 3. Gate

```text
python -m atomzero.evaluation \
  --check pytest:tests/test_claims.py \
  --check pytest:tests/test_contradictions.py \
  --archive .../gates2.jsonl --json
GATE_EXIT=0
gate_id ac1eb8a7ee8b5271
total 2, passed 2, failed 0
```

## 4. Human-gated decision

```text
record_gated_decision(... decision="approve", gate=<gate report>)
decision approve, gate_ok True, note_id 67ec4d33504e04ec
verify_ledger -> {"ok": true, "entries": 1, "errors": []}
```

## 5. Measured result

```text
pre  scoreboard: commit 99fdf6f, 793 passed, 0 failed
post scoreboard: commit 6213fdf, 794 passed, 0 failed
delta: tests_passed +1, tests_failed 0
trend ok: True, alerts: []
```

Full suite on the merged main: **794 passed**. Self-improvement benchmark:
**12/12**. No model weights changed; weights remain `0.8.0`.

## Honest boundary

- A human selected and approved the action; the planner proposed the conflict
  action from the live claim audit.
- The measured test delta (+1) is a regression-guard count, while the direct
  weakness evidence is the cleaner conflict-reason set (`numeric_mismatch`
  removed, `date_mismatch` retained).
- This is not autonomous self-modification and does not enable the
  answerability gate.
