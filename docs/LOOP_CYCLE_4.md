# Fourth measured self-improvement cycle - opt-in runner

This is the first cycle where the bridge packet was executed through the opt-in
`atomzero-agent-run` runner and the gate ran inside the isolated worktree.

## 1. Signal and packet

A live drift comparison of the stored signal and transformer answerability
profiles produced `ece_increased`. The planner selected
`imp-01-calibration_drift` (`requires_human: True`). The bridge packet allowed:

```text
src/atomzero/drift.py
tests/test_drift.py
docs/DRIFT.md
```

## 2. Opt-in runner

`atomzero-agent-run prepare` created:

```text
branch: agent/imp-01-calibration_drift
worktree: C:\Users\biris\AppData\Local\Temp\atom-zero-agent-run4
packet: .agent-task.json
```

An external Implementer worked in that worktree and committed `f425264`
(`feat: report policy-metric drift`): top-level/nested policy metrics,
`refusal_recall_decreased`, `refusal_precision_decreased` and
`answerable_coverage_decreased` alerts. Merge: `1a2b704`.

The gate initially failed because the caller's `PYTHONPATH` leaked the main
checkout into the worktree test process. That runner bug was fixed in
`94c74eb`; the gate now isolates `PYTHONPATH` to the target worktree.

## 3. Runner verification

```text
atomzero-agent-run verify --packet .../.agent-task.json \
  --worktree .../atom-zero-agent-run4 --archive .../gates4b.jsonl
VERIFY_EXIT=0
status: verified
out_of_scope: []
gate_id: 5134439e1f4e69df
total 2, passed 2, failed 0
```

The real profile comparison now reports:

```text
alerts: ece_increased, refusal_precision_decreased,
        answerable_coverage_decreased
refusal_precision:  0.5526 -> 0.4932 (-0.0594)
answerable_coverage: 0.6699 -> 0.2718 (-0.3981)
```

## 4. Human-gated decision

```text
record_gated_decision(... decision="approve", gate=<gate report>)
decision approve, gate_ok True, note_id 9347fb604d401146
verify_ledger -> {"ok": true, "entries": 1, "errors": []}
```

## 5. Measured result

```text
pre  scoreboard: commit bd5ac21, 821 passed, 0 failed
post scoreboard: commit 1a2b704, 839 passed, 0 failed
delta: tests_passed +18, tests_failed 0
trend ok: True, alerts: []
```

Full suite: **839 passed**. Self-improvement benchmark: **12/12**. No model
weights changed; weights remain `0.8.0`.

## Honest boundary

- The external Implementer was a subagent run by the development harness, not
  an agent spawned by the local Atom Zero runtime.
- A human approved the change; the planner proposed it, the bridge bounded it,
  and the runner isolated and verified it.
- The direct signal is policy/ECE drift; the scoreboard delta counts tests.
- The runner checks scope after the work, not before; that is why the allowed
  file list and the scope failure are both surfaced.
