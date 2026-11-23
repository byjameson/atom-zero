# Third measured self-improvement cycle - agent-bridge consumed

This is the first cycle where `atomzero-agent-bridge` produced the packet and an
external Implementer subagent consumed it in an isolated worktree.

## 1. Signal and packet

A live drift run on the stored profiles failed:

```text
python -m atomzero.drift \
  --baseline benchmarks/results/answerability_signal/answerability_profile.json \
  --profile benchmarks/results/answerability_transformer/answerability_profile.json
input error: profile.brier must be a number
```

The transformer profile stores `brier`/`ece` under `metrics` and `n` as a
mapping. The improvement planner selected `imp-01-calibration_drift`
(`requires_human: True`), and `atomzero-agent-bridge` emitted a bounded packet:

```text
allowed_files: src/atomzero/drift.py, tests/test_drift.py, docs/DRIFT.md
checks: pytest:tests/test_drift.py, pytest:tests/test_improvement.py
handoff: external_agent_harness_only
auto_start: False
```

## 2. External Implementer

An Implementer subagent worked in the isolated
`fix/drift-profile-format` worktree. Commit `5dae7de` added top-level/nested
metric lookup, scalar/mapping/metrics `n`, resolved thresholds and a
`threshold_changed` alert. Merge: `a466c28`.

The real profile comparison now works:

```text
status: drifted
alerts: ["ece_increased"]
Brier: 0.2703 -> 0.2762 (+0.0058)
ECE:   0.1407 -> 0.1612 (+0.0205)
n:     200 -> 200
```

The default Brier tolerance is `+0.02`, so the +0.0058 move is intentionally
below that alert; ECE exceeds its `+0.02` tolerance.

## 3. Gate

```text
python -m atomzero.evaluation \
  --check pytest:tests/test_drift.py \
  --check pytest:tests/test_improvement.py \
  --archive .../gates3.jsonl --json
GATE_EXIT=0
gate_id d38196191f97b81c
total 2, passed 2, failed 0
```

## 4. Human-gated decision

```text
record_gated_decision(... decision="approve", gate=<gate report>)
decision approve, gate_ok True, note_id 0e3a20ce697dcfda
verify_ledger -> {"ok": true, "entries": 1, "errors": []}
```

## 5. Measured result

```text
pre  scoreboard: commit 9905fa3, 800 passed, 0 failed
post scoreboard: commit a466c28, 813 passed, 0 failed
delta: tests_passed +13, tests_failed 0
trend ok: True, alerts: []
```

The full suite also needed a wall-clock test fix: `test_brief_tag_filter...`
derived task dates from `date.today()` while evaluating the brief as
`2026-11-23`; it now derives fixtures from the same fixed date. That is an
additional regression guard in the 813.

## Honest boundary

- The external Implementer was a subagent in this development harness, not the
  local Atom Zero runtime spawning an agent.
- A human approved the action; the planner proposed it and the bridge packaged
  it.
- The direct weakness evidence is the profile-format failure becoming a working
  drift report; the scoreboard delta counts tests, not a product-quality score.
- No weights changed; weights remain `0.8.0`.
