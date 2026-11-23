# Fifth measured self-improvement cycle - autonomous opt-in runner loop

This is the first cycle where `atomzero-self-improve` executed a live signal
through an external agent process end-to-end.

## 1. Live signal

A runner scope audit found that an external agent's **committed** out-of-scope
change was invisible to `agent_runner.verify`: it only inspected
`git status`, not committed changes. The improvement planner selected:

```text
imp-01-runner_scope_gap [runner_scope_gap]
requires_human: True
```

## 2. Self-improve loop

`run_loop` performed:

```text
signals -> planner -> bridge packet -> opt-in runner
```

The runner command was an actual nested `opencode run` process in the isolated
worktree. It returned 0.

Worktree: `C:\Users\biris\AppData\Local\Temp\atom-zero-selfloop2\worktree`
Branch: `agent/imp-01-runner_scope_gap`
Commit: `e91a209` (`fix: include committed changes in runner scope check`)
Merge: `bd5a9fe`

The external agent changed only the allowed files:

```text
docs/AGENT_RUNNER.md
src/atomzero/agent_runner.py
tests/test_agent_runner.py
```

`_changed_files` now unions `git diff --name-only base...HEAD` with
`git status --porcelain`; committed out-of-scope changes fail scope.

## 3. Gate

```text
atomzero-agent-run verify ... --base main --archive .../gates5.jsonl
status: verified
out_of_scope: []
gate_id: 7fb857e6552ae5cc
total 2, passed 2, failed 0
```

## 4. Human approval / veto

The loop required explicit `approve=True`; `veto=True` and missing approval
return without creating a worktree or a decision. Approval was recorded:

```text
decision approve, gate_ok True, note_id 12561b75dca58765
```

## 5. Measured result

```text
pre  scoreboard: commit 7031273, 844 passed, 0 failed
post scoreboard: commit bd5a9fe, 846 passed, 0 failed
delta: tests_passed +2, tests_failed 0
trend ok: True, alerts: []
```

Full suite: **846 passed**. Self-improvement benchmark: **12/12**. No model
weights changed; weights remain `0.8.0`.

## Honest boundary

- The external agent was a nested `opencode run` process selected explicitly by
  an opt-in runner command; Atom Zero still does not spawn agents by itself.
- A human still approves the action; the loop can execute only when both
  `opt_in` and `approve` are set, and `veto` stops it.
- The score delta counts tests; the direct weakness was the committed-change
  scope gap, now covered by a regression test.
- The external process is not sandboxed by Atom Zero; the harness owns that
  boundary.
