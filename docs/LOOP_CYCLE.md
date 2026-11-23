# First measured self-improvement cycle

This is a completed, real cycle rather than a plan example. The weakness was a
reproducible Windows HTTP flake, not a synthetic signal.

## 1. Signal

Repeated full-suite runs showed `test_http_document_endpoints_and_grounded_chat`
failing with `ConnectionAbortedError: [WinError 10053]`. A pre-fix scoreboard
record was written:

```text
commit 5a490c9
tests_passed 792, tests_failed 1
benchmark 12/12, AISC 17/19
```

`plan_from_inputs(scoreboard=summary)` produced:

```text
imp-01-scoreboard_regression [scoreboard_regression]
requires_human: True
priority: 85
```

## 2. Change

In the isolated `fix/oversized-upload-413` worktree, oversized request bodies
are drained (up to 1 MiB) before the server writes a `413`. Previously the
server responded while unread bytes remained, so Windows could reset the client
connection before it read the response.

Files: `src/atomzero/chat_server.py`, `tests/test_chat_server_docs.py`.
Commit: `467dc44`; merge: `15d2bba`.

## 3. Gate

The allowlisted focused gate ran on the fixed tree:

```text
python -m atomzero.evaluation \
  --check pytest:tests/test_chat_server_docs.py \
  --archive .../gates.jsonl --json
GATE_EXIT=0
gate_id ab5fe9b0fa0f48bc
total 1, passed 1, failed 0
```

The gate archive contains only the check name, kind, status and duration.

## 4. Human-gated decision

```text
record_gated_decision(... decision="approve", gate=<gate report>)
decision approve, gate_ok True, note_id fdc2c0d61298d078
verify_ledger -> {"ok": true, "entries": 1, "errors": []}
```

The approval would have been rejected without a passing gate.

## 5. Measured result

A post-fix scoreboard record was written:

```text
commit 15d2bba
tests_passed 793, tests_failed 0
benchmark 12/12, AISC 17/19
```

Trend:

```text
delta: tests_passed +1, tests_failed -1, benchmark_passed 0, aisc_passed 0
trend ok: True
alerts: []
latest: 15d2bba, 793 passed, 0 failed
```

Full suite on the merged main: **793 passed**. Self-improvement benchmark:
**12/12**. The specific flaky test also passed 12 consecutive runs in the fix
worktree.

## Honest boundary

- The loop was human-driven: a human selected the weakness and approved the
  change. That is the current design, not a hidden claim of autonomy.
- The scoreboard measures selected tests, not product value.
- The gate shows the focused test passed; it does not prove the flake cannot
  appear elsewhere.
- No model weights were changed; the weights remain `0.8.0`.
