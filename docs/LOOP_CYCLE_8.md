# Eighth measured self-improvement cycle - fully autonomous, no human approval

This is the first cycle where the loop ran without per-change human approval.
`--opt-in` enabled execution; the focused gate, self-code diff, full suite,
local merge and post-merge suite were the decision.

## 1. Live signal

```text
atomzero-self-code --root . --propose
medium broad_exception src/atomzero/audit.py:105
run_audit swallowed every Exception from a bad spec
```

The loop targeted that file and used the previous decision ledger as planner
history.

## 2. Autonomous run

```text
command: atomzero-self-improve --self-code-root . \
  --target-path src/atomzero/audit.py --opt-in --json
branch:  agent/imp-01-self_code_medium-20261124-015548
agent:   ef6a186 fix(audit): narrow bad-spec exception handling
merge:   d51fdd6 autonomous: merge imp-01-self_code_medium
```

The agent narrowed the handler to expected failures:

```python
except (OSError, ValueError, TypeError, KeyError) as exc:
```

and added tests: an expected bad-spec error is reported; an unexpected
`RuntimeError` propagates.

## 3. Gates

```text
focused gate:       1/1 PASS, gate_id da83c0302933d484
self-code diff:     143 -> 142
                    resolved 1, introduced 0, persisting 142
full sandbox test:  1066 passed, 0 failed
merge:              local main 308e13f -> d51fdd6
post-merge test:    1066 passed, 0 failed
cleanup:            worktree removed, branch deleted
decision:           approve, gate_ok true, note_id 68d3cef095336f92
```

## 4. Sandbox report

```json
{
  "backend": "worktree",
  "os_vm": false,
  "docker_available": false,
  "wsl_available": true,
  "isolation": "disposable worktree + sanitized env + timeout",
  "network_isolated": false
}
```

This machine has no usable Docker/VM backend for the loop. The worktree
sandbox removes token-like environment variables, uses a temporary home,
bounds the process with a timeout and never pushes or publishes. It is a
blast-radius reduction, not an OS-level security boundary.

## 5. Measured result

```text
pre  scoreboard: commit 308e13f, 1064 passed, 0 failed, AISC 23/25, benchmark 19/19
post scoreboard: commit d51fdd6, 1066 passed, 0 failed, AISC 23/25, benchmark 19/19
delta: tests_passed +2
trend ok: True, alerts: []
```

- Full suite: **1066 passed**.
- AISC strict: **25 specs, 23 PASS, 2 expected answerability failures**.
- Self-improvement benchmark: **19/19**.
- No model weights changed; weights remain `0.8.0`.

## Honest boundary

- The external agent is not sandboxed by Atom Zero itself; the harness owns its
  process boundary.
- The worktree sandbox can still read host files and does not block network at
  the OS level.
- Auto-merge changes local `main`; it does not push, tag or publish.
- The focused gate ran one test file; the full suite was run before and after
  the merge.
- Human approval remains available with `--require-approval` for high-risk
  runs, but it is not required by default.
