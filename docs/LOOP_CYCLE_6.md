# Sixth measured self-improvement cycle - self-code branch feeds the loop

This is the first cycle where the **live signal came from Atom Zero's own
source code**, not from a document or an audit of retrieval behavior.

## 1. Live signal

`atomzero-self-code --root . --propose` scanned the repository with the AST
detector and found a medium finding:

```text
medium broad_exception src/atomzero/agent_runner.py:157
the gate call swallowed every Exception, including unexpected bugs
```

The planner turned it into:

```text
imp-01-self_code_medium [self_code_medium]
requires_human: True
```

The bounded proposal was:

```text
allowed_files: src/atomzero/agent_runner.py, tests/test_agent_runner.py
checks:        pytest:tests/test_agent_runner.py
risk:          medium
requires_human: true
auto_apply:     false
```

## 2. External execution

`atomzero-self-improve` ran the audit, built the packet from the proposal and
started an actual nested `opencode run` process in an isolated worktree.

```text
worktree: C:\Users\biris\AppData\Local\Temp\atom-zero-selfloop6\worktree
branch:   agent/imp-01-self_code_medium
commit:   b1be750 Narrow agent_runner gate exception handling
merge:    071408a merge: self-code narrow gate exception
```

The external agent changed only the allowed files:

```text
src/atomzero/agent_runner.py | 2 +-
tests/test_agent_runner.py   | 30 ++++++++++++++++++++++++++++++
```

It narrowed the handler to expected failures:

```python
except (OSError, ValueError, subprocess.SubprocessError) as exc:
```

and added two regression tests: unexpected `KeyError` propagates; expected
`OSError` becomes a failed gate report.

## 3. Gate and decision

```text
atomzero-agent-run verify --base main --archive .../gates.jsonl
status: verified
out_of_scope: []
gate_id: cac82f022c879258
total 1, passed 1, failed 0
decision: approve, gate_ok: True
note_id: 12561b75dca58765
```

The note id was the same as cycle 5 because the loop used one constant note
string. That weakness was fixed immediately after this cycle: the note now
binds the action id and gate id, so future decision ids are unique per gate
(`test_loop_completes_and_records` covers the hash).

## 4. Measured result

```text
pre  scoreboard: commit d719b6c, 855 passed, 0 failed, AISC 20/22, benchmark 13/13
post scoreboard: commit 62f5bd2, 857 passed, 0 failed, AISC 20/22, benchmark 13/13
delta: tests_passed +2
trend ok: True, alerts: []
```

- Full suite: **857 passed**.
- AISC strict: **22 specs, 20 PASS, 2 expected answerability failures**.
- Self-improvement benchmark: **13/13**.
- No model weights changed; weights remain `0.8.0`.

## Honest boundary

- The audit and proposal are deterministic; the code edit was made by an
  external agent invoked through an explicit opt-in runner command.
- A human still sets `--opt-in` and `--approve`; `--veto` stops before work.
- The audit is conservative and lexical/AST based. It found one real broad
  handler; many other broad handlers in HTTP and best-effort paths remain
  intentional and should be reviewed, not blindly rewritten.
- The gate ran one focused test file. The full suite was run after the merge.
- The external agent is not sandboxed by Atom Zero; scope is checked after it
  runs.
