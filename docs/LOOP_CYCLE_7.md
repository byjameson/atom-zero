# Seventh measured self-improvement cycle - targeted audit, regression gate

This cycle used the new self-code targeting, decision history and before/after
audit gate on a real source finding.

## 1. Live signal

```text
atomzero-self-code --root . --propose
medium broad_exception src/atomzero/aisc_coverage.py:114
the per-spec loading handler swallowed every Exception
```

The loop was pinned to that file:

```text
atomzero-self-improve --self-code-root . \
  --target-path src/atomzero/aisc_coverage.py ...
```

The planner also received the previous cycle's decision ledger as bounded
history. The selected action was `imp-01-self_code_medium`; the packet scope
came from the matching proposal:

```text
allowed_files: src/atomzero/aisc_coverage.py, tests/test_aisc_coverage.py
checks:        pytest:tests/test_aisc_coverage.py
```

## 2. First run exposed a gate false positive

The external agent made the right change, the focused gate passed, but the
before/after audit reported:

```text
status: self_code_regression
resolved: 1
introduced: 1
introduced: broad_exception src/atomzero/chat_server.py:5311
```

`chat_server.py` was not changed. The cause was the audit's default
`max_per_kind=20` cap: removing one `broad_exception` let the next existing
finding enter the top 20, so the diff looked like a new regression.

Fix (`b55ee8f`): the loop now audits both sides with
`max_per_kind=200`, so truncation cannot manufacture a regression. A regression
test captures the cap. `docs/SELF_CODE_DIFF.md` documents the boundary.

## 3. Second run completed

```text
worktree: C:\Users\biris\AppData\Local\Temp\atom-zero-selfloop7\worktree
branch:   agent/imp-01-self_code_medium
commit:   06c4850 fix: narrow AISC coverage error handling
merge:    978fc15 merge: narrow AISC coverage errors
```

The agent narrowed the handler to expected failures:

```python
except (OSError, ValueError, TypeError, KeyError) as exc:
```

and added two regression tests: an expected `ValueError` is reported per spec;
an unexpected `RuntimeError` propagates.

```text
gate_id: 7f0d5be2bce36219
gate: 1/1 PASS
self-code diff: before 144, after 143
resolved: 1, introduced: 0, persisting: 143
decision: approve, gate_ok: True
note_id: c4584a45fe839be8
```

## 4. Measured result

```text
pre  scoreboard: commit b55ee8f, 1033 passed, 0 failed, AISC 23/25, benchmark 19/19
post scoreboard: commit 978fc15, 1035 passed, 0 failed, AISC 23/25, benchmark 19/19
delta: tests_passed +2
trend ok: True, alerts: []
```

- Full suite: **1035 passed**.
- AISC strict: **25 specs, 23 PASS, 2 expected answerability failures**.
- Self-improvement benchmark: **19/19**.
- No model weights changed; weights remain `0.8.0`.

## Honest boundary

- The audit is still lexical/AST based; a broad handler that is intentionally a
  catch-all remains a finding until a human marks or resolves it.
- The first run was correctly blocked by the gate, but the reason was a gate
  bug, not a bad patch. The bug is now fixed and covered.
- The gate ran one focused test file; the full suite was run after merge.
- The external agent is not sandboxed by Atom Zero; scope is checked after it
  runs.
