# Self-code diff

`atomzero-self-code-diff` compares two `atomzero-self-code` reports. It answers
whether a change actually reduced findings, introduced new ones, or only moved
code around.

```powershell
atomzero-self-code --root . --propose --json > before.json
# apply a bounded patch through the opt-in runner
atomzero-self-code --root . --propose --json > after.json
atomzero-self-code-diff before.json after.json --json
```

## Matching

Findings are paired by `kind`, `path` and `message`, ignoring the line number.
A line shift is therefore reported as **persisting**, not as a fake
resolution plus introduction. Duplicate findings with the same identity are
paired in line order.

## Output

```json
{
  "resolved": [],
  "introduced": [],
  "persisting": [],
  "summary": {
    "before": 0,
    "after": 0,
    "resolved": 0,
    "introduced": 0,
    "persisting": 0,
    "new_high": 0,
    "new_medium": 0,
    "resolved_high": 0,
    "resolved_medium": 0
  },
  "ok": true,
  "auto_apply": false,
  "weights_changed": false,
  "method": "deterministic_self_code_diff_v1"
}
```

`ok` is true only when no new high or medium finding was introduced. The CLI
returns `0` when `ok` is true, `1` when it is false, and `2` for an input error.

## Use in a cycle

1. Audit the base worktree before the agent runs.
2. Audit the patched worktree after the gate.
3. Require `ok: true`; review `resolved` and `persisting`.
4. Record the counts and the gate decision; never auto-apply the patch because
   the diff is clean.

The diff is read-only and content-free: it keeps finding ids, kinds, severity,
paths, lines and messages, not source text, prompts or queries.

## Loop gate

When `atomzero-self-improve` runs with `--self-code-root`, it audits the base
repository before the agent and the isolated worktree after the agent. It then
compares the two reports:

- `ok: true` -> the gated decision can be recorded;
- `ok: false` (a new high or medium finding) -> the loop returns
  `self_code_regression` and records **no** decision, ledger entry or
  scoreboard delta.

This prevents a patch that passes its focused tests from adding a new
arbitrary-code or broad-exception risk elsewhere in the change. Both audits use
a high per-kind cap so a removed finding cannot push a pre-existing finding
past the default truncation and look like a new regression.
