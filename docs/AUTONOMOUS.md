# Autonomous loop and sandbox

The self-improvement loop now runs **without per-change human approval** by
default. The master `--opt-in` switch still enables execution, and an operator
can restore the old gate with `--require-approval`.

## Decision flow

```text
live signal / self-code audit
  -> planner
  -> bridge packet (scope + checks)
  -> external agent in a disposable worktree
  -> focused allowlisted gate
  -> self-code before/after diff
  -> full test suite in the sandbox
  -> local merge into main
  -> full test suite on main
  -> content-free decision ledger
  -> optional scoreboard update
```

If any stage fails, the loop stops and does **not** record a decision. A failed
post-merge test triggers `git reset --hard` back to the pre-merge commit.

Autonomous branches get a timestamp suffix and are cleaned up after a successful
merge, so repeated runs do not collide. A verified branch can be kept with
`--no-auto-merge` or by passing an explicit `--branch`.

The loop never pushes, tags or publishes. Those remain explicit release
actions.

## Commands

Autonomous (default):

```powershell
atomzero-self-improve --self-code-root . --target-path src/atomzero/module.py `
  --opt-in --runner "opencode run --dir {worktree} --auto <prompt>" `
  --repo . --base main --ledger data/improvement_ledger.jsonl `
  --archive data/gates.jsonl --json
```

Human approval for a high-risk run:

```powershell
atomzero-self-improve ... --opt-in --require-approval --approve
```

Verify only, no merge:

```powershell
atomzero-self-improve ... --opt-in --no-auto-merge
```

Inspect the sandbox backend:

```powershell
atomzero-self-improve --sandbox-info
```

## Sandbox reality on this machine

`--sandbox-info` currently reports:

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

This is **not** an OS-level VM. It provides:

- a disposable Git worktree for the agent;
- a sanitized environment with token-like variables removed;
- a temporary `HOME`/`TEMP`;
- a hard timeout and bounded output;
- no push, tag or publish.

It does **not** provide:

- host filesystem isolation;
- OS-level network isolation;
- protection from a determined child process escaping a simple timeout.

Docker, WSL or a VM can be added as a stronger backend when installed. This
machine currently has no Docker/VM backend usable by the loop. WSL is present
but the project test environment is not installed inside it.

## Why this is still bounded

- The packet's `allowed_files` is checked after the agent.
- The focused gate and the full suite must both pass.
- The self-code diff blocks a new high/medium finding.
- The repo must be clean before merging.
- A failed post-merge suite rolls back automatically.
- `--veto` still stops the loop.
- No credentials are passed to the sandboxed test process.
- No push, tag or publish is possible from the loop.

## Limits

- The external agent is not sandboxed by Atom Zero itself; the harness owns its
  process boundary.
- The worktree sandbox is a blast-radius reduction, not a security boundary.
- Full test runs are expensive; the timeout defaults to 900 seconds.
- Auto-merge changes local `main`; it does not update any remote surface.
- Model weights still change only through an offline, separately approved
  training step. The loop improves code, not weights.
