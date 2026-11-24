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

## Curriculum adapter runs

The staged copy curriculum (`research/run_staged_curriculum.py`) can be one
step of the same opt-in loop. It trains the stage adapters, runs the final
mixed eval and behavior gate, and optionally promotes into the separate
curriculum registry. It runs only when `--opt-in` and
`--adapter-curriculum-command` are set; without the flag the code loop is
unchanged.

```powershell
atomzero-self-improve --signal signals.json --opt-in `
  --adapter-promotable 1 `
  --adapter-base checkpoints/transformer_behavior_50m_long `
  --adapter-curriculum-command "python research/run_staged_curriculum.py --data-dir {data_dir} --out-dir {out_dir} --base {base} --stages echo_kv,echo_json,two_kv,two_json,multi --seed 23 --final-eval {final_eval} --require-content --min-exact-match 0.25 --min-key-copy-rate 0.25 --result {result} --promote --active-file {active_file}" `
  --adapter-curriculum-result research/results/staged_curriculum_run.json `
  --adapter-curriculum-out checkpoints/staged_curriculum `
  --adapter-curriculum-active-file data/curriculum_active_adapter.json `
  --adapter-curriculum-final-eval data/staged_copy_v1/mixed_eval.jsonl `
  --json
```

Template placeholders are `{base}`, `{data_dir}`, `{out_dir}`, `{result}`,
`{active_file}` and `{final_eval}`. The loop-side floors default to 0.25 for
exact match and key copy; see
[STAGED_COPY_CURRICULUM.md](STAGED_COPY_CURRICULUM.md#autonomous-self-improve-integration)
for every flag and default.

The result contains a content-free block:

```json
"curriculum": {
  "status": "pass",
  "returncode": 0,
  "elapsed_seconds": 41.27,
  "gate_status": "pass",
  "exact_match": 0.4,
  "key_copy": 0.548,
  "promotion": {"requested": true, "promoted": true,
                "reason": "promoted"},
  "result_path": "research/results/staged_curriculum_run.json",
  "reasons": []
}
```

A nonzero exit, missing/invalid ledger, failing gate or missed floor sets the
loop status to `curriculum_failed`; no code change is attempted and no success
is recorded. Raw command output, rows and prompt text are never copied into
the loop result.

When the planner selects `adapter_candidate_ready`, this command replaces the
built-in adapter training step. On any other action it runs as a bounded
pre-step before the code change. A passing promotion writes only the
curriculum registry by default; it does not change `data/active_adapter.json`
or the product weights.

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
machine currently has no Docker/VM backend usable by the loop. `wsl.exe` is
present, but no Linux distribution is installed, so it is not a usable test
backend.

The first fully autonomous run is documented in `docs/LOOP_CYCLE_8.md`.

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
- Model training still happens only through an explicit curriculum command.
  A passing curriculum promotion writes the separate curriculum registry; it
  never changes the product registry or product weights unless an operator
  points it at `data/active_adapter.json`, which the curriculum docs warn
  against. The code loop itself improves code, not weights.
