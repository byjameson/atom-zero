# Opt-in agent runner

`atomzero-agent-bridge` produces a bounded packet. `atomzero-agent-run` is the
opt-in executor around it: it can prepare an isolated worktree, optionally call
a configured external agent command, check that only allowed files changed, and
run the packet's allowlisted gate. It never merges, pushes, tags or publishes.

```powershell
$env:PYTHONPATH="$PWD\src"

# Prepare only; no agent is started.
python -m atomzero.agent_runner prepare `
  --packet packet.json --opt-in --repo . --base main --json

# Prepare, run an external command, then verify.
python -m atomzero.agent_runner run --packet packet.json --opt-in `
  --repo . --runner opencode `
  --runner-arg run --runner-arg --dir --runner-arg "{worktree}" `
  --runner-arg "Read {packet} and implement only the allowed files" `
  --archive data/gate_archive.jsonl --json

# Verify an existing worktree.
python -m atomzero.agent_runner verify `
  --packet packet.json --worktree "C:\path\to\worktree" --json
```

## Safety boundary

- `prepare` and `run` require `--opt-in`; `verify` is read-only.
- The packet must be `requires_human: true`, `auto_start: false` and
  `handoff: external_agent_harness_only`.
- Allowed files must be repository-relative; `..` and absolute paths are
  rejected.
- Changed files are checked with `git status`; any file outside the allowed
  list fails the run.
- The gate is executed **inside the worktree**; a pytest target is resolved
  relative to that worktree.
- The runner does not merge/push/tag/publish and does not change weights.
- An external command is only executed when explicitly supplied. Without it,
  `prepare` performs no implementation.

## Runner command placeholders

Each argument may contain:

- `{worktree}` — isolated worktree path;
- `{packet}` — path to `.agent-task.json`;
- `{prompt}` — packet goal.

The command is executed with the worktree as cwd. It is not a shell string; the
arguments are passed directly. The runner records the exit code and a bounded
output tail, then verifies scope and the gate.

## Honest limits

- The runner does not sandbox the external process; the harness must do that.
- A passing gate proves only the selected checks passed.
- A packet cannot prove the agent stayed in scope before the runner checks.
- The runner does not merge or publish; a human/reviewer still decides.

## AISC

[`docs/aisc/atomzero-agent-runner.toml`](aisc/atomzero-agent-runner.toml)
models `PacketValidator -> WorktreePreparer -> GateVerifier -> RunnerReport`,
all verified, no irreversible components and `min_assurance = 0.90`.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_agent_runner.py -q
python -m aisc check docs/aisc/atomzero-agent-runner.toml --strict
```
