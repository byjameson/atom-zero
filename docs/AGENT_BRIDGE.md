# Agent handoff bridge

Atom Zero does **not** spawn coding agents, edit its own source or publish from
inside the local runtime. What it can do is turn a human-gated improvement
action into a bounded task packet for an **external** agent harness such as
OpenCode.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.agent_bridge --plan plan.json `
  --action-id imp-01-conflicts `
  --allowed-file src/atomzero/claims.py `
  --allowed-file tests/test_claims.py `
  --check pytest:tests/test_claims.py --json
```

## Packet

The packet is data only and carries:

- role, repository, action id/kind/title/goal;
- an explicit `allowed_files` list;
- allowlisted `pytest:` / `aisc:` / `manual:` checks;
- the action's acceptance gate;
- constraints: isolated worktree, only allowed files, no merge/push/tag/publish,
  return diff and test output, stop on secrets or out-of-scope files;
- `requires_human: True`, `auto_start: False`,
  `handoff: external_agent_harness_only`, `weights_changed: False`.

There is no function that starts a worktree, calls a model, runs a command or
writes source here. The external harness owns execution. Its result still has
to pass `atomzero-gate` and receive a gated decision before the ledger records
an approval.

## Current capability

| Question | Answer |
|---|---|
| Can Atom Zero run local tools? | Yes, via the constrained tool-plan runtime with confirmation. |
| Can it spawn external coding agents itself? | No. |
| Can it ask an agent harness to do bounded work? | Yes: it emits this packet; a human starts the harness. |
| Can it approve its own change? | No; approval requires a human and a passing gate. |
| Can it change weights automatically? | No. |

## Future path

A safe direction is an **agent-capable but bounded** loop: the planner emits a
packet, the external harness works in an isolated worktree, the gate runs, and a
human approves or rejects. The bridge makes the planner/harness boundary
explicit instead of pretending the local model is an autonomous engineer.

## Honest limits

- The packet is only as good as the action and the allowed-file list.
- `manual:` checks are not auto-passed by the external harness.
- A packet does not prove that an agent did the work or stayed in scope.
- The bridge does not sandbox the external harness; that is the harness's job.
- No model weights are touched.
