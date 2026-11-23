# Self-code audit and bounded proposals

`atomzero-self-code` is the first branch that reviews Atom Zero's own source
code rather than its documents. It is deterministic, local and read-only.

It does **not** edit files, call a model, create a worktree or apply a patch.
It produces findings and, with `--propose`, one bounded proposal per finding
that a human or the opt-in runner can execute.

## What it scans

| Finding | Severity | Detection |
| --- | --- | --- |
| `dangerous_builtin` | high | real `eval(...)` / `exec(...)` calls in the Python AST |
| `shell_true` | high | real `shell=True` keyword in `subprocess` calls |
| `broad_exception` | medium | `except Exception` / `BaseException` handlers |
| `wall_clock_test` | low | `datetime.now()` / `date.today()` in `tests/` |
| `todo_marker` | low | TODO / FIXME / XXX in comments and docs |

The AST-based checks avoid text false positives: a docstring saying
`without eval()`, a regex containing `shell=True`, or `model.eval()` is not a
finding. Findings are capped per kind (`--max-per-kind`, default `20`), sorted
and assigned a stable `finding_id`.

## Usage

```powershell
atomzero-self-code --root . --json
atomzero-self-code --root . --propose --json
```

A proposal contains:

```json
{
  "proposal_id": "prop-...",
  "finding_id": "sc-...",
  "kind": "broad_exception",
  "severity": "medium",
  "path": "src/atomzero/agent_runner.py",
  "goal": "narrow the broad exception or re-raise after logging",
  "risk": "medium",
  "allowed_files": ["src/atomzero/agent_runner.py", "tests/test_agent_runner.py"],
  "checks": ["pytest:tests/test_agent_runner.py"],
  "requires_human": true,
  "auto_apply": false,
  "weights_changed": false
}
```

High-risk proposals deliberately have **no automated checks** and a manual
review goal. They are never auto-applied.

## Feeding the planner

`atomzero-improve` accepts the report directly:

```powershell
atomzero-self-code --root . --propose --json > self-code.json
atomzero-improve --self-code-json self-code.json --json
```

`collect_signals` turns high and medium findings into the
`self_code_high` and `self_code_medium` signals. Low findings are reported but
do not create planner actions.

## Feeding the bounded loop

The opt-in loop can run its own audit and derive packet scope from the top
proposal:

```powershell
atomzero-self-improve --self-code-root . --opt-in --approve `
  --runner "opencode run --dir {worktree} --auto <prompt>" `
  --repo . --base main --ledger data/improvement_ledger.jsonl
```

The loop still requires both `--opt-in` and `--approve`; `--veto` stops before
a worktree is created. It records a gated decision only after the gate passes.
It never merges, pushes, tags or publishes.

## Safety boundary

- Scope is limited to the proposal's `allowed_files`.
- The runner's committed-change check covers `git diff base...HEAD` plus the
  working tree.
- The gate runs inside the isolated worktree.
- High-risk arbitrary-code findings require manual review.
- Approval is human; `auto_apply` and `weights_changed` are always false.

## Tests and evidence

- `tests/test_self_code.py`: AST false positives, determinism, bounds,
  proposal shape and CLI JSON.
- `tests/test_improvement.py`: high/medium signal mapping and plan actions.
- `tests/test_self_improve.py`: proposal-derived scope and clean-tree
  `no_action`.
- `benchmarks/self_improvement_regression.py`: audit -> plan regression check.
- `docs/LOOP_CYCLE_6.md`: the first measured cycle fed by this audit; the
  external agent narrowed `agent_runner.verify`'s broad handler and added two
  regression tests (`gate_id cac82f022c879258`, `855/0 -> 857/0`).
