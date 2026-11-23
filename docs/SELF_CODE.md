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

## Inline suppression markers

A finding can be ignored with a marker on its own line or on the line
immediately above it. The marker is a comment and the match is
case-insensitive:

```python
subprocess.run(command, shell=True)  # self-code: ignore

try:
    run_step()
# self-code: ignore broad_exception,shell_true
except Exception:
    run_step()

try:
    run_step()
except Exception:  # self-code: ignore broad_exception  TODO later
    run_step()
```

- `# self-code: ignore` ignores every kind on the guarded line.
- `# self-code: ignore kind1,kind2` ignores only the listed kinds. Other
  findings on the same line, such as a task marker in the same comment, still
  appear.
- The rule applies to AST findings and to task markers in Python, Markdown and
  the other scanned text formats.
- Suppressed findings do not consume the `--max-per-kind` budget.

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

### Targeting one file

Pass `--target-path` with `--self-code-root` (or with a `self_code` signal
report) to work on one exact repo-relative path:

```powershell
atomzero-self-improve --self-code-root . `
  --target-path src/atomzero/agent_runner.py `
  --opt-in --approve `
  --runner "opencode run --dir {worktree} --auto <prompt>" `
  --repo . --base main --ledger data/improvement_ledger.jsonl
```

- Backslashes are normalized and a leading `./` is stripped before matching.
- The report is reduced to the findings and proposals for that path. The
  planner still produces the action for the target's severity
  (`self_code_high` or `self_code_medium`) and the packet takes
  `allowed_files` and `checks` from the matching proposal.
- A missing path, a low-severity-only target, or a report without a matching
  proposal raises a clear input error instead of silently falling back to
  another file.
- Target selection does not bypass `--opt-in`, `--approve`, `--veto`, scope
  checks or the evaluation gate.

## Safety boundary

- Scope is limited to the proposal's `allowed_files`.
- The runner's committed-change check covers `git diff base...HEAD` plus the
  working tree.
- The gate runs inside the isolated worktree.
- High-risk arbitrary-code findings require manual review.
- Approval is human; `auto_apply` and `weights_changed` are always false.

## Tests and evidence

- `tests/test_self_code.py`: AST false positives, determinism, bounds,
  suppression markers, proposal shape and CLI JSON.
- `tests/test_improvement.py`: high/medium signal mapping and plan actions.
- `tests/test_self_improve.py`: proposal-derived scope, target selection,
  clean-tree `no_action` and CLI pass-through.
- `benchmarks/self_improvement_regression.py`: audit -> plan regression check.
- `docs/LOOP_CYCLE_6.md`: the first measured cycle fed by this audit; the
  external agent narrowed `agent_runner.verify`'s broad handler and added two
  regression tests (`gate_id cac82f022c879258`, `855/0 -> 857/0`).
