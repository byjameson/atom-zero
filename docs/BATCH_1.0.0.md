# 1.0.0 batch plan - parallel self-development

Release cadence: **one major version per release**. `0.19.0` is published;
`1.0.0` is the next boundary. Intermediate commits stay local; one push at the
boundary.

## Base state

- `0.19.0` published with the bounded self-improve loop.
- Post-release main: self-code audit, bounded proposals, planner signals,
  proposal-derived runner scope, sixth measured cycle
  (`gate_id cac82f022c879258`, `855/0 -> 857/0`).
- Main gate: 857 Atom Zero tests, 223 AISC tests, 23 strict specs
  (21 PASS, 2 expected answerability FAIL), benchmark 13/13.
- Weights remain `0.8.0`; answerability gate remains closed.

## Parallel Implementers

| Branch | Worktree | Scope | Gate |
| --- | --- | --- | --- |
| `feat/self-code-policy` | `%TEMP%\opencode\atom-zero-policy` | inline `# self-code: ignore` markers and `--target-path` selection | `tests/test_self_code.py`, `tests/test_self_improve.py` |
| `feat/planner-feedback` | `%TEMP%\opencode\atom-zero-feedback` | bounded, deterministic priority adjustment from the decision ledger | `tests/test_improvement.py`, `tests/test_self_improve.py` |
| `feat/scoreboard-collect` | `%TEMP%\opencode\atom-zero-score` | parse pytest/benchmark/AISC summaries into content-free scoreboard counts | `tests/test_scoreboard.py` |

Each Implementer:

- works only in its own worktree and assigned files;
- adds tests and docs;
- never merges, pushes, tags or publishes;
- returns commit hash, changed files, exact test output and limitations.

## Integration order

1. Merge `feat/planner-feedback` first (planner is the base layer).
2. Merge `feat/self-code-policy` (self-code and loop integration).
3. Merge `feat/scoreboard-collect` (measurement).
4. Run the full suite, AISC strict audit, AISC coverage and benchmark on the
   merged tree.
5. Add a measured cycle if the merged behavior produces a live signal.
6. Update `TEST_RESULTS.md`, `docs/HANDOFF.md`, `CHANGELOG.md` and
   `docs/LOOP_CYCLE_*.md`.
7. At the 1.0.0 boundary: bump version, build/twine, update release notes and
   all surfaces, run `scripts/verify_release.py --version 1.0.0`, then one push
   with tag and release.

## Acceptance

- No auto-apply, auto-merge, push, tag or publish from any loop.
- Human approval remains required.
- Every new feature has focused tests and a full-suite run.
- No model weights change.
- Blocked surfaces (LinkedIn) are reported as blocked, not green.
