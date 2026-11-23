# atomzero 0.16.0 - opt-in runner and policy drift

Status: **published.** PyPI `0.16.0`, GitHub source/tag/release `v0.16.0`,
Hugging Face, ModelScope, dev.to and Medium are updated.
`scripts/verify_release.py --version 0.16.0` reports **0 failures**. LinkedIn
remains **blocked/unverified**. This release follows the 0.3 cadence after
`0.13.0`. It adds the opt-in agent runner and policy-metric
drift alerts, and closes a fourth measured cycle. No model or runtime weights
changed; the weights remain `0.8.0`.

## What is in this batch

1. **Opt-in agent runner.** `atomzero-agent-run prepare|run|verify` prepares an
   isolated git worktree from a bridge packet, optionally invokes an external
   command with `{worktree}`/`{packet}`/`{prompt}` placeholders, checks scope
   against the packet's allowed files, and runs the allowlisted gate inside the
   worktree. It never merges, pushes, tags or publishes.
2. **Policy-metric drift.** `atomzero-drift` reads `refusal_recall`,
   `refusal_precision` and `answerable_coverage` from the top level or the
   nested `aisc.policy.recommendation.point`; drops beyond 0.02 raise sorted
   alerts. The real signal -> transformer comparison now reports:
   `ece_increased`, `refusal_precision_decreased`,
   `answerable_coverage_decreased`.
3. **Gate isolation fix.** The gate no longer leaks the caller's main-checkout
   `PYTHONPATH` into the worktree test process.
4. **Fourth measured cycle.** Live drift signal -> planner action ->
   agent-bridge packet -> `atomzero-agent-run prepare` -> external Implementer
   -> runner gate (`gate_id 5134439e1f4e69df`) -> gated approval -> scoreboard
   `821/0` -> `839/0` with no trend alerts.

## Verified

- **839 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests); wheel, sdist and twine check
  pass.
- AISC strict: **21 specs, 19 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror).
- `benchmarks/self_improvement_regression.py`: **12/12 checks passed**.
- Four measured cycles have content-free gate archives and `gate_ok: true`
  decisions.
- `scripts/verify_release.py --version 0.16.0`: **0 failures**.
- Release commit `d687f32`, annotated tag `v0.16.0`.
- LinkedIn is blocked/unverified.
- No model weights changed; the weights remain `0.8.0`.

## Install

```powershell
pip install atomzero==0.16.0
atomzero-quickstart
```

## Honest limits

- The runner is opt-in and does not sandbox the external process; the harness
  must do that. It checks scope after the work, not before.
- Atom Zero still does not spawn coding agents itself; the external Implementer
  is part of the development harness.
- A human approves every action and can veto it.
- Scoreboard deltas count selected tests, not product value.
- The answerability gate remains off and CitationGate stays at `0.5`.
- LinkedIn publication remains blocked/unverified.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.16.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.16.0>
- Agent runner: <https://github.com/byjameson/atom-zero/blob/main/docs/AGENT_RUNNER.md>
- Fourth cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_4.md>
- Drift: <https://github.com/byjameson/atom-zero/blob/main/docs/DRIFT.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
