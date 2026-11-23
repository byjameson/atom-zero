# atomzero 0.19.0 - autonomous opt-in runner loop

Status: **published.** PyPI `0.19.0`, GitHub source/tag/release `v0.19.0`,
Hugging Face, ModelScope, dev.to and Medium are updated.
`scripts/verify_release.py --version 0.19.0` reports **0 failures**. LinkedIn
remains **blocked/unverified**. This release follows the 0.3 cadence after
`0.16.0`. It adds the bounded self-improve loop and
committed-change scope verification. No model or runtime weights changed; the
weights remain `0.8.0`.

## What is in this batch

1. **Bounded self-improve loop.** `atomzero-self-improve` composes:
   live signals -> planner -> bridge packet -> opt-in runner -> gate ->
   approve/veto -> gated decision -> scoreboard -> trend.
   `--opt-in` and `--approve` are both required; `--veto` returns before a
   worktree is created.
2. **Committed-change scope check.** `agent_runner.verify` now unions
   `git diff --name-only base...HEAD` with `git status --porcelain`, so an
   external agent cannot hide an out-of-scope file by committing it.
3. **Fifth measured cycle (actual external agent).** A live runner-scope
   signal produced a packet; the opt-in runner launched an actual nested
   `opencode run` process in an isolated worktree; the external agent changed
   only allowed files; the gate passed (`gate_id 7fb857e6552ae5cc`); approval
   recorded `gate_ok: true`; scoreboard `844/0` -> `846/0`.

## Verified

- **846 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests); wheel, sdist and twine check
  pass.
- AISC strict: **21 specs, 19 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror).
- `benchmarks/self_improvement_regression.py`: **12/12 checks passed**.
- Five measured cycles have content-free gate archives and `gate_ok: true`
  decisions.
- `scripts/verify_release.py --version 0.19.0`: **0 failures**.
- Release commit `a6ad17c`, annotated tag `v0.19.0`.
- LinkedIn is blocked/unverified.
- No model weights changed; the weights remain `0.8.0`.

## Install

```powershell
pip install atomzero==0.19.0
atomzero-quickstart
```

## Honest limits

- The runner is opt-in, and a human still approves every action. `--veto`
  stops the loop; Atom Zero does not autonomously accept its own changes.
- The external nested agent is not sandboxed by Atom Zero; the harness owns
  that boundary. Scope is checked after the agent runs.
- Scoreboard deltas count selected tests, not product value.
- The answerability gate remains off and CitationGate stays at `0.5`.
- LinkedIn publication remains blocked/unverified.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.19.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.19.0>
- Fifth cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_5.md>
- Agent runner: <https://github.com/byjameson/atom-zero/blob/main/docs/AGENT_RUNNER.md>
- Self-improvement: <https://github.com/byjameson/atom-zero/blob/main/docs/SELF_IMPROVEMENT.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
