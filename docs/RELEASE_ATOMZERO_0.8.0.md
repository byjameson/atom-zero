# atomzero 0.8.0 - human-gated self-improvement loop

Status: **published.** PyPI `0.8.0`, GitHub source/tag/release `v0.8.0`,
Hugging Face, ModelScope, dev.to, Medium and LinkedIn are updated.
`scripts/verify_release.py --version 0.8.0` reports **0 failures**. This batch
adds the planning layer for a self-improving system without autonomous weight
updates. No model or runtime weights changed; the weights remain `0.8.0`.

## What is in this batch

1. **Improvement loop.** `atomzero-improve` collects local weakness signals
   (unsupported claims, conflicts, gaps, single-source clusters, stale/undated
   sources, weak sources, pending feedback, AISC coverage gaps), prioritizes
   actions and marks every action `requires_human: True`.
2. **Content-free decision ledger.** `record_decision` stores only action id,
   kind, decision, timestamp and a note hash; text/query/answer/chunk/session
   keys are rejected. `summarize_ledger` reports counts by decision and kind.
3. **HTTP.** `POST /v1/improvement/plan` and
   `POST /v1/improvement/decision` expose the loop locally. The ledger path is
   `data/improvement_ledger.jsonl` by default and is verified before/after an
   append.
4. **Proactive reminders.** A pending human-gated plan can start a local
   check-in with `show improvements` and `run evaluation gate`; only the action
   count and top kind are stored.
5. **Evaluation gate.** `atomzero-gate` runs an allowlist of `pytest`, `aisc`
   and `manual` checks. There is no generic shell, timeouts fail, manual checks
   are not auto-passed, and `auto_apply`/`weights_changed` are always false.
6. **Weekly review.** The decision-ledger summary is included in the weekly
   review and its Markdown export.
7. **Self-improvement benchmark.**
   `benchmarks/self_improvement_regression.py` checks the plan, human gate,
   ledger, proactive reminder, evaluation allowlist, AISC signal and upstream
   claim/temporal audits. **10/10 checks pass.**

## Verified

- **755 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests); wheel, sdist and twine check
  pass.
- AISC strict: **16 specs, 14 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror).
- `benchmarks/self_improvement_regression.py`: **10/10 checks passed**.
- `scripts/verify_release.py --version 0.8.0`: **0 failures**.
- Release commit `514632e`, annotated tag `v0.8.0`.
- No model weights changed; the weights remain `0.8.0`.

## Install

```powershell
pip install atomzero==0.8.0
atomzero-quickstart
```

## Honest limits

- This is not autonomous self-modification. A human must approve every action.
- The planner ranks lexical/count signals; it cannot prove an action will help.
- The evaluation gate proves only that selected checks passed.
- The decision ledger is local and unsigned; `verify_ledger` checks shape and
  content-free rules, not authorship.
- The answerability gate remains off and CitationGate stays at `0.5`.
- AISC coverage is not assurance and does not lower failing thresholds.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.8.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.8.0>
- Self-improvement architecture: <https://github.com/byjameson/atom-zero/blob/main/docs/SELF_IMPROVEMENT.md>
- Improvement loop: <https://github.com/byjameson/atom-zero/blob/main/docs/IMPROVEMENT_LOOP.md>
- Evaluation gate: <https://github.com/byjameson/atom-zero/blob/main/docs/EVALUATION_GATE.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
