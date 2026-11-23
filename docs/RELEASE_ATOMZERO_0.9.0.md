# atomzero 0.9.0 - measured self-improvement

Status: **prepared locally; publication pending.** This batch makes the
human-gated improvement loop measurable. No model or runtime weights changed;
the weights remain `0.8.0`.

## What is in this batch

1. **Gated approval.** `record_gated_decision` rejects `approve` unless the
   evaluation gate passed every check. The ledger stores `gate_ok`,
   `gate_total`, `gate_passed` and `gate_method`; command output is not copied.
2. **Suggested checks.** Every improvement action carries `suggested_checks`
   in `kind:target` form, ready for `atomzero-gate --check`.
3. **Content-free scoreboard.** `atomzero-scoreboard` records commit, test/
   benchmark/AISC counts and a note hash, and reports latest/previous deltas.
4. **Measured regression signal.** The improvement plan treats a scoreboard
   regression as its highest-priority signal.
5. **Scoreboard HTTP.** `POST /v1/scoreboard/record`, `/summary` and `/verify`.
6. **Weekly review.** The scoreboard appears in the weekly review and Markdown
   export.
7. **Gate archive.** `atomzero-gate --archive` stores a content-free gate id,
   counts, method and per-check status/duration; output and commands are not
   stored.
8. **Trend report.** `atomzero-trend` combines decisions and runs and raises
   sorted alerts for test/benchmark/AISC regressions.
9. **Source-review links.** Weak-source and single-source actions point at the
   reliability/consensus workflow and focused checks.
10. **End-to-end benchmark.** `benchmarks/self_improvement_regression.py` now
    checks signal -> plan -> gate -> gated decision -> score -> trend.
    **12/12 checks pass.**

## Verified

- **783 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests); wheel, sdist and twine check
  pass.
- AISC strict: **18 specs, 16 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror).
- `benchmarks/self_improvement_regression.py`: **12/12 checks passed**.
- No model weights changed; the weights remain `0.8.0`.

## Install

```powershell
pip install atomzero==0.9.0
atomzero-quickstart
```

## Honest limits

- This is not autonomous self-modification; a human still approves every action.
- The scoreboard measures selected checks, not product value.
- The trend report can miss regressions in unmeasured areas.
- The gate archive is local and unsigned; it proves shape, not authorship.
- The answerability gate remains off and CitationGate stays at `0.5`.
- AISC expected failures remain visible and are not converted into passes.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.9.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.9.0>
- Scoreboard: <https://github.com/byjameson/atom-zero/blob/main/docs/SCOREBOARD.md>
- Trend: <https://github.com/byjameson/atom-zero/blob/main/docs/TREND.md>
- Evaluation gate: <https://github.com/byjameson/atom-zero/blob/main/docs/EVALUATION_GATE.md>
- Self-improvement: <https://github.com/byjameson/atom-zero/blob/main/docs/SELF_IMPROVEMENT.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
