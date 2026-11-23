# atomzero 0.10.0 - calibrated loop

Status: **published.** PyPI `0.10.0`, GitHub source/tag/release `v0.10.0`,
Hugging Face, ModelScope, dev.to and Medium are updated.
`scripts/verify_release.py --version 0.10.0` reports **0 failures**. LinkedIn
accepted the publish action but the profile feed still showed the older post;
treat LinkedIn as **blocked/unverified**, not green. This batch adds
calibration drift monitoring and closes two measured improvement cycles. No
model or runtime weights changed; the weights remain `0.8.0`.

## What is in this batch

1. **Calibration drift monitor.** `atomzero-drift` compares a current profile
   with a baseline, reports Brier/ECE deltas, and raises `brier_increased`,
   `ece_increased` and `sample_size_dropped` alerts with configurable
   tolerances. Drift is also a prioritized improvement-plan signal.
2. **Measured cycle 1 (HTTP stability).** A reproducible Windows `WinError
   10053` flake was fixed by draining bounded oversized bodies before `413`.
   Focused gate `gate_id ab5fe9b0fa0f48bc`; approval `gate_ok: true`;
   scoreboard `792 passed / 1 failed` -> `793 passed / 0 failed`.
3. **Measured cycle 2 (conflict noise).** A live claim audit showed date
   digits double-counted as numeric conflicts. The fix removes date/code digits
   before deciding whether an independent numeric mismatch remains. Gate
   `gate_id ac1eb8a7ee8b5271`; approval `gate_ok: true`; the same audit now
   reports `date_mismatch` only. Scoreboard `793` -> `794` passed with no trend
   alerts.

## Verified

- **794 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests); wheel, sdist and twine check
  pass.
- AISC strict: **19 specs, 17 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror).
- `benchmarks/self_improvement_regression.py`: **12/12 checks passed**.
- Both closed cycles have a content-free gate archive and decision-ledger
  record.
- `scripts/verify_release.py --version 0.10.0`: **0 failures**.
- The read-only Release agent returned **release gate: PASS** after the stale
  Medium PyPI link and `783` count were corrected to 0.10.0 / 794.
- Release commit `ceb26a0`, annotated tag `v0.10.0`.
- LinkedIn is blocked/unverified.
- No model weights changed; the weights remain `0.8.0`.

## Install

```powershell
pip install atomzero==0.10.0
atomzero-quickstart
```

## Honest limits

- The loops are human-gated: the planner proposes, a human approves.
- The scoreboard measures selected tests, not product value.
- The drift monitor compares profiles; it does not run the benchmark itself.
- The answerability gate remains off and CitationGate stays at `0.5`.
- AISC expected failures remain visible and are not converted into passes.
- LinkedIn publication remains blocked/unverified on this machine.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.10.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.10.0>
- First cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE.md>
- Second cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_2.md>
- Drift: <https://github.com/byjameson/atom-zero/blob/main/docs/DRIFT.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
