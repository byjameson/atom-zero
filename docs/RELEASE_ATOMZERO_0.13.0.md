# atomzero 0.13.0 - agent bridge and calibrated profiles

Status: **prepared locally; publication pending.** This is the first release
under the **0.3 minor cadence** after `0.10.0` (no 0.11/0.12). It adds the
bounded agent-handoff bridge, nested calibration-profile support and the first
agent-bridge-consumed improvement cycle. No model or runtime weights changed;
the weights remain `0.8.0`.

## What is in this batch

1. **Agent handoff bridge.** `atomzero-agent-bridge` turns a human-gated
   improvement action into a bounded task packet (allowed files, allowlisted
   checks, constraints). It does not spawn agents, edit source, run commands,
   merge, push or change weights; an external harness executes it.
2. **Nested calibration profiles.** `atomzero-drift` now reads top-level or
   `metrics` Brier/ECE, scalar or mapping `n`, and `selection.threshold`; a
   threshold change is a new alert. The real signal -> transformer profile
   comparison now works instead of failing with `profile.brier must be a
   number`.
3. **Third measured cycle (agent-bridge consumed).** A live drift profile
   failure was packaged by the bridge, implemented by an external Implementer
   in `fix/drift-profile-format`, passed the gate (`gate_id
   d38196191f97b81c`), received a gated approval, and moved the scoreboard from
   `800` to `813` passed with no trend alerts.
4. **Wall-clock test fix.** The brief tag-filter fixture now derives task dates
   from the same fixed date it evaluates, instead of `date.today()`.

## Verified

- **813 Atom Zero tests passed** at the release commit.
- **AISC 0.5.1: 223 tests passed** (+6 subtests); wheel, sdist and twine check
  pass.
- AISC strict: **20 specs, 18 PASS, 2 expected answerability failures**
  (`AtomZeroAnswerability` and its generated mirror).
- `benchmarks/self_improvement_regression.py`: **12/12 checks passed**.
- Three measured cycles have content-free gate archives and `gate_ok: true`
  decisions.
- No model weights changed; the weights remain `0.8.0`.

## Install

```powershell
pip install atomzero==0.13.0
atomzero-quickstart
```

## Honest limits

- Atom Zero still does not spawn coding agents itself; the bridge is data for an
  external harness and a human approves the handoff.
- The planner proposes; it does not prove that an action will help.
- The drift monitor compares profiles; it does not run the benchmark.
- Scoreboard deltas count selected tests/checks, not product value.
- The answerability gate remains off and CitationGate stays at `0.5`.
- LinkedIn publication remains blocked/unverified.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.13.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.13.0>
- Agent bridge: <https://github.com/byjameson/atom-zero/blob/main/docs/AGENT_BRIDGE.md>
- Third cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_3.md>
- Drift: <https://github.com/byjameson/atom-zero/blob/main/docs/DRIFT.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
