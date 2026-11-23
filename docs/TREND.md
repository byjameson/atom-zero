# Improvement trend report

`atomzero-trend` combines the content-free decision ledger and the measured
scoreboard. It answers a simple question: are we approving work and are the
numbers moving in the right direction?

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.trend `
  --ledger data/improvement_ledger.jsonl `
  --scoreboard data/scoreboard.jsonl --json
```

## Report

- decision counts by approve/reject/defer and action kind;
- latest measured run plus test/benchmark/AISC deltas;
- alerts: `tests_failed`, `tests_passed_decreased`,
  `benchmark_passed_decreased`, `aisc_passed_decreased`,
  `benchmark_incomplete`;
- `ok` is true only when there are no alerts;
- `weights_changed` is always false.

The report reads only content-free ledgers and never copies test output, gate
output, prompts, queries or source text.

## Honest limits

- Alerts describe selected metrics, not product quality or user value.
- A positive delta can hide a regression in an unmeasured area.
- The report is local and unsigned; it does not prove that a decision was wise.
- AISC expected failures remain visible and are not counted as a green score.

## AISC

[`docs/aisc/atomzero-trend.toml`](aisc/atomzero-trend.toml) models
`TrendInputs -> AlertEvaluator -> TrendReport`, all verified, no irreversible
components and `min_assurance = 0.90`.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_trend.py -q
python -m aisc check docs/aisc/atomzero-trend.toml --strict
```
