# Self-improvement scoreboard

A plan is not an improvement until the measured checks move. `atomzero-scoreboard`
records a small, content-free run summary and reports the latest result plus the
delta from the previous run.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.scoreboard --file data/scoreboard.jsonl `
  --record --commit 514632e --tests-passed 755 --tests-failed 0 `
  --benchmark-passed 10 --benchmark-total 10 `
  --aisc-passed 14 --aisc-total 16 --json

python -m atomzero.scoreboard --file data/scoreboard.jsonl --summary
python -m atomzero.scoreboard --file data/scoreboard.jsonl --verify --json
```

## Record

Each JSONL record stores:

- `version`, `at` (UTC), `commit` (7-40 lowercase hex)
- `tests_passed`, `tests_failed`
- `benchmark_passed`, `benchmark_total`
- `aisc_passed`, `aisc_total`
- `note_id` (16-hex SHA-256 prefix or empty)

It never stores test output, prompts, queries, source text, logs or the note.
`verify` rejects those keys, invalid counts and malformed commits.

## Summary

- `latest` and `previous` records, ordered by timestamp and commit.
- `delta` for tests passed/failed, benchmark passed and AISC passed.
- `green` is true only when the latest run has zero test failures.
- `weights_changed` is always false.

## Honest limits

- A scoreboard records selected metrics, not overall quality or user value.
- Passing tests can coexist with a bad product decision; the human gate and
  evaluation gate still matter.
- AISC expected failures stay in the AISC counts and are not hidden.
- Records are local and unsigned; `verify` checks shape and content-free rules,
  not authorship.
- The scoreboard does not run tests; it records results produced by the
  evaluation gate or a human.

## AISC

[`docs/aisc/atomzero-scoreboard.toml`](aisc/atomzero-scoreboard.toml) models
`RunNormalizer -> ScoreboardSummarizer -> TrendReport`, all verified, no
irreversible components and `min_assurance = 0.90`.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_scoreboard.py -q
python -m aisc check docs/aisc/atomzero-scoreboard.toml --strict
```
