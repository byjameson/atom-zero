# Self-improvement scoreboard

A plan is not an improvement until the measured checks move. `atomzero-scoreboard`
records a small, content-free run summary and reports the latest result plus the
delta from the previous run.

```powershell
$env:PYTHONPATH="$PWD\src"

# Manual counts
python -m atomzero.scoreboard --file data/scoreboard.jsonl `
  --record --commit 514632e --tests-passed 755 --tests-failed 0 `
  --benchmark-passed 10 --benchmark-total 10 `
  --aisc-passed 14 --aisc-total 16 --json

# Extract counts from saved test, benchmark and audit output
python -m atomzero.scoreboard --file data/scoreboard.jsonl --record `
  --from-pytest pytest.txt --from-benchmark benchmark.json `
  --from-aisc aisc.txt --json

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

## Collect counts from output

`--from-pytest FILE`, `--from-benchmark FILE` and `--from-aisc FILE` read a
saved report, extract only the counts and merge them into the run recorded by
`--record`. The raw output is discarded and never written to the scoreboard.

- `--from-pytest` reads the last pytest summary line, for example
  `857 passed in 81.49s` or `2 failed, 855 passed in 12.3s`. Errors count as
  failures.
- `--from-benchmark` accepts the benchmark JSON report
  `{"passed": 13, "total": 13, "ok": true}` or the line
  `summary: 13/13 checks passed`.
- `--from-aisc` accepts `summary: 22 specs, 20 pass, 2 fail, 0 error` or
  `22 specs, 20 PASS, 2 FAIL`.

Use `-` as the file to read one source from stdin; only one source may use
stdin. Explicit `--tests-passed`, `--tests-failed`, `--benchmark-passed`,
`--benchmark-total`, `--aisc-passed` and `--aisc-total` override the parsed
values field by field.

When `--record` is used without `--commit`, the commit is derived from
`git rev-parse --short HEAD` in the current directory. Pass `--commit`
explicitly if the command runs outside a git checkout.

```powershell
# One source from stdin, explicit counts override the parsed ones
Get-Content pytest.txt | python -m atomzero.scoreboard `
  --file data/scoreboard.jsonl --record --from-pytest - `
  --tests-failed 0 --json
```

## HTTP

`POST /v1/scoreboard/record` accepts the same fields as `record_run` and writes
to `scoreboard_path` (default `data/scoreboard.jsonl`).
`POST /v1/scoreboard/summary` and `POST /v1/scoreboard/verify` are read-only.
The record route stores counts and a note hash only; the note text and test
output are never written.

```powershell
$body = @{
  commit = "514632e"; tests_passed = 755; tests_failed = 0
  benchmark_passed = 10; benchmark_total = 10
  aisc_passed = 14; aisc_total = 16
} | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/scoreboard/record `
  -ContentType "application/json" -Body $body
```

The weekly review includes the scoreboard summary (latest run and delta), and
its Markdown export has a `Scoreboard` section. This keeps a measured trend
next to tasks, gaps and improvement decisions.

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
- The output parsers only extract counts from saved reports. Malformed or
  unexpected output raises an error instead of guessing.

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
