# Evaluation gate

A proposed improvement is not accepted because it sounds plausible. It is
accepted only when an explicit, allowlisted check set passes.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.evaluation `
  --check pytest:tests/test_claims.py `
  --check pytest:tests/test_consensus.py `
  --json

python -m atomzero.evaluation `
  --check aisc:docs/aisc/atomzero-claims.toml `
  --aisc-path "$env:AISC_PATH"
```

## Allowlisted checks

| Kind | Command | Purpose |
|---|---|---|
| `pytest` | `python -m pytest <target> -q` | focused regression gate |
| `aisc` | `python -m aisc check <absolute target> --strict` in the AISC checkout | architecture/spec gate |
| `manual` | never auto-executed | a human must confirm |

There is no generic shell kind. The module never edits files, never applies a
plan and never changes weights. A timeout is a failure, not a pass.

## Archive

`--archive PATH` appends a content-free gate entry (JSONL) with a deterministic
`gate_id`, counts, method and per-check name/kind/status/duration. It never
stores check output, commands, queries or source text. The archive is verified
before an atomic rewrite and can be read with `load_archive`.

```powershell
python -m atomzero.evaluation `
  --check pytest:tests/test_claims.py `
  --archive data/gate_archive.jsonl --json
```

## Report

```json
{
  "checks": [
    {"name": "...", "kind": "pytest", "target": "...", "ok": true,
     "status": "pass", "exit_code": 0, "duration_ms": 1187,
     "output": "..."}
  ],
  "summary": {"total": 2, "passed": 2, "failed": 0, "manual_required": 0},
  "ok": true,
  "auto_apply": false,
  "weights_changed": false,
  "method": "allowlisted_evaluation_gate_v1"
}
```

`manual_required` checks count as not passed. A human can decide separately,
but the gate does not pretend the decision happened.

## Honest limits

- A passing gate proves only that the selected checks passed. It does not prove
  that the change is valuable, complete or safe outside those checks.
- The pytest target is resolved relative to the repository root; the AISC target
  is resolved relative to the repository root and executed in the AISC checkout.
- The gate trusts the local Python environment and the selected test files.
- Output is truncated to the last 4,000 characters for the report.
- It is a review gate, not a sandbox. Do not put untrusted check targets in a
  plan that a remote party controls.

## AISC

[`docs/aisc/atomzero-evaluation.toml`](aisc/atomzero-evaluation.toml) models
`EvaluationPlan -> AllowlistedRunner -> GateReport`. The runner is reversible
and there are no irreversible components; `min_assurance = 0.90`.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_evaluation.py -q
python -m aisc check docs/aisc/atomzero-evaluation.toml --strict
```
