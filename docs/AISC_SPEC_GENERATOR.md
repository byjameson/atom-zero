# AISC spec generator

`src/atomzero/aisc_spec.py` turns measured answerability evidence into AISC
TOML. It is an **evidence-to-TOML formatter**, not an evaluator: it reads the
existing profile, validates the required fields, and renders a canonical spec.

The module never imports AISC. AISC stays an optional external checker, so the
package remains installable and testable without it.

## Source of truth

The default profile is:

```text
benchmarks/results/answerability_hybrid/answerability_profile.json
```

The generator requires these fields and fails if any is missing:

```text
n, brier, ece, threshold, refusal_precision, refusal_recall,
answerable_coverage, commit, dataset_hash, calibration_date,
recalibration_due
```

Missing or malformed evidence is an error. No value is defaulted, sampled or
reconstructed. `collect_evidence()` also records the profile path (`source`,
relative to the repository when possible) and `generated_on`.

`generated_on` defaults to the profile's own `created` date, falling back to
`calibration_date`. This keeps the artifact reproducible. Use
`--generated-on YYYY-MM-DD` (or the corresponding keyword argument) when a
different generation stamp is required; the same value must then be supplied
to `--check`.

## Generated TOML

`build_spec()` emits exactly these root fields, in this order:

```text
name, version, description, min_assurance, max_context_tokens,
answerability_n, answerability_brier, answerability_ece,
refusal_precision, refusal_recall, answerable_coverage,
refusal_threshold, min_refusal_precision, min_refusal_recall,
min_answerable_coverage
```

It emits one `[[components]]` table named `AnswerabilitySignal` with
`kind = "model"`, `verified = true`, an evidence string (source, generation
date, sample size, commit, dataset hash and calibration dates), the guarantee
text and `context_tokens`.

Measured metrics use six decimal places. Minimum targets are fixed constants:

| field | value |
|---|---:|
| `min_refusal_precision` | 0.90 |
| `min_refusal_recall` | 0.90 |
| `min_answerable_coverage` | 0.80 |
| `min_assurance` | 0.80 |

Field order, formatting and LF line endings are fixed. The same evidence and
generation stamp always produce the same bytes.

The measured evidence is copied as-is. The generator **does not** lower a
target to make a failing metric pass. If the profile reports weak calibration
or a refusal-policy miss, the generated spec keeps that failure and AISC keeps
warning about it.

## Generated artifact

The checked-in artifact is:

```text
docs/aisc/atomzero-answerability-generated.toml
```

It is generated from the current hybrid profile. That profile reports:

| metric | measured | target | result |
|---|---:|---:|---|
| answerability Brier | 0.244274 | ≤0.15 | fail |
| answerability ECE | 0.129867 | ≤0.10 | fail |
| refusal precision | 0.631000 | ≥0.90 | fail |
| refusal recall | 0.546400 | ≥0.90 | fail |
| answerable coverage | 0.699000 | ≥0.80 | fail |

AISC therefore still reports `ANSWERABILITY_POOR` and
`REFUSAL_POLICY_MISMATCH`. That is the intended outcome until the signal and
refusal policy actually improve.

## Usage

From the repository root:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.aisc_spec
python -m atomzero.aisc_spec --check
python -m atomzero.aisc_spec --json
```

Options:

```text
--profile PATH       answerability profile JSON
--out PATH           generated TOML path
--check              compare only; exit 1 on drift or a missing file
--generated-on DATE  pin the YYYY-MM-DD generation stamp
--json               print {ok, changed, path} as JSON
```

`--check` never writes. It compares the canonical text with the file bytes, so
a line-ending change is also drift. Writes use a sibling temporary file,
`fsync` and `os.replace`, so readers never observe a partial spec.

Python API:

```python
from atomzero.aisc_spec import collect_evidence, build_spec, write_spec

evidence = collect_evidence(".", generated_on="2026-11-23")
text = build_spec(evidence)
result = write_spec(
    "docs/aisc/atomzero-answerability-generated.toml",
    evidence,
    check=True,
)
```

`write_spec()` returns `{"ok": bool, "changed": bool, "path": str}`. In check
mode `ok` means the file is current. In write mode `ok` means the write (or
no-op) succeeded.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_aisc_spec.py -q
python -m atomzero.aisc_spec --check
```

If AISC is available, the external check is still expected to fail under
`--strict`:

```powershell
python -m aisc check <repo>/docs/aisc/atomzero-answerability-generated.toml --strict
```

## Limits

- The generator does not re-run the benchmark or recompute metrics. It trusts
  the validated profile; stale evidence remains stale until the profile is
  regenerated.
- It does not implement AISC's checks. It only emits a spec that AISC can
  inspect.
- `assurance = 0.80` and `context_tokens = 2000` are architecture inputs, not
  measured claims. The measured calibration and refusal metrics are the
  empirical record.
- A profile located outside the repository makes `source` an absolute path in
  the artifact, which is less portable.
- The generated spec is evidence of a failing experiment, not a production
  pilot or a passing gate.
