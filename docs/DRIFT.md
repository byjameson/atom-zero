# Calibration drift monitor

A model file can stay unchanged while its measured calibration drifts because
the evaluation set, retrieval, or threshold changed. `atomzero-drift` compares
a current profile with a stored baseline and reports the delta. It is a
reporting layer: no benchmark is run and no weight or gate is changed.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.drift --baseline baseline.json --profile current.json --json
```

## Profile fields

Required: `brier`, `ece` (numbers). They are read from the top level and, when
absent there, from a nested `metrics` mapping; top-level values win. Optional:
`n`, `threshold`, `commit`. `n` may be a scalar, a split mapping (preferring
`heldout`, then `calibration`, then `n`) or `metrics.n`. `threshold` is read
from the top level, then `selection.threshold`, then `metrics.threshold`.
The comparison reports:

- `delta.brier`, `delta.ece`, `delta.n`;
- `baseline`/`current` `brier`, `ece`, `n`, `threshold` and `commit`;
- `alerts`: `brier_increased`, `ece_increased`, `sample_size_dropped`,
  `threshold_changed`;
- `status`: `improved`, `stable` or `drifted`;
- `ok` is false when any alert exists;
- `weights_changed` is always false.

A resolved threshold that differs between the two profiles by more than
`1e-9` is reported as the `threshold_changed` alert.

Default tolerances are `+0.02` Brier, `+0.02` ECE and a 20% sample-size drop;
all are configurable.

## Honest limits

- A better Brier/ECE does not prove a better product; the evaluation set can
  change or be too small.
- A sample-size drop is not always bad (a cleaner set can be smaller), but it
  is surfaced as an alert rather than hidden.
- A changed threshold is an alert, not a verdict: the monitor does not decide
  which threshold is better for the product.
- The monitor does not run the answerability benchmark, calibrate a model or
  enable the answerability gate.
- Profiles are local files; the monitor does not verify authorship.

## AISC

[`docs/aisc/atomzero-drift.toml`](aisc/atomzero-drift.toml) models
`ProfileNormalizer -> DriftEvaluator -> DriftReport`, all verified, no
irreversible components and `min_assurance = 0.90`.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_drift.py -q
python -m aisc check docs/aisc/atomzero-drift.toml --strict
```
