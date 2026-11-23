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

Optional policy metrics are `refusal_recall`, `refusal_precision` and
`answerable_coverage`. Each is read from the top level and, when absent there,
from `aisc.policy.recommendation.point` (the path used by the transformer
profile). AISC names the refusal-recall ratio `no_answer_refusal`, so that
field is accepted as the nested form of `refusal_recall`. Resolution is per
metric, and top-level values win. Missing or non-numeric values are reported
as `null`; they do not raise or alert.

The comparison reports:

- `delta.brier`, `delta.ece`, `delta.n`;
- `baseline`/`current` `brier`, `ece`, `n`, `threshold` and `commit`;
- `policy.baseline`, `policy.current` and `policy.delta` with the resolved
  `refusal_recall`, `refusal_precision` and `answerable_coverage` values;
- `alerts`: `brier_increased`, `ece_increased`, `sample_size_dropped`,
  `threshold_changed`, `refusal_recall_decreased`,
  `refusal_precision_decreased`, `answerable_coverage_decreased`;
- `status`: `improved`, `stable` or `drifted`;
- `ok` is false when any alert exists;
- `weights_changed` is always false.

A resolved threshold that differs between the two profiles by more than
`1e-9` is reported as the `threshold_changed` alert. A policy metric alerts
only when both profiles resolve it and the current value is more than the
configured tolerance below the baseline.

Default tolerances are `+0.02` Brier, `+0.02` ECE, a 20% sample-size drop and
a `0.02` drop for each policy metric; all are configurable. The CLI exposes
the policy tolerances as `--max-refusal-recall-drop`,
`--max-refusal-precision-drop` and `--max-coverage-drop`.

## Honest limits

- A better Brier/ECE does not prove a better product; the evaluation set can
  change or be too small.
- A sample-size drop is not always bad (a cleaner set can be smaller), but it
  is surfaced as an alert rather than hidden.
- A changed threshold is an alert, not a verdict: the monitor does not decide
  which threshold is better for the product.
- Policy metrics can be resolved from different threshold contexts when the
  profiles store them in different places; a policy-decrease alert is a
  signal to inspect, not a verdict.
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
