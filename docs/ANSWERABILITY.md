# Answerability signal and refusal policy

**Date:** 2026-11-24

**Status:** experiment, not shipped as a default gate

## TL;DR

- Added an **independent optional answerability gate** in
  `src/atomzero/answerability.py`. It answers "does the retrieved context
  contain an answer?" and is separate from CitationGate ("is the citation
  valid?").
- The best local experiment combined 42 retrieval/answer-type features with a
  tiny SQuAD v2 QA confidence signal. Held-out AUC is **0.663**.
- Calibration now uses **prior-weighted isotonic regression**. The calibration
  split is 64.0% answerable while the held-out target split is 51.5%
  answerable; the isotonic fit is reweighted from the calibration prevalence to
  the measured target prevalence.
- The change reduces held-out ECE from **0.1299 to 0.0642** and Brier from
  **0.2443 to 0.2326** on the hybrid, with the ROC/ranking unchanged (AUC
  **0.6627**). Feature-only Brier / ECE moves from **0.2703 / 0.1407** to
  **0.2535 / 0.0519**.
- The requested refusal targets are **still not met**: refusal recall 0.5464,
  refusal precision 0.6310, answerable coverage 0.6990 at the
  calibration-selected threshold 0.4. No threshold in 0.1-0.9 meets all three
  targets together.
- The **Brier target is not met and is not claimed**: 0.2326 is above the 0.15
  upper bound. Prior weighting corrects a prevalence mismatch; it does not add
  discriminative signal, so the Brier floor is set by the class overlap in the
  SQuAD v2 stress sample.
- CitationGate remains the default gate at threshold **0.5**. The answerability
  gate defaults to **off**; it can be run in `shadow` or `enforce` mode only by
  an explicit environment setting.
- The AISC strict check was **not re-executed** in this environment because the
  external `ai-systems-algebra` root is unavailable. No 0.98 assurance claim is
  made.

## Report format

| field | value |
|---|---|
| Commit | `099ca53` (artifact run on `feat/answerability-calibration`) |
| Dataset / split | `rajpurkar/squad_v2`; train 1,165 / calibration 289 / held-out 200 |
| Signal features | 42 retrieval and answer-type features + 5 tiny QA features in the hybrid; CitationGate probabilities are not used |
| Calibration method | prior-weighted isotonic (`prior_weighted_isotonic_v1`); calibration prevalence 0.6401 -> target prevalence 0.5150 |
| n | calibration **289** (target >=200); held-out **200** |
| Brier / ECE | **0.2326 / 0.0642** (hybrid, held-out) |
| Out-of-fold Brier / ECE | **0.2346 / 0.1110** (hybrid, grouped 5-fold on the held-out rows) |
| Threshold | **0.4** selected by the unchanged internal calibration rule; AISC policy was not re-run |
| Refusal precision / recall | **0.6310 / 0.5464** |
| Answerable coverage | **0.6990** |
| Held-out results | 103 answerable / 97 no-answer; 53 no-answer refused; 72 answerable accepted |
| AISC strict | not re-executed; the generated spec still carries a Brier miss and a policy miss |
| Remaining limits | targets not met; Brier > 0.15; SQuAD v2 stress distribution; tiny QA model is an experiment artifact; no production pilot |

## Independent signal

The signal is not a renamed citation score. It uses:

- retrieval score distribution: top score, second score, margin, standard
  deviation, normalized entropy, count above 0.5/0.7;
- query coverage in the chunk, snippet and top sentences;
- question type (`who`, `when`, `where`, `how many`, yes/no, why, how);
- answer-type candidates: novel person/date/number/place spans;
- entity and number/year alignment between question and context;
- retrieval confidence and context/snippet length.

The hybrid adds five features from a local tiny SQuAD v2 QA model
(`mrm8488/bert-tiny-finetuned-squadv2`, 4.4M parameters): no-answer score,
best-span score, span margin, span length and answer probability. The QA model
is local and deterministic; it is not part of the default package.

The feature-only gate is wired into `grounded_answer` behind:

```text
ATOMZERO_ANSWERABILITY_GATE      off (default) | shadow | enforce
ATOMZERO_ANSWERABILITY_PROFILE   path to a calibrated JSON profile
ATOMZERO_ANSWERABILITY_THRESHOLD optional threshold override
```

CitationGate remains a separate gate and is evaluated independently. Explicit
document-inspection queries keep their narrow quote bypass and are not refused
by the answerability gate.

## Calibration method

`benchmarks/answerability_calibration.py` provides the shared helpers:

- `prior_weighted_isotonic(raw_scores, labels, target_prevalence)` fits
  `IsotonicRegression` with sample weights
  `target / calibration_prevalence` for answerable rows and
  `(1 - target) / (1 - calibration_prevalence)` for no-answer rows;
- `expected_calibration_error` and `brier_score` are the shared metric helpers;
- `grouped_out_of_fold_calibration` fits a selection-only map on the training
  folds (`GroupKFold` when context groups are available, otherwise seeded
  `KFold`), predicts the held-out fold and returns one out-of-fold probability
  per row. The fitted fold maps are discarded and never enter the production
  profile.

Profiles now store:

```json
{
  "calibration_method": "prior_weighted_isotonic_v1",
  "prevalence": {"calibration": 0.640138, "target": 0.515},
  "calibration_in_sample": {"split": "calibration", "...": "..."},
  "calibration_oof": {"split": "heldout", "method": "grouped_out_of_fold", "...": "..."},
  "calibration_metrics": {"split": "heldout", "method": "grouped_out_of_fold", "...": "..."}
}
```

The legacy `calibration_metrics` key is preserved but now reports the
out-of-fold estimate; it is no longer the optimistic in-sample fit. Every
observation row also persists `raw_score`, the pre-isotonic probability, so
future calibration experiments do not need QA inference.

### Before / after (held-out)

| signal | metric | unweighted | prior-weighted | delta |
|---|---|---:|---:|---:|
| hybrid | Brier | 0.2443 | **0.2326** | -0.0117 |
| hybrid | ECE | 0.1299 | **0.0642** | -0.0657 |
| hybrid | AUC | 0.6627 | **0.6627** | 0.0000 |
| feature-only | Brier | 0.2703 | **0.2535** | -0.0168 |
| feature-only | ECE | 0.1407 | **0.0519** | -0.0888 |

## Results

### Held-out target check

| metric | hybrid | target | pass |
|---|---:|---:|:---:|
| refusal recall | 0.5464 | >=0.90 | no |
| refusal precision | 0.6310 | >=0.90 | no |
| answerable coverage | 0.6990 | >=0.80 | no |
| answerability Brier | 0.2326 | <=0.15 | no |
| answerability ECE | 0.0642 | <=0.10 | yes |

Wilson 95% intervals:

- refusal precision: **[0.5242, 0.7263]**
- refusal recall: **[0.4474, 0.6418]**
- answerable coverage: **[0.6046, 0.7791]**

### Calibration report

| report | split | n | Brier | ECE | AUC |
|---|---|---:|---:|---:|---:|
| hybrid in-sample | calibration | 289 | 0.2061 | 0.1048 | 0.7140 |
| hybrid out-of-fold | held-out | 200 | 0.2346 | 0.1110 | 0.6290 |
| feature-only in-sample | calibration | 289 | 0.2286 | 0.1160 | 0.6382 |
| feature-only out-of-fold | held-out | 200 | 0.2573 | 0.0475 | 0.5436 |

The production map is the single map fitted on the calibration split. The
out-of-fold row is an honest reporting estimate that refits selection-only maps
inside the held-out sample; it is not used by the gate. The in-sample row is
optimistic because the production map saw those rows.

### Signal comparison

| signal | held-out AUC | Brier | ECE |
|---|---:|---:|---:|
| feature-only logistic | 0.5635 | 0.2535 | 0.0519 |
| DistilBERT cross-encoder | 0.5375 | 0.2762 | 0.1612 |
| **hybrid retrieval + tiny QA** | **0.6627** | **0.2326** | **0.0642** |

The cross-encoder was trained on the same split and is a stronger-model
baseline, not a shipped gate. It did not generalize: calibration AUC 0.622,
held-out AUC 0.538. The tiny QA model alone was an exploratory check
(AUC ~0.656); the hybrid is the only formalized best-effort artifact.

### Thresholds

Held-out hybrid policy:

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.3 | 0.971 | 0.842 | 0.165 | 0.580 |
| **0.4** | **0.699** | **0.631** | **0.546** | **0.625** |
| 0.5 | 0.680 | 0.629 | 0.577 | 0.630 |
| 0.6 | 0.544 | 0.591 | 0.701 | 0.620 |
| 0.7 | 0.252 | 0.522 | 0.866 | 0.550 |
| 0.8 | 0.233 | 0.518 | 0.876 | 0.545 |
| 0.9 | 0.010 | 0.485 | 0.990 | 0.485 |

The internal threshold rule is unchanged; it selects 0.4 on the calibration
split (no feasible threshold, misses all three targets). The threshold moved
from 0.6 to 0.4 because prior weighting changes the probability scale, not
because the selection rule changed. AISC policy on the held-out split was not
re-run.

The feature-only profile also regenerated, but its internal threshold rule
selected a degenerate 0.2 threshold with zero measured refusals at the selected
point. Its calibration numbers are useful; its policy point is not. The
feature-only gate remains off.

## AISC commands

AISC is an optional external checker. The run in this environment used
`--skip-aisc`, so the generated profile and report have no AISC policy block.
The checked-in spec files were regenerated from the new hybrid profile, but a
strict AISC execution was **not** performed.

```powershell
python -m aisc policy answerability_observations.csv `
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 `
  --min-refusal-recall 0.90 --min-refusal-precision 0.90 `
  --min-answerable-coverage 0.80

python -m aisc check answerability.toml --require-calibration --strict
```

The checked-in `benchmarks/results/answerability_*/aisc_*.txt` files were
produced by the earlier unweighted-calibration run. They were not regenerated
and are **not** evidence for the metrics in this document.

## Why the targets fail

- SQuAD v2 no-answer questions are constructed to share entities and
  vocabulary with the paragraph. Lexical overlap, retrieval margins and simple
  answer-type presence therefore do not separate them.
- Prior weighting fixes the calibration prevalence mismatch, not the
  discrimination. Held-out AUC stays at **0.663**, far below what a
  0.90/0.90/0.80 policy needs.
- At 90% no-answer recall, the hybrid also rejects too many answerable
  questions (coverage collapses). At useful coverage, refusal recall is too
  low.
- The train/calibration split is from SQuAD v2 train; the held-out sample is
  validation. The distribution shift is part of why the calibration does not
  transfer perfectly; the out-of-fold ECE (0.111) is a better estimate of that
  transfer than the single held-out ECE (0.064).
- The Brier score is **0.2326**, above the 0.15 upper bound. The class overlap
  sets a floor that prevalence reweighting cannot remove; this branch does not
  claim the Brier target can pass.
- Only SQuAD v2 was measured. This is not an operational pilot and cannot
  justify a production assurance claim.

## Decision

- **Do not enable the answerability gate by default.** Keep CitationGate's
  default profile `squad_v2_stress` and threshold `0.5`.
- The optional gate remains available in `shadow`/`enforce` mode for
  experiments. It is not a replacement for CitationGate and is not used as a
  citation-validity signal.
- The **hybrid profile is an experiment artifact**; the feature-only profile
  under `benchmarks/results/answerability_signal` is the one the optional
  feature gate can load. The hybrid profile is documented for evidence, not as
  a drop-in product profile.
- No thresholds, AISC targets, model weights or CitationGate defaults changed.
- **No 0.98 assurance claim.** The target metrics are not met.
- Next steps that could change the result: a larger local QA/NLI model, more
  target-distribution calibration data, per-domain recalibration, and a
  held-out operational pilot. A threshold sweep alone is not enough.

## Artifacts

Hybrid (best effort):

- [`benchmarks/results/answerability_hybrid/answerability_observations.csv`](../benchmarks/results/answerability_hybrid/answerability_observations.csv)
- [`benchmarks/results/answerability_hybrid/answerability_calibration.csv`](../benchmarks/results/answerability_hybrid/answerability_calibration.csv)
- [`benchmarks/results/answerability_hybrid/answerability_profile.json`](../benchmarks/results/answerability_hybrid/answerability_profile.json)
- [`benchmarks/results/answerability_hybrid/answerability_report.md`](../benchmarks/results/answerability_hybrid/answerability_report.md)
- [`benchmarks/results/answerability_hybrid/answerability.toml`](../benchmarks/results/answerability_hybrid/answerability.toml)

Feature-only gate:

- [`benchmarks/results/answerability_signal/answerability_profile.json`](../benchmarks/results/answerability_signal/answerability_profile.json)
- [`benchmarks/results/answerability_signal/answerability_observations.csv`](../benchmarks/results/answerability_signal/answerability_observations.csv)
- [`benchmarks/results/answerability_signal/answerability_calibration.csv`](../benchmarks/results/answerability_signal/answerability_calibration.csv)
- [`benchmarks/results/answerability_signal/answerability_report.md`](../benchmarks/results/answerability_signal/answerability_report.md)

Generated AISC spec (regenerated from the new hybrid profile):

- [`docs/aisc/atomzero-answerability-generated.toml`](aisc/atomzero-answerability-generated.toml)
- [`docs/aisc/atomzero-answerability.toml`](aisc/atomzero-answerability.toml)

DistilBERT baseline:

- [`benchmarks/results/answerability_transformer/answerability_profile.json`](../benchmarks/results/answerability_transformer/answerability_profile.json)
- [`benchmarks/results/answerability_transformer/answerability_report.md`](../benchmarks/results/answerability_transformer/answerability_report.md)

## Reproduce

```powershell
$env:PYTHONPATH="$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python benchmarks/answerability_signal.py --skip-aisc
python benchmarks/answerability_hybrid.py --skip-aisc
```

The hybrid run needs `torch`, `transformers` and the local tiny QA model; both
runs need `datasets`, `scikit-learn` and `numpy`. AISC is not required for the
calibration measurements above.
