# Answerability signal and refusal policy

Commit: `099ca539f1849fd576beb935b4f70f52bbb55edc`
Dataset / split: rajpurkar/squad_v2 validation (50 contexts / 200 questions)
Signal features: 42 local retrieval and answer-type features; CitationGate probabilities are not used.
Calibration method: prior_weighted_isotonic_v1 (prior-weighted isotonic map fitted on the context-grouped calibration split).
Prevalence: calibration=0.6401, target=0.5150
n: train=1165, calibration=289, held-out test=200
Brier / ECE: 0.2535 / 0.0519
Threshold: 0.20
Refusal precision / recall: 0.0000 / 0.0000
Answerable coverage: 0.9903

## Held-out results

| metric | value | target | pass |
|---|---:|---:|:---:|
| refusal_recall | 0.0000 | 0.90 | no |
| refusal_precision | 0.0000 | 0.90 | no |
| answerable_coverage | 0.9903 | 0.80 | yes |
| answerability Brier | 0.2535 | 0.15 | no |
| answerability ECE | 0.0519 | 0.10 | yes |

Wilson 95% intervals:

- refusal precision: [0.0, 0.7935]
- refusal recall: [0.0, 0.0381]
- answerable coverage: [0.947, 0.9983]

## Calibration report

| report | split | n | Brier | ECE | AUC |
|---|---|---:|---:|---:|---:|
| in-sample | calibration | 289 | 0.2286 | 0.1160 | 0.6382 |
| out-of-fold | heldout | 200 | 0.2573 | 0.0475 | 0.5436 |

`calibration_metrics` in the profile is the out-of-fold report. The in-sample row is optimistic because the production map saw those rows.

## Threshold table (held-out)

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.1 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.2 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.3 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.4 | 0.913 | 0.526 | 0.103 | 0.520 |
| 0.5 | 0.670 | 0.553 | 0.433 | 0.555 |
| 0.6 | 0.058 | 0.492 | 0.969 | 0.500 |
| 0.7 | 0.049 | 0.490 | 0.969 | 0.495 |
| 0.8 | 0.049 | 0.490 | 0.969 | 0.495 |
| 0.9 | 0.000 | 0.485 | 1.000 | 0.485 |

## AISC policy

```text

```

## AISC strict check

```text

```

## Decision and limitations

- SQuAD v2 no-answer questions share entities and vocabulary with the source paragraph; this is a stress distribution, not a production pilot.
- The signal is retrieval-feature based; no NLI or QA model is used.
- The 0.90 refusal target is not met; CitationGate remains the only default refusal gate.
- The answerability gate is implemented behind an environment flag and defaults to off, so this experiment does not change product behavior.
- Prior weighting corrects the calibration/target prevalence gap but does not add discriminative signal; the Brier floor is set by the class overlap, so the 0.15 Brier target is not claimed.
- Out-of-fold calibration metrics are a reporting estimate; the production map is fitted once on the calibration split and is not refitted on held-out rows.
- No 0.98 assurance claim is made.
