# Hybrid answerability signal and refusal policy

Commit: `099ca539f1849fd576beb935b4f70f52bbb55edc`
Dataset / split: rajpurkar/squad_v2 validation (50 contexts / 200 questions)
Signal features: 42 retrieval/answer-type features plus 5 tiny SQuAD v2 QA features (no-answer score, best span, margin, span length, answer probability). CitationGate probabilities are not used.
Calibration method: prior_weighted_isotonic_v1 (prior-weighted isotonic map fitted on the context-grouped calibration split).
Prevalence: calibration=0.6401, target=0.5150
n: train=1165, calibration=289, held-out test=200
Brier / ECE: 0.2326 / 0.0642
Threshold: 0.40
Refusal precision / recall: 0.6310 / 0.5464
Answerable coverage: 0.6990

## Held-out results

| metric | value | target | pass |
|---|---:|---:|:---:|
| refusal_recall | 0.5464 | 0.90 | no |
| refusal_precision | 0.6310 | 0.90 | no |
| answerable_coverage | 0.6990 | 0.80 | no |
| answerability Brier | 0.2326 | 0.15 | no |
| answerability ECE | 0.0642 | 0.10 | yes |

## Calibration report

| report | split | n | Brier | ECE | AUC |
|---|---|---:|---:|---:|---:|
| in-sample | calibration | 289 | 0.2061 | 0.1048 | 0.7140 |
| out-of-fold | heldout | 200 | 0.2346 | 0.1110 | 0.6290 |

`calibration_metrics` in the profile is the out-of-fold report. The in-sample row is optimistic because the production map saw those rows.

## Threshold table (held-out)

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.1 | 0.990 | 0.929 | 0.134 | 0.575 |
| 0.2 | 0.990 | 0.929 | 0.134 | 0.575 |
| 0.3 | 0.971 | 0.842 | 0.165 | 0.580 |
| 0.4 | 0.699 | 0.631 | 0.546 | 0.625 |
| 0.5 | 0.680 | 0.629 | 0.577 | 0.630 |
| 0.6 | 0.544 | 0.591 | 0.701 | 0.620 |
| 0.7 | 0.252 | 0.522 | 0.866 | 0.550 |
| 0.8 | 0.233 | 0.518 | 0.876 | 0.545 |
| 0.9 | 0.010 | 0.485 | 0.990 | 0.485 |

## AISC policy

```text

```

## AISC strict check

```text

```

## Baselines

| signal | held-out AUC | Brier | ECE |
|---|---:|---:|---:|
| feature-only logistic | 0.5635 | 0.2535 | 0.0519 |
| DistilBERT cross-encoder | 0.5375 | 0.2762 | 0.1612 |
| **hybrid (this run)** | **0.6627** | **0.2326** | **0.0642** |

## Decision and limitations

- SQuAD v2 no-answer questions share entities and vocabulary with the source paragraph; this is a stress distribution, not a production pilot.
- The hybrid uses a tiny local QA model as a confidence feature; it is an experiment artifact, not a default product gate.
- The hybrid profile is not loadable by the feature-only optional gate; that gate uses the profile under answerability_signal.
- No threshold meets the 0.90 refusal-recall, 0.90 refusal-precision and 0.80 answerable-coverage targets together.
- CitationGate remains a separate gate and the default refusal threshold remains 0.5.
- Prior weighting corrects the calibration/target prevalence gap but does not add discriminative signal; the Brier floor is set by the class overlap, so the 0.15 Brier target is not claimed.
- Out-of-fold calibration metrics are a reporting estimate; the production map is fitted once on the calibration split and is not refitted on held-out rows.
- No 0.98 assurance claim is made.
