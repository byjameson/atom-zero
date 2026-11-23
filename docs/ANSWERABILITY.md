# Answerability signal and refusal policy

**Date:** 2026-11-23  
**Status:** experiment, not shipped as a default gate

## TL;DR

- Added an **independent optional answerability gate** in
  `src/atomzero/answerability.py`. It answers “does the retrieved context
  contain an answer?” and is separate from CitationGate (“is the citation
  valid?”).
- The best local experiment combined 42 retrieval/answer-type features with a
  tiny SQuAD v2 QA confidence signal. It reached held-out AUC **0.663**,
  Brier **0.244**, ECE **0.130**.
- The requested targets are **not met**: refusal recall 0.546, refusal
  precision 0.631, answerable coverage 0.699 at the calibration-selected
  threshold 0.6. No threshold in 0.1–0.9 meets all three targets together.
- CitationGate remains the default gate at threshold **0.5**. The answerability
  gate defaults to **off**; it can be run in `shadow` or `enforce` mode only by
  an explicit environment setting.
- AISC strict check: **FAIL** with 2 warnings (`ANSWERABILITY_POOR`,
  `REFUSAL_POLICY_MISMATCH`). No 0.98 assurance claim is made.

## Report format

| field | value |
|---|---|
| Commit | `b4b68b1` (code commit for the optional gate and benchmark scripts) |
| Dataset / split | `rajpurkar/squad_v2`; train 1,165 / calibration 289 / held-out 200 |
| Signal features | 42 retrieval and answer-type features + 5 tiny QA features in the hybrid; CitationGate probabilities are not used |
| Calibration method | logistic model + isotonic calibration; context-grouped train/calibration split |
| n | calibration **289** (target ≥200); held-out **200** |
| Brier / ECE | **0.2443 / 0.1299** (hybrid, held-out) |
| Threshold | **0.6** selected by AISC policy on the calibration split |
| Refusal precision / recall | **0.6310 / 0.5464** |
| Answerable coverage | **0.6990** |
| Held-out results | 103 answerable / 97 no-answer; 68 no-answer refused; 56 answerable accepted |
| AISC strict | **FAIL**: `ANSWERABILITY_POOR`, `REFUSAL_POLICY_MISMATCH` |
| Remaining limits | targets not met; SQuAD v2 stress distribution; tiny QA model is an experiment artifact; no production pilot |

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

## Results

### Held-out target check

| metric | hybrid | target | pass |
|---|---:|---:|:---:|
| refusal recall | 0.5464 | ≥0.90 | no |
| refusal precision | 0.6310 | ≥0.90 | no |
| answerable coverage | 0.6990 | ≥0.80 | no |
| answerability Brier | 0.2443 | ≤0.15 | no |
| answerability ECE | 0.1299 | ≤0.10 | no |

Wilson 95% intervals:

- refusal precision: **[0.5242, 0.7263]**
- refusal recall: **[0.4474, 0.6418]**
- answerable coverage: **[0.6046, 0.7791]**

### Signal comparison

| signal | held-out AUC | Brier | ECE |
|---|---:|---:|---:|
| feature-only logistic | 0.5635 | 0.2703 | 0.1407 |
| DistilBERT cross-encoder | 0.5375 | 0.2762 | 0.1612 |
| **hybrid retrieval + tiny QA** | **0.6627** | **0.2443** | **0.1299** |

The cross-encoder was trained on the same split and is a stronger-model
baseline, not a shipped gate. It did not generalize: calibration AUC 0.622,
held-out AUC 0.538. The tiny QA model alone was an exploratory check
(AUC ≈0.656); the hybrid is the only formalized best-effort artifact.

### Thresholds

Held-out hybrid policy:

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.5 | 0.913 | 0.526 | 0.103 | 0.520 |
| **0.6** | **0.699** | **0.631** | **0.546** | **0.625** |
| 0.7 | 0.544 | 0.591 | 0.701 | 0.620 |
| 0.8 | 0.243 | 0.522 | 0.876 | 0.550 |
| 0.9 | 0.233 | 0.518 | 0.876 | 0.545 |

AISC policy on the **calibration** split selected 0.6 (no feasible threshold;
misses all three targets). AISC policy on the **held-out** split reports the
nearest threshold 0.7: refusal recall 0.701, refusal precision 0.591,
answerable coverage 0.544, overall 0.620; misses all three targets.

## AISC commands

Policy:

```powershell
python -m aisc policy answerability_observations.csv `
  --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 `
  --min-refusal-recall 0.90 --min-refusal-precision 0.90 `
  --min-answerable-coverage 0.80
```

Output summary:

```text
Hedefleri karşılayan eşik yok. En yakın: 0.7
refusal recall=0.701; refusal precision=0.591;
answerable coverage=0.544; overall=0.620
Eksik: refusal_recall, refusal_precision, answerable_coverage
```

Strict check:

```powershell
python -m aisc check answerability.toml --require-calibration --strict
```

Output summary:

```text
WARNING ANSWERABILITY_POOR:
  Brier=0.244, ECE=0.130 (upper Brier 0.15, ECE 0.10)
WARNING REFUSAL_POLICY_MISMATCH:
  refusal recall 0.546 < 0.90
  refusal precision 0.631 < 0.90
  answerable coverage 0.699 < 0.80
Sonuç: FAIL (0 kritik, 2 uyarı)
```

## Why the targets fail

- SQuAD v2 no-answer questions are constructed to share entities and
  vocabulary with the paragraph. Lexical overlap, retrieval margins and simple
  answer-type presence therefore do not separate them.
- The hybrid’s calibration is acceptable only in a narrow range; its held-out
  AUC is 0.663, far below what a 0.90/0.90/0.80 policy needs.
- At 90% no-answer recall, the hybrid also rejects too many answerable
  questions (coverage collapses). At useful coverage, refusal recall is too
  low.
- The train/calibration split is from SQuAD v2 train; the held-out sample is
  validation. The distribution shift is part of why the calibration does not
  transfer.
- Only SQuAD v2 was measured. This is not an operational pilot and cannot
  justify a production assurance claim.

## Decision

- **Do not enable the answerability gate by default.** Keep CitationGate’s
  default profile `squad_v2_stress` and threshold `0.5`.
- The optional gate remains available in `shadow`/`enforce` mode for
  experiments. It is not a replacement for CitationGate and is not used as a
  citation-validity signal.
- The **hybrid profile is an experiment artifact**; the feature-only profile
  under `benchmarks/results/answerability_signal` is the one the optional
  feature gate can load. The hybrid profile is documented for evidence, not as
  a drop-in product profile.
- **No 0.98 assurance claim.** The target metrics are not met.
- Next steps that could change the result: a larger local QA/NLI model, more
  target-distribution calibration data, per-domain recalibration, and a
  held-out operational pilot. A threshold sweep alone is not enough.

## Artifacts

Hybrid (best effort):

- [`benchmarks/results/answerability_hybrid/answerability_observations.csv`](../benchmarks/results/answerability_hybrid/answerability_observations.csv)
- [`benchmarks/results/answerability_hybrid/answerability_profile.json`](../benchmarks/results/answerability_hybrid/answerability_profile.json)
- [`benchmarks/results/answerability_hybrid/answerability_report.md`](../benchmarks/results/answerability_hybrid/answerability_report.md)
- [`benchmarks/results/answerability_hybrid/answerability.toml`](../benchmarks/results/answerability_hybrid/answerability.toml)
- `aisc_policy_heldout.txt`, `aisc_policy_calibration.txt`, `aisc_check.txt`

Feature-only gate:

- [`benchmarks/results/answerability_signal/answerability_profile.json`](../benchmarks/results/answerability_signal/answerability_profile.json)
- [`benchmarks/results/answerability_signal/answerability_observations.csv`](../benchmarks/results/answerability_signal/answerability_observations.csv)
- [`benchmarks/results/answerability_signal/answerability_report.md`](../benchmarks/results/answerability_signal/answerability_report.md)

DistilBERT baseline:

- [`benchmarks/results/answerability_transformer/answerability_profile.json`](../benchmarks/results/answerability_transformer/answerability_profile.json)
- [`benchmarks/results/answerability_transformer/answerability_report.md`](../benchmarks/results/answerability_transformer/answerability_report.md)
