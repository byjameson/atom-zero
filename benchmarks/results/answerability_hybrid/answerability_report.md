# Hybrid answerability signal and refusal policy

Commit: `b4b68b18b5d7122d182e387527f4f54d8a59b094`
Dataset / split: rajpurkar/squad_v2 validation (50 contexts / 200 questions)
Signal features: 42 retrieval/answer-type features plus 5 tiny SQuAD v2 QA features (no-answer score, best span, margin, span length, answer probability). CitationGate probabilities are not used.
Calibration method: logistic model + isotonic calibration on a context-grouped calibration split.
n: train=1165, calibration=289, held-out test=200
Brier / ECE: 0.2443 / 0.1299
Threshold: 0.60
Refusal precision / recall: 0.6310 / 0.5464
Answerable coverage: 0.6990

## Held-out results

| metric | value | target | pass |
|---|---:|---:|:---:|
| refusal_recall | 0.5464 | 0.90 | no |
| refusal_precision | 0.6310 | 0.90 | no |
| answerable_coverage | 0.6990 | 0.80 | no |
| answerability Brier | 0.2443 | 0.15 | no |
| answerability ECE | 0.1299 | 0.10 | no |

## Threshold table (held-out)

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.1 | 0.990 | 0.929 | 0.134 | 0.575 |
| 0.2 | 0.990 | 0.929 | 0.134 | 0.575 |
| 0.3 | 0.990 | 0.929 | 0.134 | 0.575 |
| 0.4 | 0.971 | 0.842 | 0.165 | 0.580 |
| 0.5 | 0.971 | 0.850 | 0.175 | 0.585 |
| 0.6 | 0.699 | 0.631 | 0.546 | 0.625 |
| 0.7 | 0.544 | 0.591 | 0.701 | 0.620 |
| 0.8 | 0.243 | 0.521 | 0.876 | 0.550 |
| 0.9 | 0.233 | 0.518 | 0.876 | 0.545 |

## AISC policy

```text
$ C:\Users\biris\AppData\Local\Programs\Python\Python311\python.exe -m aisc policy C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_hybrid\answerability_observations.csv --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --min-refusal-recall 0.9 --min-refusal-precision 0.9 --min-answerable-coverage 0.8 --json

{
  "n": 200,
  "points": [
    {
      "threshold": 0.1,
      "accepted": 186,
      "refused": 14,
      "true_positive": 102,
      "false_positive": 84,
      "precision": 0.548387,
      "recall": 0.990291,
      "coverage": 0.93,
      "precision_wilson_95": [
        0.476609,
        0.618207
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 13,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.134021,
      "refusal_precision": 0.928571,
      "overall_accuracy": 0.575
    },
    {
      "threshold": 0.2,
      "accepted": 186,
      "refused": 14,
      "true_positive": 102,
      "false_positive": 84,
      "precision": 0.548387,
      "recall": 0.990291,
      "coverage": 0.93,
      "precision_wilson_95": [
        0.476609,
        0.618207
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 13,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.134021,
      "refusal_precision": 0.928571,
      "overall_accuracy": 0.575
    },
    {
      "threshold": 0.3,
      "accepted": 186,
      "refused": 14,
      "true_positive": 102,
      "false_positive": 84,
      "precision": 0.548387,
      "recall": 0.990291,
      "coverage": 0.93,
      "precision_wilson_95": [
        0.476609,
        0.618207
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 13,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.134021,
      "refusal_precision": 0.928571,
      "overall_accuracy": 0.575
    },
    {
      "threshold": 0.4,
      "accepted": 181,
      "refused": 19,
      "true_positive": 100,
      "false_positive": 81,
      "precision": 0.552486,
      "recall": 0.970874,
      "coverage": 0.905,
      "precision_wilson_95": [
        0.479703,
        0.623087
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 100,
      "no_answer_refused": 16,
      "answerable_coverage": 0.970874,
      "no_answer_refusal": 0.164948,
      "refusal_precision": 0.842105,
      "overall_accuracy": 0.58
    },
    {
      "threshold": 0.5,
      "accepted": 180,
      "refused": 20,
      "true_positive": 100,
      "false_positive": 80,
      "precision": 0.555556,
      "recall": 0.970874,
      "coverage": 0.9,
      "precision_wilson_95": [
        0.482555,
        0.626234
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 100,
      "no_answer_refused": 17,
      "answerable_coverage": 0.970874,
      "no_answer_refusal": 0.175258,
      "refusal_precision": 0.85,
      "overall_accuracy": 0.585
    },
    {
      "threshold": 0.6,
      "accepted": 116,
      "refused": 84,
      "true_positive": 72,
      "false_positive": 44,
      "precision": 0.62069,
      "recall": 0.699029,
      "coverage": 0.58,
      "precision_wilson_95": [
        0.529861,
        0.70378
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 72,
      "no_answer_refused": 53,
      "answerable_coverage": 0.699029,
      "no_answer_refusal": 0.546392,
      "refusal_precision": 0.630952,
      "overall_accuracy": 0.625
    },
    {
      "threshold": 0.7,
      "accepted": 85,
      "refused": 115,
      "true_positive": 56,
      "false_positive": 29,
      "precision": 0.658824,
      "recall": 0.543689,
      "coverage": 0.425,
      "precision_wilson_95": [
        0.553129,
        0.750782
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 56,
      "no_answer_refused": 68,
      "answerable_coverage": 0.543689,
      "no_answer_refusal": 0.701031,
      "refusal_precision": 0.591304,
      "overall_accuracy": 0.62
    },
    {
      "threshold": 0.8,
      "accepted": 37,
      "refused": 163,
      "true_positive": 25,
      "false_positive": 12,
      "precision": 0.675676,
      "recall": 0.242718,
      "coverage": 0.185,
      "precision_wilson_95": [
        0.514634,
        0.803669
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 25,
      "no_answer_refused": 85,
      "answerable_coverage": 0.242718,
      "no_answer_refusal": 0.876289,
      "refusal_precision": 0.521472,
      "overall_accuracy": 0.55
    },
    {
      "threshold": 0.9,
      "accepted": 36,
      "refused": 164,
      "true_positive": 24,
      "false_positive": 12,
      "precision": 0.666667,
      "recall": 0.23301,
      "coverage": 0.18,
      "precision_wilson_95": [
        0.503337,
        0.797856
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 24,
      "no_answer_refused": 85,
      "answerable_coverage": 0.23301,
      "no_answer_refusal": 0.876289,
      "refusal_precision": 0.518293,
      "overall_accuracy": 0.545
    }
  ],
  "answerable": 103,
  "no_answer": 97,
  "source": "C:\\Users\\biris\\AppData\\Local\\Temp\\opencode\\atom-zero-clean\\benchmarks\\results\\answerability_hybrid\\answerability_observations.csv",
  "recommendation": {
    "feasible": false,
    "threshold": 0.7,
    "targets": {
      "refusal_recall": 0.9,
      "refusal_precision": 0.9,
      "answerable_coverage": 0.8
    },
    "point": {
      "threshold": 0.7,
      "accepted": 85,
      "refused": 115,
      "true_positive": 56,
      "false_positive": 29,
      "precision": 0.658824,
      "recall": 0.543689,
      "coverage": 0.425,
      "precision_wilson_95": [
        0.553129,
        0.750782
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 56,
      "no_answer_refused": 68,
      "answerable_coverage": 0.543689,
      "no_answer_refusal": 0.701031,
      "refusal_precision": 0.591304,
      "overall_accuracy": 0.62
    },
    "misses": [
      "refusal_recall",
      "refusal_precision",
      "answerable_coverage"
    ],
    "reason": "Hedefleri karşılayan eşik yok"
  }
}
```

## AISC strict check

```text
$ C:\Users\biris\AppData\Local\Programs\Python\Python311\python.exe -m aisc check C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_hybrid\answerability.toml --require-calibration --strict

AISC check — AtomZeroAnswerability
Kaynak: C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_hybrid\answerability.toml
Topoloji: sequential
Bileşen: 1
Güvence sınırı: 0.80 (hedef 0.80)
Bağımsız başarı referansı: 0.800
Bağımlı başarı referansı (ρ=0.00): 0.800
Doğrulama borcu: 0
Geri dönüşsüz: 0
Bağlam: 2000 / 50000 token

2 bulgu:
 - WARNING ANSWERABILITY_POOR: Answerability kalibrasyonu zayıf: Brier=0.244, ECE=0.130 (üst sınır Brier≤0.15, ECE≤0.10). Öneri: Answerability sinyalini iyileştirin veya daha fazla etiketli gözlem toplayın.
 - WARNING REFUSAL_POLICY_MISMATCH: Ret politikası hedefleri karşılamıyor: refusal recall 0.546 < 0.90; refusal precision 0.631 < 0.90; answerable coverage 0.699 < 0.80. Eşik=0.60. Öneri: Answerability sinyalini ve refusal eşiğini hedef dağıtımda yeniden kalibre edin.

Sonuç: FAIL (0 kritik, 2 uyarı)
```

## Baselines

| signal | held-out AUC | Brier | ECE |
|---|---:|---:|---:|
| feature-only logistic | 0.5635 | 0.2703 | 0.1407 |
| DistilBERT cross-encoder | 0.5375 | 0.2762 | 0.1612 |
| **hybrid (this run)** | **0.6627** | **0.2443** | **0.1299** |

## Decision and limitations

- SQuAD v2 no-answer questions share entities and vocabulary with the source paragraph; this is a stress distribution, not a production pilot.
- The hybrid uses a tiny local QA model as a confidence feature; it is an experiment artifact, not a default product gate.
- The hybrid profile is not loadable by the feature-only optional gate; that gate uses the profile under `benchmarks/results/answerability_signal`.
- No threshold meets the 0.90 refusal-recall, 0.90 refusal-precision and 0.80 answerable-coverage targets together.
- CitationGate remains a separate gate and the default refusal threshold remains 0.5.
- No 0.98 assurance claim is made.
