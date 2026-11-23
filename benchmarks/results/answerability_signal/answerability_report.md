# Answerability signal and refusal policy

Commit: `b4b68b18b5d7122d182e387527f4f54d8a59b094`
Dataset / split: rajpurkar/squad_v2 validation (50 contexts / 200 questions)
Signal features: 42 local retrieval and answer-type features; CitationGate probabilities are not used.
Calibration method: logistic model + isotonic calibration on a context-grouped calibration split.
n: train=1165, calibration=289, held-out test=200
Brier / ECE: 0.2703 / 0.1407
Threshold: 0.60
Refusal precision / recall: 0.5526 / 0.4330
Answerable coverage: 0.6699

## Held-out results

| metric | value | target | pass |
|---|---:|---:|:---:|
| refusal_recall | 0.4330 | 0.90 | no |
| refusal_precision | 0.5526 | 0.90 | no |
| answerable_coverage | 0.6699 | 0.80 | no |
| answerability Brier | 0.2703 | 0.15 | no |
| answerability ECE | 0.1407 | 0.10 | no |

Wilson 95% intervals:

- refusal precision: [0.441, 0.6592]
- refusal recall: [0.3388, 0.5323]
- answerable coverage: [0.5744, 0.7532]

## Threshold table (held-out)

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.1 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.2 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.3 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.4 | 0.990 | 0.000 | 0.000 | 0.510 |
| 0.5 | 0.913 | 0.526 | 0.103 | 0.520 |
| 0.6 | 0.670 | 0.553 | 0.433 | 0.555 |
| 0.7 | 0.340 | 0.518 | 0.753 | 0.540 |
| 0.8 | 0.049 | 0.490 | 0.969 | 0.495 |
| 0.9 | 0.039 | 0.487 | 0.969 | 0.490 |

## AISC policy

```text
$ C:\Users\biris\AppData\Local\Programs\Python\Python311\python.exe -m aisc policy C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_signal\answerability_observations.csv --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --min-refusal-recall 0.9 --min-refusal-precision 0.9 --min-answerable-coverage 0.8 --json

{
  "n": 200,
  "points": [
    {
      "threshold": 0.1,
      "accepted": 199,
      "refused": 1,
      "true_positive": 102,
      "false_positive": 97,
      "precision": 0.512563,
      "recall": 0.990291,
      "coverage": 0.995,
      "precision_wilson_95": [
        0.443537,
        0.581113
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 0,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.0,
      "refusal_precision": 0.0,
      "overall_accuracy": 0.51
    },
    {
      "threshold": 0.2,
      "accepted": 199,
      "refused": 1,
      "true_positive": 102,
      "false_positive": 97,
      "precision": 0.512563,
      "recall": 0.990291,
      "coverage": 0.995,
      "precision_wilson_95": [
        0.443537,
        0.581113
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 0,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.0,
      "refusal_precision": 0.0,
      "overall_accuracy": 0.51
    },
    {
      "threshold": 0.3,
      "accepted": 199,
      "refused": 1,
      "true_positive": 102,
      "false_positive": 97,
      "precision": 0.512563,
      "recall": 0.990291,
      "coverage": 0.995,
      "precision_wilson_95": [
        0.443537,
        0.581113
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 0,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.0,
      "refusal_precision": 0.0,
      "overall_accuracy": 0.51
    },
    {
      "threshold": 0.4,
      "accepted": 199,
      "refused": 1,
      "true_positive": 102,
      "false_positive": 97,
      "precision": 0.512563,
      "recall": 0.990291,
      "coverage": 0.995,
      "precision_wilson_95": [
        0.443537,
        0.581113
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 102,
      "no_answer_refused": 0,
      "answerable_coverage": 0.990291,
      "no_answer_refusal": 0.0,
      "refusal_precision": 0.0,
      "overall_accuracy": 0.51
    },
    {
      "threshold": 0.5,
      "accepted": 181,
      "refused": 19,
      "true_positive": 94,
      "false_positive": 87,
      "precision": 0.519337,
      "recall": 0.912621,
      "coverage": 0.905,
      "precision_wilson_95": [
        0.446906,
        0.590964
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 94,
      "no_answer_refused": 10,
      "answerable_coverage": 0.912621,
      "no_answer_refusal": 0.103093,
      "refusal_precision": 0.526316,
      "overall_accuracy": 0.52
    },
    {
      "threshold": 0.6,
      "accepted": 124,
      "refused": 76,
      "true_positive": 69,
      "false_positive": 55,
      "precision": 0.556452,
      "recall": 0.669903,
      "coverage": 0.62,
      "precision_wilson_95": [
        0.468619,
        0.640892
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 69,
      "no_answer_refused": 42,
      "answerable_coverage": 0.669903,
      "no_answer_refusal": 0.43299,
      "refusal_precision": 0.552632,
      "overall_accuracy": 0.555
    },
    {
      "threshold": 0.7,
      "accepted": 59,
      "refused": 141,
      "true_positive": 35,
      "false_positive": 24,
      "precision": 0.59322,
      "recall": 0.339806,
      "coverage": 0.295,
      "precision_wilson_95": [
        0.465932,
        0.709112
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 35,
      "no_answer_refused": 73,
      "answerable_coverage": 0.339806,
      "no_answer_refusal": 0.752577,
      "refusal_precision": 0.51773,
      "overall_accuracy": 0.54
    },
    {
      "threshold": 0.8,
      "accepted": 8,
      "refused": 192,
      "true_positive": 5,
      "false_positive": 3,
      "precision": 0.625,
      "recall": 0.048544,
      "coverage": 0.04,
      "precision_wilson_95": [
        0.305738,
        0.863158
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 5,
      "no_answer_refused": 94,
      "answerable_coverage": 0.048544,
      "no_answer_refusal": 0.969072,
      "refusal_precision": 0.489583,
      "overall_accuracy": 0.495
    },
    {
      "threshold": 0.9,
      "accepted": 7,
      "refused": 193,
      "true_positive": 4,
      "false_positive": 3,
      "precision": 0.571429,
      "recall": 0.038835,
      "coverage": 0.035,
      "precision_wilson_95": [
        0.250454,
        0.841783
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 4,
      "no_answer_refused": 94,
      "answerable_coverage": 0.038835,
      "no_answer_refusal": 0.969072,
      "refusal_precision": 0.487047,
      "overall_accuracy": 0.49
    }
  ],
  "answerable": 103,
  "no_answer": 97,
  "source": "C:\\Users\\biris\\AppData\\Local\\Temp\\opencode\\atom-zero-clean\\benchmarks\\results\\answerability_signal\\answerability_observations.csv",
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
      "accepted": 59,
      "refused": 141,
      "true_positive": 35,
      "false_positive": 24,
      "precision": 0.59322,
      "recall": 0.339806,
      "coverage": 0.295,
      "precision_wilson_95": [
        0.465932,
        0.709112
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 35,
      "no_answer_refused": 73,
      "answerable_coverage": 0.339806,
      "no_answer_refusal": 0.752577,
      "refusal_precision": 0.51773,
      "overall_accuracy": 0.54
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
$ C:\Users\biris\AppData\Local\Programs\Python\Python311\python.exe -m aisc check C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_signal\answerability.toml --require-calibration --strict

AISC check — AtomZeroAnswerability
Kaynak: C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_signal\answerability.toml
Topoloji: sequential
Bileşen: 1
Güvence sınırı: 0.80 (hedef 0.80)
Bağımsız başarı referansı: 0.800
Bağımlı başarı referansı (ρ=0.00): 0.800
Doğrulama borcu: 0
Geri dönüşsüz: 0
Bağlam: 2000 / 50000 token

2 bulgu:
 - WARNING ANSWERABILITY_POOR: Answerability kalibrasyonu zayıf: Brier=0.270, ECE=0.141 (üst sınır Brier≤0.15, ECE≤0.10). Öneri: Answerability sinyalini iyileştirin veya daha fazla etiketli gözlem toplayın.
 - WARNING REFUSAL_POLICY_MISMATCH: Ret politikası hedefleri karşılamıyor: refusal recall 0.433 < 0.90; refusal precision 0.553 < 0.90; answerable coverage 0.670 < 0.80. Eşik=0.60. Öneri: Answerability sinyalini ve refusal eşiğini hedef dağıtımda yeniden kalibre edin.

Sonuç: FAIL (0 kritik, 2 uyarı)
```

## Decision and limitations

- SQuAD v2 no-answer questions share entities and vocabulary with the source paragraph; this is a stress distribution, not a production pilot.
- The signal is retrieval-feature based; no NLI or QA model is used.
- The 0.90 refusal target is not met; CitationGate remains the only default refusal gate.
- The answerability gate is implemented behind an environment flag and defaults to off, so this experiment does not change product behavior.
- No 0.98 assurance claim is made.
