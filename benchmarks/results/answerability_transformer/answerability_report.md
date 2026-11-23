# Local DistilBERT answerability baseline

Commit: `b4b68b18b5d7122d182e387527f4f54d8a59b094`
Dataset / split: rajpurkar/squad_v2 validation (50 contexts / 200 questions)
Signal: DistilBERT cross-encoder over question + top retrieved context; this is a stronger-model baseline, not the shipped gate.
Calibration method: isotonic calibration on a context-grouped calibration split.
n: train=1165, calibration=289, held-out test=200
Brier / ECE: 0.2762 / 0.1612
Threshold: 0.60
Refusal precision / recall: 0.5246 / 0.3299
Answerable coverage: 0.7184

## Held-out results

| metric | value | target | pass |
|---|---:|---:|:---:|
| refusal_recall | 0.3299 | 0.90 | no |
| refusal_precision | 0.5246 | 0.90 | no |
| answerable_coverage | 0.7184 | 0.80 | no |
| answerability Brier | 0.2762 | 0.15 | no |
| answerability ECE | 0.1612 | 0.10 | no |

## Threshold table (held-out)

| threshold | answerable cov. | refusal precision | refusal recall | overall |
|---:|---:|---:|---:|---:|
| 0.1 | 1.000 | 0.000 | 0.000 | 0.515 |
| 0.2 | 1.000 | 0.000 | 0.000 | 0.515 |
| 0.3 | 1.000 | 0.000 | 0.000 | 0.515 |
| 0.4 | 1.000 | 0.000 | 0.000 | 0.515 |
| 0.5 | 1.000 | 0.000 | 0.000 | 0.515 |
| 0.6 | 0.718 | 0.525 | 0.330 | 0.530 |
| 0.7 | 0.272 | 0.493 | 0.753 | 0.505 |
| 0.8 | 0.010 | 0.487 | 1.000 | 0.490 |
| 0.9 | 0.010 | 0.487 | 1.000 | 0.490 |

## AISC policy

```text
$ C:\Users\biris\AppData\Local\Programs\Python\Python311\python.exe -m aisc policy C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_transformer\answerability_observations.csv --thresholds 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9 --min-refusal-recall 0.9 --min-refusal-precision 0.9 --min-answerable-coverage 0.8 --json

{
  "n": 200,
  "points": [
    {
      "threshold": 0.1,
      "accepted": 200,
      "refused": 0,
      "true_positive": 103,
      "false_positive": 97,
      "precision": 0.515,
      "recall": 1.0,
      "coverage": 1.0,
      "precision_wilson_95": [
        0.446107,
        0.583327
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 103,
      "no_answer_refused": 0,
      "answerable_coverage": 1.0,
      "no_answer_refusal": 0.0,
      "refusal_precision": null,
      "overall_accuracy": 0.515
    },
    {
      "threshold": 0.2,
      "accepted": 200,
      "refused": 0,
      "true_positive": 103,
      "false_positive": 97,
      "precision": 0.515,
      "recall": 1.0,
      "coverage": 1.0,
      "precision_wilson_95": [
        0.446107,
        0.583327
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 103,
      "no_answer_refused": 0,
      "answerable_coverage": 1.0,
      "no_answer_refusal": 0.0,
      "refusal_precision": null,
      "overall_accuracy": 0.515
    },
    {
      "threshold": 0.3,
      "accepted": 200,
      "refused": 0,
      "true_positive": 103,
      "false_positive": 97,
      "precision": 0.515,
      "recall": 1.0,
      "coverage": 1.0,
      "precision_wilson_95": [
        0.446107,
        0.583327
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 103,
      "no_answer_refused": 0,
      "answerable_coverage": 1.0,
      "no_answer_refusal": 0.0,
      "refusal_precision": null,
      "overall_accuracy": 0.515
    },
    {
      "threshold": 0.4,
      "accepted": 200,
      "refused": 0,
      "true_positive": 103,
      "false_positive": 97,
      "precision": 0.515,
      "recall": 1.0,
      "coverage": 1.0,
      "precision_wilson_95": [
        0.446107,
        0.583327
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 103,
      "no_answer_refused": 0,
      "answerable_coverage": 1.0,
      "no_answer_refusal": 0.0,
      "refusal_precision": null,
      "overall_accuracy": 0.515
    },
    {
      "threshold": 0.5,
      "accepted": 200,
      "refused": 0,
      "true_positive": 103,
      "false_positive": 97,
      "precision": 0.515,
      "recall": 1.0,
      "coverage": 1.0,
      "precision_wilson_95": [
        0.446107,
        0.583327
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 103,
      "no_answer_refused": 0,
      "answerable_coverage": 1.0,
      "no_answer_refusal": 0.0,
      "refusal_precision": null,
      "overall_accuracy": 0.515
    },
    {
      "threshold": 0.6,
      "accepted": 139,
      "refused": 61,
      "true_positive": 74,
      "false_positive": 65,
      "precision": 0.532374,
      "recall": 0.718447,
      "coverage": 0.695,
      "precision_wilson_95": [
        0.449674,
        0.613333
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 74,
      "no_answer_refused": 32,
      "answerable_coverage": 0.718447,
      "no_answer_refusal": 0.329897,
      "refusal_precision": 0.52459,
      "overall_accuracy": 0.53
    },
    {
      "threshold": 0.7,
      "accepted": 52,
      "refused": 148,
      "true_positive": 28,
      "false_positive": 24,
      "precision": 0.538462,
      "recall": 0.271845,
      "coverage": 0.26,
      "precision_wilson_95": [
        0.405034,
        0.666597
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 28,
      "no_answer_refused": 73,
      "answerable_coverage": 0.271845,
      "no_answer_refusal": 0.752577,
      "refusal_precision": 0.493243,
      "overall_accuracy": 0.505
    },
    {
      "threshold": 0.8,
      "accepted": 1,
      "refused": 199,
      "true_positive": 1,
      "false_positive": 0,
      "precision": 1.0,
      "recall": 0.009709,
      "coverage": 0.005,
      "precision_wilson_95": [
        0.206543,
        1.0
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 1,
      "no_answer_refused": 97,
      "answerable_coverage": 0.009709,
      "no_answer_refusal": 1.0,
      "refusal_precision": 0.487437,
      "overall_accuracy": 0.49
    },
    {
      "threshold": 0.9,
      "accepted": 1,
      "refused": 199,
      "true_positive": 1,
      "false_positive": 0,
      "precision": 1.0,
      "recall": 0.009709,
      "coverage": 0.005,
      "precision_wilson_95": [
        0.206543,
        1.0
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 1,
      "no_answer_refused": 97,
      "answerable_coverage": 0.009709,
      "no_answer_refusal": 1.0,
      "refusal_precision": 0.487437,
      "overall_accuracy": 0.49
    }
  ],
  "answerable": 103,
  "no_answer": 97,
  "source": "C:\\Users\\biris\\AppData\\Local\\Temp\\opencode\\atom-zero-clean\\benchmarks\\results\\answerability_transformer\\answerability_observations.csv",
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
      "accepted": 52,
      "refused": 148,
      "true_positive": 28,
      "false_positive": 24,
      "precision": 0.538462,
      "recall": 0.271845,
      "coverage": 0.26,
      "precision_wilson_95": [
        0.405034,
        0.666597
      ],
      "answerable": 103,
      "no_answer": 97,
      "answerable_accepted": 28,
      "no_answer_refused": 73,
      "answerable_coverage": 0.271845,
      "no_answer_refusal": 0.752577,
      "refusal_precision": 0.493243,
      "overall_accuracy": 0.505
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
$ C:\Users\biris\AppData\Local\Programs\Python\Python311\python.exe -m aisc check C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_transformer\answerability.toml --require-calibration --strict

AISC check — AtomZeroAnswerability
Kaynak: C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean\benchmarks\results\answerability_transformer\answerability.toml
Topoloji: sequential
Bileşen: 1
Güvence sınırı: 0.80 (hedef 0.80)
Bağımsız başarı referansı: 0.800
Bağımlı başarı referansı (ρ=0.00): 0.800
Doğrulama borcu: 0
Geri dönüşsüz: 0
Bağlam: 2000 / 50000 token

2 bulgu:
 - WARNING ANSWERABILITY_POOR: Answerability kalibrasyonu zayıf: Brier=0.276, ECE=0.161 (üst sınır Brier≤0.15, ECE≤0.10). Öneri: Answerability sinyalini iyileştirin veya daha fazla etiketli gözlem toplayın.
 - WARNING REFUSAL_POLICY_MISMATCH: Ret politikası hedefleri karşılamıyor: refusal recall 0.330 < 0.90; refusal precision 0.525 < 0.90; answerable coverage 0.718 < 0.80. Eşik=0.60. Öneri: Answerability sinyalini ve refusal eşiğini hedef dağıtımda yeniden kalibre edin.

Sonuç: FAIL (0 kritik, 2 uyarı)
```

## Decision and limitations

- SQuAD v2 no-answer questions share entities and vocabulary with the source paragraph; this is a stress distribution.
- The DistilBERT cross-encoder is a local baseline; it is not shipped in the default product package and is not enabled by default.
- The target refusal metrics are still reported as pass/fail, not as an assurance claim.
- No 0.98 assurance claim is made.
