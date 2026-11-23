# CitationGate threshold and refusal policy

Atom Zero applies a calibrated citation-validity gate to local cited answers.
The gate maps each rerank score to a validity probability, drops citations
below a configurable threshold, and refuses the answer when no citation remains.

## Defaults and configuration

The default is intentionally not the authored `0.6` threshold:

| setting | default |
|---|---|
| threshold | `0.5` |
| profile | `squad_v2_stress` |
| mode | `enforce` |

Environment overrides:

```text
ATOMZERO_CITATION_THRESHOLD    float in 0..1
ATOMZERO_CITATION_PROFILE      squad_v2_stress | authored_v1
ATOMZERO_CITATION_CALIBRATION  path to a JSON calibration map
ATOMZERO_CITATION_GATE         enforce | shadow | off
```

The calibration JSON accepts `x_thresholds`/`y_thresholds` at the top level or
nested under `isotonic`. Built-in profiles:

- `authored_v1`: AISC authored-document citation calibration (118 held-out
  citations, 40 positive), Brier 0.0293 / ECE 0.0382.
- `squad_v2_stress`: AISC SQuAD v2 stress calibration (100 questions).

These maps are distribution-specific starting points. A production deployment
must recalibrate on its own queries; the gate alone does not solve
answerability.

## Response behavior

`grounded_answer` returns `refused: true`, `reason:
"no_citation_above_threshold"` and a `gate` object when every selected citation
falls below the threshold:

```json
{
  "text": "Yeterli güvenilir yerel kaynak bulamadım. (No citation passed the local confidence threshold.)",
  "sources": [],
  "evidence": [],
  "refused": true,
  "reason": "no_citation_above_threshold",
  "gate": {"applied": true, "profile": "squad_v2_stress",
           "threshold": 0.5, "accepted": 0, "dropped": 3}
}
```

When citations pass, each evidence item carries `citation_probability` and
`citation_threshold`. The agent runtime and docs endpoints propagate the same
`refused`/`reason`/`gate` fields. `Yeterli güvenilir yerel kaynak bulamadım`
is also used when no local hit exists.

Explicit document-inspection queries (for example, “what does the safety
document say?”) quote the requested source even if its validity probability is
below the gate. This is a narrow quote path, not general answer release;
document text is still treated as untrusted data and never as instructions.

## SQuAD v2 policy simulation

`benchmarks/citation_gate_eval.py` runs 50 contexts / 200 questions (103
answerable, 97 no-answer). Citation precision counts released answers whose
visible snippet contains the gold answer.

| profile | threshold | answered/correct | no-answer refused | precision | overall |
|---|---:|---:|---:|---:|---:|
| authored_v1 | 0.3 | 93 / 82 | 13/97 | 0.463 | 0.475 |
| authored_v1 | 0.5 | 69 / 64 | 34/97 | 0.485 | 0.490 |
| authored_v1 | 0.6 | 58 / 56 | 44/97 | 0.504 | 0.500 |
| **squad_v2_stress** | **0.5** | **92 / 82** | **18/97** | **0.479** | **0.500** |
| squad_v2_stress | 0.6 | 2 / 2 | 97/97 | 1.000 | 0.495 |

The default retains 92/103 answerable questions, refuses 18/97 no-answer
questions, and reaches precision 0.479. It is a starting point, not a
production proof: precision remains near the release-all baseline because the
rerank score is not an answerability signal. A separate answerability signal
and target-distribution calibration are still required.

Run:

```powershell
python benchmarks/citation_gate_eval.py
```

Raw result:
[`benchmarks/results/citation_gate_eval.json`](../benchmarks/results/citation_gate_eval.json).
