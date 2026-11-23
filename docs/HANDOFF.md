# Atom Zero handoff - 2026-11-23

## Current state

- Last feature commit: `efad585` (`feat: add calibrated citation gate`)
- Working tree after the handoff commit: clean
- Atom Zero tests: **400 passed**
- AISC tests: **163 passed** (+6 subtests)
- AISC strict audit: **not claimed as passing**; the AISC spec/report update is
  owned by the AISC side. Atom Zero files were not modified in that repo.
- Build/twine: **PASSED**
- No release or push: fresh PyPI/HF/ModelScope/dev.to credentials and a GitHub
  session are still missing. The ~10-increment batch policy continues.

## Completed in this session

- Global snippet-window scoring. Public SQuAD v2 visible-snippet accuracy:
  **0.816 -> 0.884** (hybrid), **0.777 -> 0.845** (FTS).
- Contradiction normalization for numbers, dates and compatible units, plus
  topic-paraphrase alignment. Contradiction regression: **16/16**.
- Turkish morphology retrieval (suffix, final-consonant mutation, vowel drop,
  FTS prefix variants). Authored 20-query check: hybrid **0.800 -> 1.000**,
  FTS **0.700 -> 0.950**. English SQuAD results unchanged.
- Document timeline: date extraction, CLI, HTTP, chat (`timeline`,
  `zaman çizelgesi`) and UI. Timeline regression: **9/9**.
- Typed relations: `located_in`, `capital_of`, `founded_by`, `ceo_of`,
  `born_in`; CLI and HTTP filters/provenance. Relations regression: **6/6**.
- CitationGate: calibrated rerank -> validity probability, configurable
  threshold/profile, citation dropping and explicit refusal. Default is
  `squad_v2_stress` at **0.5**. SQuAD v2 policy simulation: **92/103**
  answerable retained, **18/97** no-answer refused, precision **0.479**.
  Answerability is still a separate open problem.
- Answerability experiment: 5-fold context-grouped CV AUC ~**0.56**. This is
  too weak to ship as a gate or classifier; it is not part of the product.

## Next tasks (tomorrow)

1. Add a target-distribution calibration utility that fits an isotonic map and
   recommends a threshold from labelled `score,label` data.
2. Add typed-relations chat intent and UI parity (CLI/HTTP already work).
3. Decide whether to improve answerability with stronger features or drop it.
4. Explore a small local adapter with `streamft`/`continual-lora` as external
   trainers. Keep the repos separate; Atom Zero must not import `streamft`.
   Model weights are still `0.8.0` and unchanged.
5. Prepare the `0.6.0` batch release once fresh credentials are available.
6. After the next code commit, re-run full pytest, AISC, build/twine and
   refresh `TEST_RESULTS.md`.

## Resume commands

```powershell
cd C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean
$env:PYTHONPATH="$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests -q
python benchmarks/relations_regression.py
python benchmarks/timeline_regression.py
python benchmarks/contradiction_regression.py
python benchmarks/turkish_retrieval.py
python benchmarks/citation_gate_eval.py
python benchmarks/squad_retrieval.py --max-contexts 50 --max-questions 200 --per-context 5 --seed 0
```

AISC:

```powershell
cd "C:\Users\biris\Desktop\opencode\yeni dal\ai-systems-algebra"
python -m pytest tests -q
```
