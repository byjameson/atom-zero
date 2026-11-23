# Atom Zero handoff - 2026-11-23

## Current state

- Last feature commit: `276605c` (`feat: measure independent answerability signal`)
- Working tree after the handoff commit: clean
- Atom Zero tests: **432 passed**
- AISC tests: **191 passed** (+6 subtests)
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
- Document timeline: date extraction, entity filtering, CLI, HTTP, chat
  (`timeline`, `zaman çizelgesi`) and UI. Timeline regression: **10/10**.
- Typed relations: `located_in`, `capital_of`, `founded_by`, `ceo_of`,
  `born_in`; CLI/HTTP/chat/UI filters, provenance and multi-hop relation paths.
  Relations regression: **12/12**.
- CitationGate: calibrated rerank -> validity probability, configurable
  threshold/profile, citation dropping and explicit refusal. Default is
  `squad_v2_stress` at **0.5**. SQuAD v2 policy simulation: **92/103**
  answerable retained, **18/97** no-answer refused, precision **0.479**.
  Answerability is a separate optional gate and is not enabled by default; see
  the decision below.
- Local isotonic calibration utility (`python -m atomzero.calibrate`) fits a
  PAVA map, reports Brier/ECE and recommends a threshold from score/label CSV
  for `ATOMZERO_CITATION_CALIBRATION`.
- Answerability initial experiment: 15 lexical/type features on 200 SQuAD v2
  questions; best univariate AUC **0.599**, context-grouped 5-fold CV AUC
  **0.562** (logistic) / **0.497** (boost). The follow-up 42-feature and hybrid
  experiments are recorded below. Reproducible with
  `benchmarks/answerability_eval.py`.
- External adapter validation: `research/adapter_runtime.py`,
  `research/adapter_check.py` and `research/compare_adapters.py` consume an
  adapter without importing `streamft`. The accepted adapter (368,640 LoRA
  params) scored **3.853 -> 3.139** (-18.5%) on the two-row holdout. A 20-step
  external behavior-mix adapter also loaded and passed the compatibility check.
- Independent answerability signal: added `src/atomzero/answerability.py`
  behind `ATOMZERO_ANSWERABILITY_GATE` (`off` default; `shadow`/`enforce`
  optional). It is separate from CitationGate and has 42 retrieval/answer-type
  features. The best hybrid (42 features + 5 tiny local QA features) reached
  held-out AUC **0.663**, Brier **0.244**, ECE **0.130**, but failed the
  refusal recall/precision and coverage targets. The AISC strict check is
  **FAIL** with `ANSWERABILITY_POOR` and `REFUSAL_POLICY_MISMATCH`; see
  `docs/ANSWERABILITY.md`. The default citation threshold remains **0.5** and
  the answerability gate remains **off**.

## Next tasks

1. Improve answerability only with a stronger local QA/NLI signal and
   target-distribution calibration data; do not enable the current gate by
   default. The measured targets remain unmet.
2. Train a targeted adapter on a genuinely novel per-domain queue with
   `streamft` or `continual-lora`; validate it with
   `research/adapter_check.py` and score a held-out queue with
   `compare_adapters.py`. The 267-row format queue is already known to the base
   (holdout 0.224) and both tested adapters regressed (+34.7% and +1.6%), so it
   is a negative control, not a training target. Keep the repos separate; Atom
   Zero must not import `streamft`. Model weights are still `0.8.0` and
   unchanged.
3. Prepare the `0.6.0` batch release once fresh credentials are available.
4. After the next code commit, re-run full pytest, AISC, build/twine and
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
python benchmarks/answerability_eval.py
python benchmarks/answerability_signal.py
python benchmarks/answerability_transformer.py
python benchmarks/answerability_hybrid.py
python research/adapter_check.py --base <base> --adapter <adapter>
python research/compare_adapters.py --base <base> --data <holdout.jsonl> --adapters <adapter>
python -m atomzero.calibrate citations.csv --out my-calibration.json
python benchmarks/squad_retrieval.py --max-contexts 50 --max-questions 200 --per-context 5 --seed 0```

AISC:

```powershell
cd "C:\Users\biris\Desktop\opencode\yeni dal\ai-systems-algebra"
python -m pytest tests -q
```
