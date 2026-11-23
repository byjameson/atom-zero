# Atom Zero handoff - 2026-11-23

## Current state

- Last feature commit: `ae5efa1` (`feat: extract typed relations`)
- Working tree after the handoff commit: clean
- Atom Zero tests: **393 passed**
- AISC tests: **163 passed** (+6 subtests)
- AISC audit: **PASS**, verification debt 0
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
- Answerability experiment: 5-fold context-grouped CV AUC ~**0.56**. This is
  too weak to ship as a gate or classifier; it is not part of the product.

## Next tasks (tomorrow)

1. Add typed-relations chat intent and UI parity (CLI/HTTP already work).
2. Decide whether to improve answerability with stronger features or drop it.
3. Explore a small local adapter with `streamft`/`continual-lora` as external
   trainers. Keep the repos separate; Atom Zero must not import `streamft`.
   Model weights are still `0.8.0` and unchanged.
4. Prepare the `0.6.0` batch release once fresh credentials are available.
5. After the next code commit, re-run full pytest, AISC, build/twine and
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
python benchmarks/squad_retrieval.py --max-contexts 50 --max-questions 200 --per-context 5 --seed 0
```

AISC:

```powershell
cd "C:\Users\biris\Desktop\opencode\yeni dal\ai-systems-algebra"
python -m pytest tests -q
```

AISC audit:

```powershell
cd C:\Users\biris\AppData\Local\Temp\opencode\atom-zero-clean
$env:PYTHONPATH="$PWD\src"
python scripts/aisc_audit.py --aisc-path "C:\Users\biris\Desktop\opencode\yeni dal\ai-systems-algebra"
```
