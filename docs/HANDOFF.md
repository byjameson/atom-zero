# Atom Zero handoff - 2026-11-23

## Current state

- Last released batch: `0.6.1` on `main` (memory ledger, claim/evidence
  algebra, generated AISC spec)
- Post-release main (`0.6.2` work): `POST /v1/claims/audit` and AISC 0.5
  `batch`/`diff` documentation. Not pushed yet; the next batch push is the
  `0.6.2` release boundary.
- Working tree after the release-doc commits: clean
- Atom Zero tests: **513 passed** (512 at the `0.6.1` tag)
- AISC strict audit: **7 specs PASS** (`Claims`, `ContextBudget`,
  `MemoryLifecycle`, `Proactive`, `ReadPipeline`, `TaskActions`, `ToolPlan`);
  `AtomZeroAnswerability` and `AtomZeroAnswerabilityGenerated` remain **FAIL**
  with the measured Brier 0.244 / ECE 0.130 and refusal-policy misses. This is
  reported, not hidden.
- Build/twine: **PASSED** (`atomzero-0.6.1` wheel + sdist)
- Release status: `0.6.1` is published on PyPI, GitHub, Hugging Face,
  ModelScope, dev.to, Medium and LinkedIn. `scripts/verify_release.py
  --version 0.6.1` reports **0 failures**. The `publish.yml` trusted-publisher
  run remains a historical failed attempt; uploads use a project token from the
  owner's local token file.
- Post-release `main` has one local, unpushed commit for `0.6.2` (claim audit
  HTTP route + AISC 0.5 docs). The ~10-increment batch-push policy continues.

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
- Packaged AISC integration: `atomzero-audit` and `python -m atomzero.audit`
  expose strict warnings, `--require-calibration`, JSON and remediation-ready
  issue codes; `atomzero-doctor --aisc` folds the optional audit into preflight.
  The earlier `scripts/aisc_audit.py` false-PASS path is removed.
- New **tool plan algebra**: `atomzero-plan` and `POST /v1/tools/audit` report
  side effects, irreversible steps, confirmation requirements and risk before
  execution (`src/atomzero/plan_audit.py`).
- New **context budget algebra**: `plan_hits`,
  `grounded_answer(max_context_chars=...)` and
  `atomzero-docs answer --max-context-chars` bound retrieved evidence and report
  dropped hits (`src/atomzero/context_budget.py`).
- New **proactive conversation**: opt-in local scheduler that starts a short
  chat from overdue/due tasks, knowledge gaps or recent sources, with cooldowns,
  quiet hours, a daily cap and a UI toggle (`src/atomzero/proactive.py`,
  `POST /v1/proactive*`).
- New **memory lifecycle** first slice: document/session deletion requires an
  explicit server-side `confirm: true` (409 without it, 400 for a non-boolean),
  and the AISC `AtomZeroMemoryLifecycle` spec places `HumanConfirmation` before
  the irreversible `MemoryStore`.
- Added the **content-free memory ledger**:
  `atomzero-memory-ledger`, `GET /v1/memory/ledger` and
  `POST /v1/memory/ledger/verify`. Successful deletes append counts/scope only;
  the validator rejects text, query, answer, chunk and session content fields.
  `LedgerWriter` now follows `MemoryStore` in the AISC spec. Main is ahead of
  the `v0.6.0` release assets; the ledger belongs to the next patch batch.

## Next tasks

1. Continue the memory lifecycle: cascade deletion to feedback/training queues,
   add a content-free mutation ledger, and make session-file deletion explicit.
2. Improve answerability only with a stronger local QA/NLI signal and
   target-distribution calibration data; do not enable the current gate by
   default. The measured targets remain unmet.
3. Train a targeted adapter on a genuinely novel per-domain queue with
   `streamft` or `continual-lora`; validate it with
   `research/adapter_check.py` and score a held-out queue with
   `compare_adapters.py`. The 267-row format queue is already known to the base
   (holdout 0.224) and both tested adapters regressed (+34.7% and +1.6%), so it
   is a negative control, not a training target. Keep the repos separate; Atom
   Zero must not import `streamft`. Model weights are still `0.8.0` and
   unchanged.
4. Merge the claim/evidence branch after the Reviewer gate, then cut `0.6.1`
   with the post-release memory ledger and claims work. Run
   `scripts/verify_release.py --version 0.6.1` before publishing.
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
