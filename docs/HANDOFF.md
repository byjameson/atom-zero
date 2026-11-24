# Atom Zero handoff - 2026-11-24

## Current state

- Last released batch: `0.20.0` on `main`; all canonical surfaces published (bounded self-improve loop,
  committed-change scope check, fifth measured cycle)
- Post-release main: self-code audit/diff/targeting with suppressions,
  planner history, scoreboard output collectors, escalation ladder, teacher
  answer verification, unified growth report, self-code before/after
  regression gate, autonomous sandboxed loop and the sixth/seventh/eighth
  measured cycles (`gate_id cac82f022c879258`, `855/0 -> 857/0`;
  `gate_id 7f0d5be2bce36219`, `1033/0 -> 1035/0`;
  `gate_id da83c0302933d484`, `1064/0 -> 1066/0`). Per-change human approval is
  now optional (`--require-approval`); the default loop merges locally after
  its gates. Verified-teacher adapter candidates and a held-out loss/perplexity
  gate are added for the capability-growth step; a real LoRA run on the local
  `0.8.0` base passed the gate (`docs/ADAPTER_TRAINING.md`). A real
  Qwen2.5-1.5B teacher queue and explicit cross-distribution fixture are
  prepared (`docs/TEACHER_QUEUE.md`). The first VM-run cross test passed
  in-distribution and failed cross-distribution, correctly blocking promotion
  (`docs/ADAPTER_CROSS_CYCLE.md`). A multi-family teacher + replay cycle
  improved the in-distribution gate and reduced the cross-family regression
  from 757x to 6.07x; replay weighting then produced a full-loss cross-family
  pass (`5.2549 -> 2.8027`, ppl ratio `0.086`). The unweighted content audit
  failed (`4.4327 -> 4.4123`); a content-weighted retrain passed the loss and
  content gates (`5.2549 -> 2.8489`, content `4.4327 -> 3.4892`), but the new
  exact-match audit found 0/48 held-out matches, so the registry was removed
  (`docs/DIVERSE_REPLAY_CYCLE.md`, `docs/CONTENT_AUDIT.md`,
  `docs/ADAPTER_PROMOTION.md`). A fixed-key copy curriculum improved
  content-token loss but still got 0/200 exact match; randomized-key training
  reached 3/200 exact and 19% key recall after 1,000 optimizer steps
  (`docs/COPY_CURRICULUM_CYCLE.md`, `docs/RANDOM_KEY_COPY_CYCLE.md`).
  Accumulating for the **1.0.0** boundary; not pushed yet.
- Working tree after the release-doc commits: clean
- Atom Zero tests: **1248 passed** (host) / **1248 passed** (VM, 8 vCPU)
- AISC strict audit: **26 specs, 24 PASS, 0 errors**; `AtomZeroAnswerability`
  and `AtomZeroAnswerabilityGenerated` remain **FAIL** with the measured
  Brier 0.233 / ECE 0.064 and refusal-policy misses. The ECE target is met;
  the Brier target is not. This is reported, not hidden.
- Build/twine: **PASSED** (`atomzero-0.19.0` wheel + sdist)
- No model weights changed; weights remain `0.8.0`.
- Release status: `0.20.0` is published on PyPI, GitHub, Hugging Face,
  ModelScope, dev.to, Medium, LinkedIn and X. `scripts/verify_release.py
  --version 0.20.0` reports **0 failures**; release commit `378fd0e`, annotated
  tag `v0.20.0`. The next boundary is **0.21.0**.

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
4. `0.6.1` is published. Hold the post-release `0.6.2` work (claim HTTP,
   consensus algebra, AISC 0.5 docs) until the batch boundary; then bump the
   version, rebuild, run `scripts/verify_release.py --version 0.6.2`, and push
   `main`, the tag and every public surface once.
5. After the next code commit, re-run full pytest, strict AISC batch,
   build/twine and refresh `TEST_RESULTS.md`.

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
