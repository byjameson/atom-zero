# AISC architecture audit

Atom Zero ships seventeen optional AISC architecture specs:

- [`docs/aisc/atomzero-read.toml`](aisc/atomzero-read.toml): local retrieval,
  grounded answer, citation gate and renderer.
- [`docs/aisc/atomzero-task-actions.toml`](aisc/atomzero-task-actions.toml):
  task parsing, intent evaluation, explicit human confirmation and task-store
  mutation.
- [`docs/aisc/atomzero-tool-plan.toml`](aisc/atomzero-tool-plan.toml):
  constrained tool-plan validation, explicit confirmation and bounded local
  execution.
- [`docs/aisc/atomzero-context-budget.toml`](aisc/atomzero-context-budget.toml):
  deterministic retrieval-context planning and bounded evidence.
- [`docs/aisc/atomzero-proactive.toml`](aisc/atomzero-proactive.toml):
  consent-gated, rate-limited local proactive conversation.
- [`docs/aisc/atomzero-memory-lifecycle.toml`](aisc/atomzero-memory-lifecycle.toml):
  scoped memory deletion with explicit confirmation before an irreversible
  store mutation and a content-free ledger writer after it.
- [`docs/aisc/atomzero-answerability.toml`](aisc/atomzero-answerability.toml):
  the measured answerability/refusal-policy experiment. It is intentionally
  retained with its failing metrics.
- [`docs/aisc/atomzero-answerability-generated.toml`](aisc/atomzero-answerability-generated.toml):
  the same evidence generated from the stored profile by
  `atomzero-aisc-spec`; it mirrors the failure instead of hiding it.
- [`docs/aisc/atomzero-claims.toml`](aisc/atomzero-claims.toml):
  claim extraction, evidence linking, contradiction evaluation and answer
  building. It records the lexical audit signal, not an NLI guarantee.
- [`docs/aisc/atomzero-consensus.toml`](aisc/atomzero-consensus.toml):
  claim extraction, cross-source clustering, conflict evaluation and a
  consensus report. `consensus` is lexical agreement, not truth.
- [`docs/aisc/atomzero-document-diff.toml`](aisc/atomzero-document-diff.toml):
  text loading, block differencing and a deterministic diff report.
- [`docs/aisc/atomzero-freshness.toml`](aisc/atomzero-freshness.toml):
  date extraction, fresh/aging/stale classification and a freshness report.
- [`docs/aisc/atomzero-reliability.toml`](aisc/atomzero-reliability.toml):
  content-free observation normalization, ledger writing and per-source
  summary. The ratio is an audit signal, not source truth.
- [`docs/aisc/atomzero-temporal.toml`](aisc/atomzero-temporal.toml):
  date parsing, validity evaluation and as-of filtering with explicit
  included/excluded evidence.
- [`docs/aisc/atomzero-evaluation.toml`](aisc/atomzero-evaluation.toml):
  an allowlisted evaluation gate for proposed improvements; no generic shell
  and no auto-apply.
- [`docs/aisc/atomzero-improvement.toml`](aisc/atomzero-improvement.toml):
  signal collection, prioritized human-gated planning, approval and a
  content-free plan report.
- [`docs/aisc/atomzero-scoreboard.toml`](aisc/atomzero-scoreboard.toml):
  content-free run normalization, trend summary and scoreboard report.

AISC is a separate local prototype; Atom Zero imports it only when the audit
command runs. The packaged entry point is `atomzero-audit` (also
`python -m atomzero.audit`). The checkout-local wrapper is
`python scripts/aisc_audit.py`:

```powershell
$env:AISC_PATH = "C:\path\to\ai-systems-algebra"
atomzero-audit
atomzero-audit --strict --require-calibration
atomzero-audit --json
atomzero-audit --strict --summary
atomzero-audit --coverage --aisc-path "$env:AISC_PATH"
atomzero-coverage --aisc-path "$env:AISC_PATH" --json
atomzero-doctor --aisc --aisc-path "$env:AISC_PATH"
```

`--summary` prints one aggregate line (`14 specs, 12 pass, 2 fail, 0 error`)
and exits non-zero when any spec fails. Combine it with `--json` for a small
machine-readable summary object.

`--coverage` answers a different question: are the components verified, do
their evidence strings point at files that exist, and which components have no
calibration observations? It reads evidence paths such as `tests/test_*.py`
relative to the repository root and reports missing files rather than treating
a passing check as complete coverage. Current result: **14 specs, 14 clean,
0 missing evidence files, 0 unverified components**; uncalibrated components
remain visible.

AISC 0.5 adds native batch and diff commands, which Atom Zero's multi-spec
checkout can use directly:

```powershell
cd "$env:AISC_PATH"
python -m aisc batch "$env:ATOMZERO_PATH\docs\aisc\*.toml" --strict
python -m aisc batch "$env:ATOMZERO_PATH\docs\aisc" --strict --json
python -m aisc diff old.toml new.toml --fail-on-breaking
```

`batch` expands globs and directories itself, so it works from PowerShell
without native glob expansion. `diff` reports component, metric, threshold and
issue changes; `--fail-on-breaking` exits non-zero when assurance drops, a
component is removed, a critical issue increases, or an irreversible component
is added.

`atomzero-doctor --aisc` folds the optional audit into the normal preflight. A
missing AISC checkout is a warning, not a hard install failure; a strict
warning is reported as `fail` when the audit is requested.

`--strict` treats AISC warnings as failures, matching
`python -m aisc check ... --strict`. `--require-calibration` asks AISC for the
high-assurance calibration evidence check.

## Current result

Architecture checks without calibration evidence pass for the three
deterministic paths:

```text
AtomZeroReadPipeline     assurance 0.95, verification debt 0, no irreversible
AtomZeroTaskActions      assurance 0.96, verification debt 0, 1 irreversible
                         (HumanConfirmation precedes TaskStore)
AtomZeroToolPlan         assurance 0.98, verification debt 0, 1 irreversible
                         (HumanConfirmation precedes ToolExecutor)
AtomZeroContextBudget    assurance 0.95, verification debt 0, no irreversible
AtomZeroProactive        assurance 0.95, verification debt 0, no irreversible
                         (ConsentGate precedes UserChannel)
AtomZeroMemoryLifecycle  assurance 0.98, verification debt 0, 1 irreversible
                         (HumanConfirmation precedes MemoryStore; LedgerWriter follows)
AtomZeroClaims           assurance 0.93, verification debt 0, no irreversible
AtomZeroConsensus        assurance 0.93, verification debt 0, no irreversible
AtomZeroDocumentDiff     assurance 0.93, verification debt 0, no irreversible
AtomZeroFreshness        assurance 0.96, verification debt 0, no irreversible
AtomZeroSourceReliability assurance 0.96, verification debt 0, no irreversible
AtomZeroTemporal        assurance 0.95, verification debt 0, no irreversible
AtomZeroEvaluationGate  assurance 0.95, verification debt 0, no irreversible
AtomZeroImprovementLoop assurance 0.95, verification debt 0, no irreversible
AtomZeroScoreboard     assurance 0.95, verification debt 0, no irreversible
```

`--strict --require-calibration` is **not green**:

```text
AtomZeroReadPipeline:    FAIL, CALIBRATION_MISSING on 4 components
AtomZeroTaskActions:     FAIL, CALIBRATION_MISSING on 4 components
AtomZeroToolPlan:        FAIL, CALIBRATION_MISSING on 3 components
AtomZeroContextBudget:   FAIL, CALIBRATION_MISSING on 3 components
AtomZeroProactive:       FAIL, CALIBRATION_MISSING on 5 components
AtomZeroMemoryLifecycle: FAIL, CALIBRATION_MISSING on 5 components
AtomZeroAnswerability:   FAIL, ANSWERABILITY_POOR + REFUSAL_POLICY_MISMATCH
```

The calibration warnings are expected: the specs describe deterministic test
coverage, but they do not yet carry per-component calibration observations.
The answerability spec carries the measured failure from
[`ANSWERABILITY.md`](ANSWERABILITY.md) so it cannot be mistaken for a passing
gate.

This distinction matters. The earlier `scripts/aisc_audit.py` used
`system.ok()` only, which ignores AISC warnings. The current audit makes the
strict result visible and exits non-zero under `--strict`.

## Scope and limits

- These specs cover the **critical deterministic local paths**. Optional
  free-form provider/model behavior is outside this audit.
- `assurance` values are engineering inputs, not measured probabilities.
  Calibrate them before treating them as empirical claims.
- AISC catches architecture-level issues (type mismatches, unguarded
  irreversible actions, verification debt and context budgets). It does not
  replace the Atom Zero test suite or behavioral benchmarks.
- The `aisc` package is external and optional; the repository remains
  installable and testable without it.
- The answerability spec is a measured experiment, not an operational pilot.
  Its warnings are evidence, not a reason to weaken the targets.
