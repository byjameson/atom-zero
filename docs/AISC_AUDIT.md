# AISC architecture audit

Atom Zero ships four optional AISC architecture specs:

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
  store mutation.
- [`docs/aisc/atomzero-answerability.toml`](aisc/atomzero-answerability.toml):
  the measured answerability/refusal-policy experiment. It is intentionally
  retained with its failing metrics.

AISC is a separate local prototype; Atom Zero imports it only when the audit
command runs. The packaged entry point is `atomzero-audit` (also
`python -m atomzero.audit`). The checkout-local wrapper is
`python scripts/aisc_audit.py`:

```powershell
$env:AISC_PATH = "C:\path\to\ai-systems-algebra"
atomzero-audit
atomzero-audit --strict --require-calibration
atomzero-audit --json
atomzero-doctor --aisc --aisc-path "$env:AISC_PATH"
```

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
                         (HumanConfirmation precedes MemoryStore)
```

`--strict --require-calibration` is **not green**:

```text
AtomZeroReadPipeline:    FAIL, CALIBRATION_MISSING on 4 components
AtomZeroTaskActions:     FAIL, CALIBRATION_MISSING on 4 components
AtomZeroToolPlan:        FAIL, CALIBRATION_MISSING on 3 components
AtomZeroContextBudget:   FAIL, CALIBRATION_MISSING on 3 components
AtomZeroProactive:       FAIL, CALIBRATION_MISSING on 5 components
AtomZeroMemoryLifecycle: FAIL, CALIBRATION_MISSING on 4 components
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
