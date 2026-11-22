# AISC architecture audit

Atom Zero has two optional AISC architecture specs:

- [`docs/aisc/atomzero-read.toml`](aisc/atomzero-read.toml): local retrieval,
  grounded answer, citation gate and renderer.
- [`docs/aisc/atomzero-task-actions.toml`](aisc/atomzero-task-actions.toml):
  task parsing, intent evaluation, explicit human confirmation and task-store
  mutation.

AISC is a separate local prototype, not an `atomzero` runtime dependency. Run
the audit only when its checkout or installed package is available:

```powershell
$env:AISC_PATH = "C:\path\to\ai-systems-algebra"
python scripts/aisc_audit.py
```

or explicitly:

```powershell
python scripts/aisc_audit.py `
  --aisc-path "C:\path\to\ai-systems-algebra" --json
```

Current local result:

```text
AtomZeroReadPipeline
Topology: sequential
Assurance bound: 0.95 (target 0.90)
Verification debt: 0
Irreversible components: 0
Result: PASS

AtomZeroTaskActions
Topology: sequential
Assurance bound: 0.96 (target 0.90)
Verification debt: 0
Irreversible components: 1
Result: PASS
```

The task spec places `HumanConfirmation` before the irreversible
`TaskStore`, matching Atom Zero's confirmed deletion flow.

## Scope and limits

- These specs cover the **critical deterministic local paths**. Optional
  free-form provider/model behavior is outside this audit and needs a separate
  calibrated spec.
- `assurance` values are engineering inputs, not measured probabilities.
  Calibrate them before treating them as empirical claims.
- AISC catches architecture-level issues (type mismatches, unguarded
  irreversible actions, verification debt and context budgets). It does not
  replace the Atom Zero test suite or behavioral benchmarks.
- The `aisc` package is external and optional; the repository remains
  installable and testable without it.
