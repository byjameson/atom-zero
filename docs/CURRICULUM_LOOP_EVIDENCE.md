# Curriculum loop evidence

Date: 2026-11-24

## What was tested

`atomzero-self-improve` was run on the VM with a curriculum command backed by
`research/run_staged_curriculum.py` in dry-run mode. This exercises the real
self-improve -> curriculum -> gate -> ledger path without retraining a model.

## Result

```text
status:      completed
curriculum:
  status:    pass
  returncode: 0
  elapsed:   0.106s
  gate:      pass
  exact:     0.40
  key copy:  0.548
  promotion: requested, not promoted (dry_run)
  reasons:   []
```

The dry-run adapter metrics are fake by design; the point is that the loop
propagates the gate and promotion result and records only aggregate values. A
real curriculum run uses the same path with `--dry-run` removed.

## Fail-closed test

The first attempt used a shell-style command prefix (`PYTHONPATH=src ...`).
The loop could not start the child process and returned:

```text
status: curriculum_failed
curriculum.status: error
reasons: ["runner_failed"]
error: "curriculum runner could not start (FileNotFoundError)"
```

That is the intended fail-closed behavior: a broken curriculum command does not
claim success and does not write a registry.

## Implication

The autonomous loop can now carry a measured capability-growth job end to end.
It still depends on the curriculum runner and the behavior gate; it is not
general self-improvement.

## Evidence

- VM command output (local `/tmp/self_improve_curriculum.json`)
- `docs/AUTONOMOUS.md`
- `docs/STAGED_COPY_CURRICULUM.md`
