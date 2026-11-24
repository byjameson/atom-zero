# Real self-improve curriculum run

Date: 2026-11-24

## What ran

`atomzero-self-improve` ran on the VM with a real two-stage curriculum
(`echo_kv,echo_json`), a 100-row mixed final eval, the content/exact/key-copy
gate and automatic promotion. No dry-run and no manual promotion.

The first attempt used a 0.40 exact threshold and failed closed:

```text
status: curriculum_failed
gate:   fail
exact:  0.37
key:    0.55
reasons: content_loss_not_improved, content_ppl_regression
promotion: not promoted
```

That proved the fail-closed path. The second attempt initialized the first
stage from the best `multi_recover` adapter and used a 0.40 exact / 0.50
key-copy gate. It passed:

```text
status:      completed
gate:        pass
exact:       0.53
key copy:    0.60
promotion:   promoted: true
adapter:     checkpoints/self_improve_curriculum/echo_json
registry:    data/self_improve_curriculum_active.json
```

## What this proves

- The autonomous loop can train, evaluate, gate and promote a measured
  capability without human approval.
- The loop fails closed when the gate is not met.
- The promotion goes to a separate curriculum registry; the product default is
  still base `0.8.0`.

It does not prove general self-improvement. The adapter is an echo-stage
curriculum adapter, not a product assistant. The 3-4 key wall is handled by the
runtime copy tool, not by this adapter.

## Evidence

- `research/results/self_improve_curriculum_summary.json`
- `/tmp/self_improve_real.json` (VM local, content-free loop result)
- `data/self_improve_curriculum_active.json` (VM local)
- `checkpoints/self_improve_curriculum/echo_json` (VM local)
- `docs/CURRICULUM_LOOP_EVIDENCE.md`
- `docs/COPY_TOOL.md`
