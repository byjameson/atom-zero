# Adapter cross-distribution cycle - in-distribution pass, cross fail

This is the first real teacher-data adapter run with an explicit
cross-distribution gate. The in-distribution candidate passed; the same adapter
failed the held-out key-value distribution. The gate correctly blocked
promotion.

## Teacher queue

```text
teacher:      Qwen2.5-1.5B-Instruct (local)
device:       CUDA for generation, CPU for training in the VM
json:         120 verified answers
keyvalue:      34 verified answers
rejected:       6
```

The format validator checks the requested name/score values. It does not prove
factual truth.

## VM run

Ubuntu 26.04 VM:

```text
SSH:          enabled
Python:       3.12 (uv-managed venv)
torch:        2.15.0+cpu
tests:        1084 passed in the VM
```

The base checkpoint and teacher fixtures were copied into the VM. The adapter
was trained inside the VM, not on the host.

## In-distribution result

```text
candidate:       ad-e89159cd2541cff9
train / held-out: 94 / 23
base loss:       6.539257
adapter loss:    1.273342
loss delta:     -5.265915
base ppl:       691.7724
adapter ppl:      3.5728
ppl ratio:        0.005165
gate:             pass / promotable
```

The autonomous self-improvement loop ran the same trainer through the adapter
action:

```text
action:          adapter_candidate_ready
trainer:         returncode 0, 33.9s
gate:            pass
weights_changed: false
```

## Cross-distribution result

```text
candidate:       ad-7b880c88d492cf32
train:           117 JSON examples
held-out:         34 key-value examples
base loss:        3.362106
adapter loss:     9.992092
loss delta:      +6.629986
base ppl:        28.8499
adapter ppl:     21852.98
ppl ratio:       757.47
gate:             fail / not promotable
reasons:          loss_not_improved, ppl_regression
```

The adapter learned the JSON syntax and catastrophically failed the key-value
syntax. This is exactly the kind of overfitting the cross gate exists to catch.

## Evidence

- `research/results/teacher_adapter_v1_gate.json`
- `research/results/teacher_adapter_v1_cross_gate.json`
- `research/results/teacher_cycle_summary.json`
- `docs/TEACHER_QUEUE.md`
- `docs/ADAPTER_TRAINING.md`

## Honest conclusion

- This is **not** evidence of general model self-improvement.
- It is evidence that the pipeline can train a real adapter and that the
  cross-distribution gate rejects a narrow format overfit.
- No weights were promoted; `0.8.0` remains the product base.
- The next attempt needs a more diverse teacher queue, cross-distribution
  examples or replay in training, and a true held-out task family rather than
  another name/score template.
