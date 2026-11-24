# Diverse teacher + replay cycle - better, still not cross-general

This cycle replaces the single JSON template with multiple teacher families,
adds replay from the previous JSON/key-value queue, and retrains in the VM.
In-distribution improves; cross-family still fails, but the regression is much
smaller than the first attempt.

## Teacher queue

```text
teacher: Qwen2.5-1.5B-Instruct
requested: 210
accepted: 194
families:
  person_json  40
  product_json 40
  person_kv    34
  product_kv   32
  city_json    25
  city_kv      23
```

Training mix (with replay):

```text
new multi-family rows: 117
old JSON replay:       120
old key-value replay:   34
unique train rows:      264
in-distribution eval:    28
held-out city family:    48
```

The held-out city family is never mixed into training.

## In-distribution gate

```text
base_loss:       5.394876
adapter_loss:    4.940428
loss_delta:     -0.454448
base_ppl:      220.2747
adapter_ppl:   139.8301
ppl_ratio:       0.634798
gate:            pass / promotable
```

## Cross-family gate

```text
candidate:       ad-28eed8a2a3024bdc
base_loss:       5.254859
adapter_loss:    7.058620
loss_delta:     +1.803761
base_ppl:      191.4946
adapter_ppl:  1162.8394
ppl_ratio:       6.072441
gate:            fail / not promotable
reasons:         loss_not_improved, ppl_regression
```

Compared with the first cross attempt:

```text
first JSON-only adapter:   ppl ratio 757x, loss 3.36 -> 9.99
diverse + replay adapter:  ppl ratio 6.07x, loss 5.25 -> 7.06
```

Replay and multi-format training reduced catastrophic forgetting by two orders
of magnitude, but the adapter still does not transfer to the unseen city family.

## Autonomous run

The loop was run in the VM with explicit train/eval splits:

```text
status:          adapter_gate_failed
candidate:       ad-e18b7a03ddd172b3
trainer:         returncode 0, 55.0s
gate:            fail
loss_delta:      +1.803761
ppl_ratio:       6.072441
weights_changed: false
ledger:          data/diverse_autonomous/ledger.jsonl
```

The gate correctly blocked promotion.

## Evidence

- `research/results/diverse_cycle_summary.json`
- `research/results/diverse_adapter_v1_indist_gate.json`
- `research/results/diverse_adapter_v1_cross_gate.json`
- `docs/TEACHER_QUEUE.md`
- `docs/ADAPTER_CROSS_CYCLE.md`

## Honest conclusion

- This is not general model self-improvement.
- Multi-family + replay is a clear improvement in retention, but not enough for
  cross-family transfer.
- The adapter was not promoted; `0.8.0` remains the product base.
- Next experiments should add replay weighting, family-balanced sampling and
  a genuinely different held-out task family (not another entity/format
  permutation).
