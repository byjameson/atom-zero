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

## Replay-weighted retry - first cross-family pass

The replay sampler was then used explicitly:

```text
primary rows:      117 multi-family
replay rows:       154 old JSON/key-value
replay weight:     0.35
steps:             600
```

Cross-family result:

```text
candidate:       ad-e396366ef725656c
family separation: true
base_loss:       5.254859
adapter_loss:    2.802659
loss_delta:     -2.452201
base_ppl:      191.4946
adapter_ppl:    16.4884
ppl_ratio:       0.086104
gate:            pass / promotable
```

The autonomous loop executed the same replay-aware trainer:

```text
status:          completed
trainer:         returncode 0, 64.4s
candidate:       ad-e396366ef725656c
train/eval:      116 / 48
family separation: true
gate:            pass
weights_changed: false
ledger:          data/diverse_autonomous_replay/ledger.jsonl
```

On the in-distribution held-out set the same adapter reached:

```text
base_loss: 5.394876 -> adapter_loss: 0.957432
base_ppl: 220.2747  -> adapter_ppl:  2.6050
ppl_ratio: 0.011826
gate: pass
```

This is the first cross-family **full-loss** gate pass in the project: the
held-out city family never appears in training, and the replay mix retains the
older formats. It is not yet a content-transfer pass.

## Content audit - promotion blocked

A content-token audit was added after the full-loss pass:

```text
baseline content loss: 4.432720
adapter content loss:  4.412256
content delta:        -0.020464
required delta:       -0.05
content ppl ratio:     0.979744
gate:                  fail / not promotable
```

The full-loss gain is almost entirely syntax/format learning. The active
adapter registry was removed; no candidate is promoted.

## Evidence

- `research/results/diverse_cycle_summary.json`
- `research/results/replay_cross_pass.json`
- `research/results/replay_content_audit.json`
- `research/results/diverse_adapter_v1_indist_gate.json`
- `research/results/diverse_adapter_v1_cross_gate.json`
- `docs/TEACHER_QUEUE.md`
- `docs/ADAPTER_CROSS_CYCLE.md`

## Honest conclusion

- This is not general model self-improvement.
- The adapter learned format transfer and retained the older JSON/key-value
  distributions; the held-out city family is new.
- The replay mix contains the old name/score key-value distribution, so format
  transfer is not the same as learning an unseen task family from scratch.
- The adapter was not promoted; `0.8.0` remains the product base.
- Next experiments should test an entirely new task family (YAML/arithmetic),
  family-balanced replay weights, and a true deployment smoke test before any
  promotion.
