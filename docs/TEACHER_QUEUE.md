# Real teacher queue and cross-distribution fixture

This fixture prepares the next adapter experiment with a real local teacher
instead of hand-written JSON data.

## Teacher

```text
model:  Qwen2.5-1.5B-Instruct (local)
device: CUDA
seed:   11
```

The teacher answers are accepted only when a deterministic format validator
passes. This validates the format, not the truth of the answer.

## Queue

Requested:

```text
json:      120
keyvalue:   40
```

Accepted:

```text
json:      120
keyvalue:   34
rejected:    6
```

The CSV format is supported by the validator but disabled by default: the
1.5B teacher often invented a different row instead of copying the requested
values. JSON and key-value formats were reliable.

## In-distribution candidate

```text
examples:  120 json teacher answers
unique:    117
train:      94
held-out:   23
candidate:  ad-e89159cd2541cff9
base loss:  6.5393
base ppl:   691.77
```

## Cross-distribution candidate

```text
train:      117 json teacher answers
held-out:    34 key-value teacher answers
candidate:  ad-7b880c88d492cf32
base loss:  3.3621
base ppl:   28.85
```

The key-value distribution is not mixed into training. The cross candidate is
intentionally harder: a model that learned JSON formatting must transfer the
same name/score mapping to a different syntax.

## Reproduce

```powershell
python research/build_teacher_queue.py `
  --json-count 120 --keyvalue-count 40 `
  --out data/teacher_queue.jsonl

python -m atomzero.adapter_candidates export `
  --examples-jsonl data/teacher_json.jsonl `
  --out-dir data/teacher_candidate --eval-ratio 0.2 --seed 13

python -m atomzero.adapter_candidates export-cross `
  --train-jsonl data/teacher_json.jsonl `
  --eval-jsonl data/teacher_keyvalue.jsonl `
  --out-dir data/teacher_cross_candidate --seed 13
```

## Measured result

The first VM-run experiment trained on the JSON split:

```text
in-distribution gate:  pass   (loss 6.5393 -> 1.2733, ppl 691.77 -> 3.57)
cross key-value gate:  fail   (loss 3.3621 -> 9.9921, ppl ratio 757.47)
```

The adapter overfit JSON formatting and did not transfer to key-value. The
cross gate blocked promotion. Full evidence: `docs/ADAPTER_CROSS_CYCLE.md`.

## Honest boundary

- The teacher is Qwen2.5-1.5B, not a frontier model.
- The validator checks the requested format and values; it does not prove
  factual correctness.
- Both distributions are synthetic name/score prompts. A cross pass would show
  format transfer, not general intelligence.
- Training and gate runs are not part of this fixture preparation.
- The next run must train on the training split and evaluate both the
  in-distribution and cross-distribution held-out sets.
