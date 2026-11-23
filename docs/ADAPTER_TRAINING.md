# Adapter training cycle 2 - real LoRA on the 0.8.0 base

This cycle trains an actual LoRA adapter on the local 49.2M `0.8.0` base
checkpoint, evaluates it with `research/compare_adapters.py`, and feeds the
result into the held-out gate.

It is a narrow, synthetic JSON-format skill, not a general capability claim.

## Reproduce

```powershell
python research/make_adapter_skill.py `
  --out data/adapter_skill_verified.jsonl

atomzero-adapter-candidate export `
  --examples-jsonl data/adapter_skill_verified.jsonl `
  --out-dir data/adapter_candidates --eval-ratio 0.2 --seed 13

python research/train_adapter_candidate.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/adapter_candidates/train.jsonl `
  --eval data/adapter_candidates/eval.jsonl `
  --out checkpoints/adapters_candidate/v0002 `
  --steps 300 --lr 0.001 --rank 8 --alpha 16 `
  --result research/results/adapter_candidate_v2.json `
  --comparison research/results/adapter_candidate_v2_comparison.json

atomzero-adapter-candidate evaluate `
  --candidate-json data/adapter_candidates/candidate.json `
  --comparison-json research/results/adapter_candidate_v2_comparison.json --json
```

## Data and split

```text
fixture examples:       320
unique after dedupe:    301
train:                  241
held-out:                60
candidate_id:           ad-10612cfe5f28b728
```

The fixture is a deterministic JSON-format skill:

```text
user:      Return only JSON for name=Ada and score=42.
assistant: {"name": "Ada", "score": 42}
```

## Training

```text
base:              atom-zero-49m (0.8.0)
device:            CUDA
LoRA target:       MLP layers (mlp.0, mlp.2)
rank / alpha:      8 / 16
trainable params:  368,640
steps:             300
lr:                0.001
train loss (tail): 0.6416
```

## Held-out result

```text
baseline loss:     5.994060
adapter loss:      0.794874
loss delta:       -5.199186
loss relative:    -86.74%
baseline ppl:     401.0395
adapter ppl:       2.2142
ppl ratio:         0.005521
eval examples:     60
```

`research/adapter_check.py` reported `ok: true` for the saved adapter and the
same loss delta.

## Gate

```text
status:       pass
promotable:   true
reasons:      loss_improved, ppl_not_worse
auto_apply:   false
weights:      false
```

Content-free evidence:

- `research/results/adapter_candidate_v2.json`
- `research/results/adapter_candidate_v2_gate.json`
- `research/results/adapter_candidate_v2_candidate.json`

The adapter itself is local at `checkpoints/adapters_candidate/v0002` and is
gitignored.

## Autonomous run

The same trainer was invoked by the autonomous self-improvement action:

```text
action:            adapter_candidate_ready
planner signal:    adapter_candidate_ready (priority 68)
trainer:           research/train_adapter_candidate.py via sandbox
trainer returncode: 0
trainer elapsed:   49.0s
candidate:         ad-10612cfe5f28b728
train/eval:        241 / 60
gate:              pass / promotable
loss delta:        -5.199186
ppl ratio:         0.005521
weights_changed:   false
```

The loop command used `atomzero-self-improve --opt-in` with
`--adapter-trainer-command`, `--adapter-base`, `--adapter-examples-jsonl`,
`--adapter-comparison-out` and `--adapter-ledger`. No human approval was needed;
the external trainer ran in the sandbox and the held-out gate decided.

## Honest boundary

- This is a **narrow synthetic format skill**, not evidence of general model
  improvement. The base was already poor at this exact JSON format; the adapter
  learned the format.
- The held-out rows come from the same synthetic distribution. A different
  distribution or a real teacher queue could fail the gate.
- The base `0.8.0` weights were not changed. The adapter is a separate local
  artifact and is not activated by the product.
- The trainer is a research harness, not part of the `atomzero` package.
- Promotion still requires a separate decision; `promotable` is a gate result,
  not a deployment.
- The real base checkpoint came from the local Hugging Face cache, not a
  newly trained checkpoint.
