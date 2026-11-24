# Fused QKV attention LoRA (research experiment)

This document describes an **experiment**, not a capability result. The tiny
`TransformerLM` stores Q, K and V in one fused
`nn.MultiheadAttention.in_proj_weight` Parameter. The existing adapter path
wraps `nn.Linear` modules only: MLP layers plus the attention output
projection (`out_proj`). Q/K/V stay frozen. QKV LoRA adds an optional adapter
on the fused projection itself so the experiment can test whether moving the
adaptation locus into attention changes the 3-4 key copy wall.

Do not read this as evidence of general reasoning, a new architecture result,
or "ASI". The only valid claim is a measured before/after on a named eval
split.

## What is adapted

`LoRAMultiheadAttention` (in `atomzero.lora`) wraps, but does not modify, the
base `nn.MultiheadAttention`:

```text
base QKV:   in_proj_weight                  (3 * dim, dim), frozen
adapter A:  lora_A                          (r, dim)
adapter B:  lora_B                          (3 * dim, r), zero-initialized
merged:     in_proj_weight + (B @ A) * (alpha / r)
```

The wrapper calls `F.multi_head_attention_forward` with the merged weight and
the base `out_proj`, `bias_k`, `bias_v`, `add_zero_attn`, dropout, masks,
`need_weights`, `average_attn_weights` and `is_causal` arguments. Because `B`
starts at zero, a freshly applied QKV adapter produces the exact base forward
output. Only A and B are trainable; the base QKV weight has
`requires_grad=False` and receives no gradient.

An optional `adapt_bias=True` adds a rank-`r` delta to `in_proj_bias`
(`lora_bias_A` shape `(r, 1)`, `lora_bias_B` shape `(3 * dim, r)`). The
trainer does not expose a bias flag; the default experiment adapts the weight
only. Bias support exists for focused experiments and is recorded as
`qkv_bias` in a saved adapter config.

## Parameter count

For `layers` blocks, embedding width `dim` and QKV rank `r`, the weight-only
adapter adds:

```text
layers * (r * dim + 3 * dim * r) = layers * 4 * r * dim
```

For the local `atom-zero-49m` base (`dim=768`, `layers=6`):

| QKV rank | QKV parameters | with MLP rank 8 (368,640) | with MLP + out_proj (405,504) |
|---:|---:|---:|---:|
| 1 | 18,432 | 387,072 | 423,936 |
| 2 | 36,864 | 405,504 | 442,368 |
| 4 | 73,728 | 442,368 | 479,232 |
| 8 | 147,456 | 516,096 | 552,960 |

The MLP/out_proj counts are from the existing 49m adapter geometry. These are
trainable-parameter counts, not a memory or compute claim: the merged QKV
weight is formed on every forward pass.

## Trainer flags

```text
--qkv-rank R    fused QKV LoRA rank, default 0 (off)
--qkv-alpha A   fused QKV LoRA alpha, default 0
```

`--qkv-alpha` must be `> 0` when `--qkv-rank > 0`. With both at 0 the trainer
takes the previous code path exactly: no QKV wrapper is created, the model
forward is unchanged, and saved adapters keep their old four-key geometry.
The result JSON records `rank_qkv` and `alpha_qkv`; the adapter config records
`r_qkv`, `alpha_qkv` and `qkv_bias`, so `apply_adapter` can rebuild the same
modules.

## Exact experiment command

Generate or reuse the staged copy curriculum first
(`docs/STAGED_COPY_CURRICULUM.md`). This command trains on the mixed 2-4 pair
stage, which is the closest available 3-4 key split:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/train_adapter_candidate.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/staged_copy_v1/stage_multi/train.jsonl `
  --eval data/staged_copy_v1/stage_multi/eval.jsonl `
  --out checkpoints/adapters_qkv/v0001_r4 `
  --steps 2000 --batch-size 4 --lr 3e-4 `
  --rank 8 --alpha 8 `
  --rank-attn 4 --alpha-attn 8 `
  --qkv-rank 4 --qkv-alpha 8 `
  --replay-jsonl data/staged_copy_v1/stage_two_json/train.jsonl `
  --replay-weight 0.2 `
  --content-weight 2 --key-weight 3 --value-weight 2 `
  --seed 23 `
  --result research/results/adapter_qkv_r4.json `
  --comparison research/results/adapter_qkv_r4_comparison.json
```

Run the same command with `--qkv-rank 0 --qkv-alpha 0` as the control. The
control must differ only in the QKV flags; otherwise the comparison confounds
adapter locus with learning rate, data mix or seed.

Then evaluate on every relevant split, not only the training stage:

```powershell
python research/compare_adapters.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --data data/staged_copy_v1/stage_multi/eval.jsonl `
  --adapters checkpoints/adapters_qkv/v0001_r4 `
  --per-family --exact-match --key-copy `
  --out research/results/adapter_qkv_r4_multi.json
```

Repeat for `stage_two_json/eval.jsonl`, `stage_echo_json/eval.jsonl` and the
mixed final eval. Report per-family exact match and key-copy rate; aggregate
loss alone can hide a broken or unchanged copy circuit.

## Controlled multi-seed capacity ablation

One from-base QKV run scored 6% exact / 18% key copy on the multi-short split.
That is a single anecdote, not a locus result. The runner
`research/run_qkv_capacity_ablation.py` performs the controlled A/B: the same
seeds, data order, steps, batch size, learning rate, loss weights, MLP
geometry and `out_proj` geometry for both arms. The only planned difference
is the fused QKV LoRA rank/alpha:

| arm | MLP | `out_proj` | fused QKV |
|---|---|---|---|
| `qkv4` | rank 8 / alpha 8 | rank 4 / alpha 8 | rank 4 / alpha 8 |
| `mlp` | rank 8 / alpha 8 | rank 4 / alpha 8 | off (rank 0) |

`mlp` is the no-QKV control: it keeps the MLP and `out_proj` adapters. The
name means the fused QKV projection is frozen, not that attention is
untouched. The command uses the 3-key split because 4-key exact match is at a
0% ceiling for both arms; run `stage_4short/eval.jsonl` as the harder check
and `mix_eval50.jsonl` for transfer. Train on the matching stage file for a
locus test, or on `mix_train.jsonl` for transfer across 2-4 keys; both arms
must always use the same train file.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/run_qkv_capacity_ablation.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/multi_short_v1/stage_3short/train.jsonl `
  --eval data/multi_short_v1/stage_3short/eval.jsonl `
  --arms qkv4,mlp `
  --seeds 23,29,37 `
  --steps 4000 --batch-size 4 --lr 3e-4 `
  --content-weight 2 --key-weight 3 --value-weight 2 `
  --out checkpoints/qkv_capacity_ablation `
  --result research/results/qkv_capacity_ablation.json
```

Use `--dry-run` to print all six trainer/evaluator commands without touching
data or checkpoints. Use `--max-eval-rows N` to score a deterministic
hash-selected subset; the subset is materialized once in a temporary
directory, shared by every arm and seed, then deleted. Per-row comparison
payloads also go to that temporary directory, so the result never stores
prompt or answer text.

For every arm x seed the runner records exact match, key-copy rate, key
recall, eval loss, content loss, train loss, LoRA parameter count, return
codes and elapsed seconds. The JSON (`--result`) and sibling Markdown report
contain per-seed rows, per-arm mean/sample-std/min/max, and a 95% t-interval
over seeds. A claim is reported only when the intervals for both exact match
and key copy do not overlap and point in the same direction. Overlap is
reported as `inconclusive`, which is not evidence of no effect; with three
seeds the design detects only large effects.

QKV rank 4 adds 73,728 trainable parameters, so a QKV win can be a capacity
effect rather than an attention-locus effect. The follow-up control is an MLP
widening matched to the QKV parameter count, run through the same seeds and
data.

## What to measure

1. **Control vs QKV on the same rows and seed.** Use the same MLP/out_proj
   geometry, data order and hyperparameters. A QKV run with a different LR is
   not a locus experiment.
2. **Key copy separately from value copy.** The current failure mode is value
   copy without key copy (`docs/COPY_CURRICULUM_CYCLE.md`). Exact match and
   key-copy rate are the primary metrics; content loss is secondary.
3. **Two or more QKV ranks.** A single rank cannot distinguish "attention
   locus helps" from "more parameters help". Try at least rank 2 and rank 4,
   and compare against an equally sized MLP capacity increase if possible.
4. **Forgetting checks.** Evaluate the earlier stages and a replay/mixed split
   after QKV training. Attention changes can damage syntax or previously
   learned stages.
5. **Generation inspection.** Read held-out generations. A lower loss with
   the same key-substitution behavior is not evidence of a copy circuit.

## Limitations and caveats

- **Research only.** The adapter is a local artifact. Nothing here changes the
  product weights, the active adapter registry or any release.
- **No fast path.** `nn.MultiheadAttention.forward` can use
  `torch._native_multi_head_attention` when its weights are static. The merged
  QKV weight carries a gradient, so this wrapper always uses the functional
  path. Expect slower training/inference than the unadapted base and do not
  compare wall-clock numbers across the two paths.
- **Fused MHA only.** `apply_qkv_lora` targets `nn.MultiheadAttention` with
  `_qkv_same_embed_dim`. Hugging Face models with separate `q_proj`, `k_proj`
  and `v_proj` linears are not covered by this wrapper; the existing
  `LoRALinear` path already handles those names.
- **Adapter soup does not merge QKV tensors yet.** `merge_adapters` accepts
  only `.A`/`.B` tensors and will reject `lora_A`/`lora_B` QKV names with an
  explicit unsupported-name error. Do not assume QKV adapters can be souped.
- **`merge_lora` is one-way.** It can fold weight and bias deltas into the base
  MHA and zero the adapter, but a folded QKV base no longer round-trips through
  an adapter directory. Prefer `save_adapter` + `apply_adapter`.
- **One seed is an anecdote.** The 3-4 key wall is a noisy, small-data regime.
  Repeat across seeds and report the distribution, not the best run.
- **Attention LoRA can change syntax before it changes copying.** Inspect
  generations and syntax loss; do not attribute a lower aggregate loss to key
  copying without the key-copy metric.
- **No ASI claim.** A successful run would be evidence about this synthetic
  transcription curriculum and this tiny model, nothing more.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_qkv_lora.py tests/test_lora_attn.py -q
```

Coverage: identity-at-initialization forward, QKV A/B gradients, frozen base
QKV weight, save -> fresh base -> `apply_adapter` bitwise roundtrip, optional
bias adapter roundtrip, rank/alpha validation and trainer flag recording.
