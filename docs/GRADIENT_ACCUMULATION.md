# Gradient accumulation for adapter training

`research/train_adapter_candidate.py` historically performed one optimizer
step per row (batch size 1). `--batch-size N` accumulates `N` rows into one
update so noisy single-row steps can be averaged without changing the model or
the LoRA math.

Default behaviour is unchanged: `--batch-size 1` is one optimizer step per
trained row, exactly as before.

## What changed

- `--batch-size N` (default `1`, validated as an integer `>= 1`) accumulates
  `N` rows before calling `optimizer.step()`. Gradients are zeroed once per
  group, never between rows inside a group.
- The group loss is the token-weighted mean over the whole group:

  ```text
  sum(row_weighted_ce) / sum(row_token_weights)
  ```

  This is not a mean of per-row means, so a long row contributes
  proportionally more tokens. Rows skipped as too short do not consume a batch
  slot.
- `--steps` still counts sampled rows, so `--batch-size 4` performs
  `ceil(rows_seen / 4)` optimizer steps; the last group may be partial.
- The result JSON and adapter `config.json` record `batch_size`,
  `optimizer_steps`, `rows_seen`, `lr_scale` and `effective_lr`.
- `--lr-scale {none,linear}` defaults to `none`. `linear` explicitly
  multiplies the learning rate by `batch_size`; the base `--lr` is still
  recorded as `lr`. Nothing is scaled silently.

## Train with batch size 4

```powershell
python research/train_adapter_candidate.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/adapter_candidates/train.jsonl `
  --eval data/adapter_candidates/eval.jsonl `
  --out checkpoints/adapters_candidate/grad_accum_v0001 `
  --steps 2000 --batch-size 4 --lr 0.001 --rank 8 --alpha 16 `
  --result research/results/adapter_candidate_grad_accum.json `
  --comparison research/results/adapter_candidate_grad_accum_comparison.json
```

Add `--lr-scale linear` only when you deliberately want `lr * batch_size`
(here `0.004`). The default `none` keeps `0.001`.

## Resuming from a previous adapter

`--init-adapter PATH` starts training from a saved adapter directory instead
of a fresh LoRA initialization. Staged curricula (echo -> single-key ->
multi-key) can now chain stages: each stage continues from the previous stage's
A/B tensors.

```powershell
python research/train_adapter_candidate.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/adapter_candidates/stage2_train.jsonl `
  --eval data/adapter_candidates/stage2_eval.jsonl `
  --out checkpoints/adapters_candidate/stage2 `
  --init-adapter checkpoints/adapters_candidate/stage1 `
  --steps 2000 --batch-size 4 --lr 0.001 `
  --result research/results/adapter_candidate_stage2.json `
  --comparison research/results/adapter_candidate_stage2_comparison.json
```

- LoRA geometry (`r_attn`, `alpha_attn`, `r_mlp`, `alpha_mlp`) is read from
  `PATH/config.json`, falling back to the `adapter.pt` config, and rebuilt
  exactly.
- Omit `--rank`, `--alpha`, `--rank-attn` and `--alpha-attn` to inherit all
  four values. If supplied, they must match the checkpoint exactly; a mismatch
  exits with an error naming the conflicting flag, the supplied value and the
  adapter value.
- A/B tensors are loaded strictly: missing, unexpected or wrongly shaped
  tensors raise instead of being silently dropped.
- The result JSON and the new adapter's `config.json` record `init_adapter`.
- Without `--init-adapter`, training is unchanged: fresh LoRA initialization
  with the same defaults and the same result/adapter config fields.

## Key/value token weighting

`--key-weight` and `--value-weight` (both finite and `>= 0`) weight the
assistant answer tokens that overlap a key name or a value. Spans come from
`research.adapter_runtime.answer_spans`, so JSON objects, `k=v;k=v` KV lines
and `k: v` YAML lines are recognised. Plain-text answers have no spans.

Token categories and their weights:

| token | weight |
|---|---:|
| user turn / ignored label (`-100`) | 0 |
| overlaps a key-name span | `--key-weight` |
| overlaps a value span | `--value-weight` |
| other alphanumeric token | `--content-weight` |
| syntax token | 1.0 |

`--key-weight` defaults to `1.0`. `--value-weight` defaults to the effective
`--content-weight`. When both key and value weights are `1.0`, the trainer
calls the legacy `_token_weights` path and the weights are bitwise identical to
the previous release. A tokenizer without usable offsets also falls back to
content-only weighting. The weighted loss is unchanged:
`sum(ce * weights) / sum(weights)`.

The result JSON and adapter `config.json` record `key_weight`, `value_weight`
and `content_weight`.

```powershell
python research/train_adapter_candidate.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/adapter_candidates/train.jsonl `
  --eval data/adapter_candidates/eval.jsonl `
  --out checkpoints/adapters_candidate/key_weighted_v0001 `
  --steps 2000 --batch-size 4 --lr 0.001 --rank 8 --alpha 16 `
  --key-weight 10 --value-weight 5 `
  --result research/results/adapter_candidate_key_weighted.json `
  --comparison research/results/adapter_candidate_key_weighted_comparison.json
```
