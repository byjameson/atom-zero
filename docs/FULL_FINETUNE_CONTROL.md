# Full fine-tune copy control (research only)

Date: 2026-11-24

## Purpose

The MLP-only and `out_proj` + MLP LoRA adapters on the copy curriculum lower
content-token loss but still produce 0/200 exact answers. Their generations
copy values and entities while replacing held-out prompt keys with training
key names.

Before collecting more LoRA data, this control asks one narrow question:

> Can the 49M base architecture learn the copy task at all when its own
> weights are allowed to change?

This is a **research control**, not a product adapter. It is never written to
`data/active_adapter.json`, never promoted, and never used as a product
default.

## What the script does

`research/finetune_copy_control.py`:

1. loads the base checkpoint with the same
   `research.adapter_runtime.load_model` helper and row format used by
   `research/train_adapter_candidate.py`;
2. trains all parameters, or a frozen subset, with AdamW;
3. uses assistant-turn cross-entropy with the same optional content-token
   weighting (default `1.0`);
4. clips gradient norm at `1.0`;
5. saves a full research checkpoint;
6. evaluates base vs control on `--eval` with
   `research.compare_adapters.compare`, including per-family metrics when the
   JSONL rows carry a `family` label and normalized exact match.

`--batch-size` is the number of rows per optimizer step. Rows are encoded
individually and their weighted losses are averaged; no padding is introduced.
The stream is reshuffled every epoch with `--seed`.

### Freeze modes

| mode | frozen parameters |
|---|---|
| `none` | none; full fine-tune |
| `embeddings` | `tok.weight` and `pos.weight` |
| `first-half` | embeddings plus the first `layers // 2` transformer blocks |

The LM head is tied to `tok.weight`. Freezing embeddings therefore also
freezes the output projection; the result records this through the parameter
counts and `frozen_groups`.

## Exact VM command

The command below is a **small-step smoke control** on the copy curriculum:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/finetune_copy_control.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/copy_curriculum_v1/train.jsonl `
  --eval data/copy_curriculum_v1/cross.jsonl `
  --out checkpoints/fullft_copy_control/v0001 `
  --steps 100 `
  --lr 1e-4 `
  --batch-size 4 `
  --max-seq 256 `
  --content-weight 5.0 `
  --seed 23 `
  --freeze none `
  --result research/results/full_finetune_control.json `
  --comparison research/results/full_finetune_control_comparison.json
```

`--content-weight 5.0` matches the earlier LoRA copy-curriculum run so the
control is comparable. Drop the flag to use the neutral default of `1.0`.
`data/copy_curriculum_v1/` is local and gitignored.

A longer run (`--steps 1000`, the default) is needed before treating the
result as evidence. The earlier LoRA adapters used 2,000 steps. A useful
follow-up is `--freeze first-half`, which tests whether the upper layers alone
can learn the task.

## Checkpoint layout

```text
checkpoints/fullft_copy_control/v0001/
  model.pt              # config, state, research metadata
  config.json           # architecture + research flags + training summary
  chat_en_bpe8k.json    # tokenizer copy
```

`model.pt` has the same `config` / `state` shape as the base checkpoint, so
`load_model(base, checkpoint_dir)` can load it directly. The runtime treats a
directory containing `model.pt` as a full research checkpoint, not as a LoRA
adapter.

The checkpoint is gitignored and is not a product artifact.

## Result JSON

The result file contains aggregate metrics only:

```text
research_control:  true
product_weights:   false
weights_changed:   true
```

`weights_changed` refers to the **research checkpoint** differing from the
base after training. It does not mean any product weight changed.

Metrics include:

- `base_loss`, `control_loss`, `loss_delta`, `loss_relative`;
- `base_content_loss`, `control_content_loss`, `content_loss_delta`;
- `base_syntax_loss`, `control_syntax_loss`, `syntax_loss_delta`;
- `base_exact_match`, `control_exact_match`, `exact_match_delta`;
- `base_metrics` / `control_metrics` with full aggregate detail, including
  exact-match counts and per-family exact-match rates;
- `families` with baseline/control loss and content loss per labelled family;
- `train` with loss, steps, batch size, freeze mode and parameter counts.

The comparison JSON is stripped of every per-row `user` / `assistant` text
before it is written. No prompt text is committed. Generated result and
comparison JSONs stay local evidence.

## Expected interpretation

Read the held-out `cross` result, not the training loss:

1. **Full fine-tune reaches nonzero exact match on unseen keys.** The base
   architecture can support the copy operation. The earlier failure then
   points at the LoRA locus/capacity or its training setup, not at an
   architectural inability or a pure data limit.
2. **Content loss improves but exact match stays 0/200.** The value-copy
   behaviour persists even with all weights trainable. This is a strong
   data/objective signal, but check the loss curve and run longer before
   calling it an architecture limit.
3. **Full fine-tune does not improve content loss over base.** This is the
   clearest base/data-limit signal. Adding more LoRA data is unlikely to help
   until the optimization or data problem is understood.

Two cautions:

- A 100-step run is a smoke test, not a conclusion. Use 1,000+ steps and
  more than one seed for a research claim.
- Full fine-tuning can memorize training key names too. The question is
  whether exact match improves on the held-out **cross** families with unseen
  keys. Per-family exact match is the signal, not training loss.

## Limits

- This control does not prove the base can generalize to arbitrary key-copy
  tasks; it only tests this curriculum and split.
- It does not produce a promotable adapter and does not change product
  defaults.
- A full checkpoint is much larger than a LoRA adapter and is not intended
  for distribution.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_finetune_control.py -q
```

The tests use a tiny synthetic transformer and tokenizer. They cover one
training step changing weights, checkpoint roundtrip, research flags in the
result JSON, prompt-free comparison output, and the three freeze modes.
