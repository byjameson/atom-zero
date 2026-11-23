# Adapters: external training, local validation

Atom Zero consumes LoRA adapters; it does not train them. Training remains in
the separate `streamft` / `continual-lora` projects. This keeps the product
package small and lets either trainer produce an adapter without adding a
dependency to `atomzero`.

## Adapter directory

An adapter is a directory with two files:

```text
my-adapter/
  adapter.pt   # {"lora": {...}, "config": {...}}
  config.json  # {"r_attn": 0, "r_mlp": 8, "alpha_mlp": 16, ...}
```

`adapter.pt` contains only the LoRA `A`/`B` tensors. The local runtime applies
them with `atomzero.lora.apply_adapter`, so the base checkpoint stays frozen.
The default product adapter targets the transformer MLP layers (`mlp.0` and
`mlp.2`).

## Validate an adapter

```powershell
python research/adapter_check.py `
  --base checkpoints/transformer_behavior_50m_long `
  --adapter checkpoints/adapters_feedback_holdout_test/v0001
```

The command checks the base config, the adapter metadata, LoRA tensor loading
and a forward pass. It prints `"ok": true` and the trainable parameter count.
Add `--data holdout.jsonl` to score an assistant-turn holdout as well.

A tokenizer is resolved from the base directory or `data/chat_en_bpe8k.json`.
Use `--tokenizer` when the checkpoint and tokenizer live in different places.

## Compare base and adapter loss

```powershell
python research/compare_adapters.py `
  --base checkpoints/transformer_behavior_50m_long `
  --data data/feedback_holdout_test.jsonl `
  --adapters checkpoints/adapters_feedback_holdout_test/v0001
```

The score is assistant-turn cross-entropy. User turns are masked; the
`<|assistant|>` tag, the answer and `<|end|>` are scored. Lower is better.
This is a comparison tool, not a general benchmark.

Measured on the two-row holdout from the external trainer:

| model | loss | ppl | vs base |
|---|---:|---:|---:|
| base 49.2M | 3.853 | 47.14 | - |
| `adapters_feedback_holdout_test/v0001` | **3.139** | **23.09** | **-18.5%** |

This is a small-sample measurement. It shows that the external-trainer adapter
loads and behaves as expected in the Atom Zero runtime; it does not make the
model general. A different adapter (`adapters_50m/v0001`) scored 3.796
(-1.5%) on the same rows, so adapter identity and a holdout evaluation matter.

## No trainer import

The validation and comparison helpers live in
`research/adapter_runtime.py` and import only `atomzero.lora`,
`atomzero.transformer`, `torch` and `tokenizers`. The product package
`src/atomzero` does not import `streamft`.
