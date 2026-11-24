# Adapter soup: weighted adapter merging

Adapter soup combines several LoRA adapters without retraining. It is useful
after a sequential curriculum: `echo_json` reached 72% exact / 82% key copy
after focus rehearsal and `two_json` reached 28% / 46%, but earlier skills are
still forgotten by later stages. Merging adapters may recover both skills; it
may also cancel them. This document describes the tool and, more importantly,
the measurement needed before believing a merged adapter.

The implementation lives in `atomzero.lora` (`merge_adapters`,
`merge_adapter_state_dicts`, `normalize_adapter_weights`) and the CLI is
`research/merge_adapters.py`. It never modifies the input adapters, the base
checkpoint or the active registry.

Fused-QKV attention adapters (`docs/QKV_LORA.md`) are **not** soup-compatible
yet: their `lora_A`/`lora_B` tensors are rejected by the `.A`/`.B` merge
contract instead of being silently dropped.

## Method

All adapters must share the same LoRA geometry and the same A/B tensor names
and shapes. The merge refuses otherwise because mixing different ranks or
alphas would silently change the effective LoRA scaling.

Weights are normalized to sum to one. Negative, non-finite or all-zero weights
are rejected; missing weights mean equal weighting. A zero weight skips that
adapter and the rest are renormalized.

### `ab` mode (default)

For adapters `i` with normalized weights `w_i`:

```text
A_merged = sum_i w_i * A_i
B_merged = sum_i w_i * B_i
```

The rank and alpha stay the same, so the merged adapter loads with the original
geometry. The forward delta is `B_merged @ A_merged`, which is **not** the
weighted average of the individual deltas:

```text
(sum_i w_i B_i) @ (sum_i w_i A_i) != sum_i w_i (B_i @ A_i)
```

The cross terms can reinforce or cancel skills. This is why `ab` is a cheap
starting point, not a guaranteed merge.

### `delta-svd` mode

For each module:

```text
D = sum_i w_i * (B_i @ A_i)
D ~= U_k S_k V_k^T
B_merged = U_k sqrt(S_k)
A_merged = sqrt(S_k) V_k^T
```

`--rank` caps the new rank. Without `--rank`, the source rank is used. A larger
cap can preserve the full weighted delta when the sum of source deltas has
higher rank, at the cost of more parameters; the output config records the new
rank. Alpha is rescaled so the effective `alpha / rank` scaling is preserved.
With a tight cap, the discarded singular directions are still lost, so this
mode is also approximate.

## Exact command

The command below assumes the rehearsal run saved the three adapters under
`checkpoints/staged_copy_rehearsal/`:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/merge_adapters.py `
  --adapters checkpoints/staged_copy_rehearsal/rehearsal_focus `
             checkpoints/staged_copy_rehearsal/echo_json `
             checkpoints/staged_copy_rehearsal/two_json `
  --weights 1 1 1 `
  --out checkpoints/adapter_soup/focus_echo_two `
  --base checkpoints/transformer_behavior_50m_long `
  --eval data/staged_copy_v1/stage_two_json/eval.jsonl `
  --result research/results/adapter_soup_focus_echo_two.json `
  --comparison research/results/adapter_soup_focus_echo_two_comparison.json `
  --per-family --exact-match --key-copy
```

The paths are the only run-specific part; the adapter directories must contain
`adapter.pt` and `config.json` written by `save_adapter` (or an equivalent
trainer). To test a different rank cap:

```powershell
python research/merge_adapters.py `
  --adapters checkpoints/staged_copy_rehearsal/rehearsal_focus `
             checkpoints/staged_copy_rehearsal/echo_json `
             checkpoints/staged_copy_rehearsal/two_json `
  --out checkpoints/adapter_soup/focus_echo_two_svd `
  --base checkpoints/transformer_behavior_50m_long `
  --eval data/staged_copy_v1/stage_two_json/eval.jsonl `
  --merge-mode delta-svd --rank 16 `
  --result research/results/adapter_soup_focus_echo_two_svd.json `
  --comparison research/results/adapter_soup_focus_echo_two_svd_comparison.json
```

One eval split is not evidence of retention. Repeat with each relevant held-out
split (`stage_echo_json/eval.jsonl`, `stage_two_json/eval.jsonl`,
`stage_multi/eval.jsonl`) and compare the soup against the base, each source
adapter and the rehearsal adapter on the same rows.

## Validation and failures

`research/merge_adapters.py` refuses to write an output when:

- an adapter directory has no `adapter.pt`;
- `r_attn`, `alpha_attn`, `r_mlp` or `alpha_mlp` differ between adapters;
- an adapter records a rank that disagrees with its tensor shapes;
- A/B tensor names are missing on one side or extra on the other;
- A/B tensor shapes differ between adapters;
- tensors contain NaN/Inf, are not 2-D or are not floating point;
- weights do not match the adapter count, are negative/non-finite, or sum to
  zero;
- `--rank` is used with `ab` mode, or is less than one.

Errors are printed through the argument parser (exit code 2) and no merged
adapter is written. `--base` additionally loads the merged adapter and runs a
forward pass; `--eval` requires `--base` because `compare_adapters.py` needs the
reference model.

## Output

```text
checkpoints/adapter_soup/focus_echo_two/
  adapter.pt     # {"lora": {...}, "config": {...}}
  config.json    # same config, save_adapter-compatible
```

The config keeps `r_attn`, `alpha_attn`, `r_mlp`, `alpha_mlp`, the legacy
`rank`/`alpha` aliases and a `merge` block with the mode, normalized weights,
source adapter paths and rank cap. `apply_adapter` ignores the extra metadata.

The result file is content-free and has this shape:

```json
{
  "created": "2026-11-24T12:00:00",
  "mode": "ab",
  "rank": null,
  "out": "checkpoints/adapter_soup/focus_echo_two",
  "adapters": [
    "checkpoints/staged_copy_rehearsal/rehearsal_focus",
    "checkpoints/staged_copy_rehearsal/echo_json",
    "checkpoints/staged_copy_rehearsal/two_json"
  ],
  "weights": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
  "config": {
    "r_attn": 4,
    "alpha_attn": 8,
    "r_mlp": 8,
    "alpha_mlp": 8,
    "rank": 8,
    "alpha": 8,
    "method": "adapter_soup_v1",
    "merge": {
      "mode": "ab",
      "weights": [0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
      "adapters": ["..."],
      "rank": null
    }
  },
  "tensors": 16,
  "parameters": 405504,
  "validation": {
    "ok": true,
    "base": "checkpoints/transformer_behavior_50m_long",
    "adapter": "checkpoints/adapter_soup/focus_echo_two",
    "lora_parameters": 405504,
    "logits_shape": [1, 16, 50257]
  },
  "evaluation": {
    "data": "data/staged_copy_v1/stage_two_json/eval.jsonl",
    "examples": 200,
    "models": [
      {
        "label": "base",
        "path": "checkpoints/transformer_behavior_50m_long",
        "metrics": {"loss": 3.21, "ppl": 24.8, "tokens": 3120}
      },
      {
        "label": "focus_echo_two",
        "path": "checkpoints/adapter_soup/focus_echo_two",
        "metrics": {
          "loss": 1.04, "ppl": 2.83, "tokens": 3120,
          "exact_match": {"rate": 0.28, "matches": 56, "examples": 200},
          "key_copy": {"rate": 0.46, "matches": 92, "examples": 200}
        },
        "vs_base": {"loss_delta": -2.17, "loss_relative": -0.676, "ppl_ratio": -0.886}
      }
    ],
    "per_family": {
      "staged_copy_two_json": {
        "baseline": {"loss": 3.21, "tokens": 3120},
        "adapter": {"loss": 1.04, "tokens": 3120}
      }
    }
  }
}
```

Field notes:

- `weights` are normalized, not the raw `--weights` values.
- `rank` is the requested cap (`null` for `ab` or an uncapped `delta-svd`).
- `config.merge.rank` repeats that cap; for `delta-svd`, `config.r_attn` and
  `config.r_mlp` are the actual output ranks, which may be lower after the cap.
- `validation` is `null` without `--base`; `evaluation` is `null` without
  `--eval`.
- `evaluation.models` always starts with the base model. Each entry keeps only
  aggregate metrics. Row-level `user`/`assistant` text from
  `compare_adapters.py` is stripped.
- `evaluation.per_family` appears only with `--per-family` and labelled rows.

The comparison file (`--comparison`) has the same `created`/`base`/`data`/
`examples`/`adapters`/`weights`/`models` shape, plus `per_family` when
requested. It contains no prompt or answer rows either.

## Honest limits

- **A/B averaging is not a skill-preserving operation.** With two adapters the
  cross term `w1 w2 (A1 B2 + A2 B1)` can strengthen, weaken or cancel both
  skills. Equal weights are only a default, not a recommendation.
- **Delta SVD is still approximate.** It preserves the weighted delta only up
  to the rank cap; a small cap drops directions. The sum of two rank-8 deltas
  can have rank 16, so `--rank 8` intentionally compresses it.
- **Same geometry is required, but same geometry does not imply compatible
  subspaces.** Adapters trained sequentially may reuse the same directions for
  different behavior.
- **One eval split is not enough.** Measure each skill separately, plus a mixed
  split, and report exact match and key copy. A lower aggregate loss can hide a
  dead skill.
- **The merged adapter is a research artifact.** It is not promoted, not added
  to the active registry and does not change product weights. Promotion still
  needs the behavior gate.
- **Merging cannot create a skill that no source adapter learned.** The multi
  stage at 8% / 18% is the current ceiling unless a source adapter carries that
  behavior.
- **SVD cost grows with matrix size.** `delta-svd` runs one SVD per adapted
  module and is intended for small research adapters, not repeated online use.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_adapter_soup.py tests/test_lora_attn.py -q
```

The tests cover equal and weighted A/B merges, geometry and tensor-shape
failures, weight normalization, `delta-svd` reconstruction, adapter roundtrip
loading/forward and the CLI result/comparison schema.
