# Depth upcycling (research only)

Date: 2026-11-25

Status: **research utility, not a product release and not a capability
claim**. The product 0.8.0 checkpoint stays unchanged.

## Why

The 49.2M from-scratch base is capacity-limited. Three- and four-key copy has
stayed at 0-6% exact match even after distillation and LoRA training. A larger
research base is the next experiment, but training one from scratch is
expensive.

Depth upcycling expands the existing 6-layer checkpoint to 8 or 12 layers by
reusing its trained blocks, then continues research pretraining or fine-tuning
on cached data. It is a cheaper way to test whether extra depth helps this
copy/transcription wall. It does not add knowledge by itself.

`research/upcycle_model.py`:

1. loads a base checkpoint without modifying it;
2. keeps embeddings, positional embeddings, the final norm and the tied LM
   head;
3. copies selected transformer blocks into new positions while preserving
   order;
4. optionally blends new duplicate blocks with a neighbor;
5. updates `cfg["layers"]`, verifies one forward pass and the logits shape;
6. writes a research checkpoint with a tokenizer copy and a sidecar that says
   `research_control: true`, `product_weights: false`,
   `weights_changed: true`.

It does **not** train, evaluate, promote or modify the product checkpoint.

## Layer mapping

The mapping is deterministic. Target block `t` copies source block

```text
floor(t * source_layers / target_layers)
```

The mapping is non-decreasing, starts at block 0 and ends at the last source
block. For a 2x expansion every block is copied once.

| Expansion | Mapping | New blocks |
|---|---|---|
| 6 -> 8 | `[0, 0, 1, 2, 3, 3, 4, 5]` | 1 and 4 |
| 6 -> 12 | `[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5]` | 1, 3, 5, 7, 9 and 11 |

Two modes are available:

- `duplicate` copies each selected block exactly.
- `interpolate` keeps the first occurrence of each source block exact and
  blends later duplicate occurrences with the adjacent source block. For a
  run of two, the new block is `0.5 * source + 0.5 * neighbor`. For the last
  source block the neighbor is the previous block.

Exact duplicates can remain functionally redundant. Interpolation only
partially breaks that symmetry; continued training is still required.

`--seed` is recorded for the later research run and seeds the deterministic
forward-check input. It does not randomize copied or interpolated weights; the
mapping is deterministic.

## Exact command (8 layers)

Run from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

# Plan first; this writes nothing.
python research/upcycle_model.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --out checkpoints/upcycle/atom-zero-49m-8l `
  --layers 8 `
  --mode interpolate `
  --seed 23 `
  --dry-run

# Write the research checkpoint and the content-free report.
python research/upcycle_model.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --out checkpoints/upcycle/atom-zero-49m-8l `
  --layers 8 `
  --mode interpolate `
  --seed 23 `
  --report research/results/upcycle_8l.json
```

## Exact command (12 layers)

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/upcycle_model.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --out checkpoints/upcycle/atom-zero-49m-12l `
  --layers 12 `
  --mode duplicate `
  --seed 23 `
  --report research/results/upcycle_12l.json
```

Swap `--mode duplicate` for `--mode interpolate` to blend the new blocks.
`--force` is only needed to replace an existing checkpoint in `--out`; it
never allows writing into the base directory.

## Output layout

```text
checkpoints/upcycle/atom-zero-49m-8l/
  model.pt              # config, state, step 0, research metadata
  config.json           # architecture + research flags + tokenizer name
  chat_en_bpe8k.json    # tokenizer copy
  upcycle.json          # content-free sidecar
```

`upcycle.json` records the base path, layer mapping, interpolation plan,
parameter counts, forward-check shape and the research flags. `model.pt` uses
the same `config` / `state` format as other checkpoints, but it is a **new
base**, not an adapter:

```python
from research.adapter_runtime import load_model

model, tokenizer = load_model("checkpoints/upcycle/atom-zero-49m-8l")
```

Do not load it as `load_model(original_base, upcycled_dir)`: the depth differs
from the original base, so the adapter path cannot apply it. `step` is 0
because no training was run. The report JSON keeps `base_step` for provenance.

## Parameter counts

Counts are unique parameters, matching
`sum(p.numel() for p in model.parameters())`. The tied LM head is stored once
in the model and shares storage with `tok.weight`, so the state-dict sum is
larger by one vocabulary embedding (`6,291,456` parameters here).

| Target | Layers | Parameters | Added | Added % | FP32 weights |
|---|---:|---:|---:|---:|---:|
| base | 6 | 49,213,440 | - | - | 196.9 MB |
| 8-layer | 8 | 63,389,184 | 14,175,744 | 28.8% | 253.6 MB |
| 12-layer | 12 | 91,740,672 | 42,527,232 | 86.4% | 367.0 MB |

The 8-layer `model.pt` was 253,581,219 bytes in the local smoke run, close to
the unique-parameter estimate plus container overhead. The 12-layer file is
expected to be about 367 MB.

## RTX 3050 6 GB estimates

These are **planning estimates**, not measured training runs. The reference is
the measured 6-layer product step time of about 0.14 s at batch 16 x seq 128
(2,048 tokens per step). Other rows scale that rate linearly with parameter
count. Real throughput depends on optimizer, batch size, sequence length,
gradient checkpointing, data loading and thermal state.

### Memory

The full-fine-tune state assumes fp32 weights, gradients and two AdamW
moments: 16 bytes per parameter. The activation figure is a lower bound; the
practical range adds allocator, attention and CUDA-context overhead.

| Target | Full-FT state | Activation lower bound | Practical range | Fit on 6 GB |
|---|---:|---:|---:|---|
| 6-layer | 0.79 GB | 0.09 GB | 1.2-2.2 GB | likely |
| 8-layer | 1.01 GB | 0.13 GB | 1.5-2.6 GB | likely |
| 12-layer | 1.47 GB | 0.19 GB | 2.1-3.4 GB | tight |

The 8-layer base should fit a full fine-tune at a small batch size on 6 GB.
The 12-layer base is tighter; reduce the batch size, use gradient accumulation
or use gradient checkpointing if CUDA runs out of memory. LoRA training uses
less optimizer state but still carries the base weights and activations.

### Compute and data

| Target | Throughput estimate | Existing 214.8M replay mix | 20 tokens/param | Time for 20 tokens/param |
|---|---:|---:|---:|---:|
| 6-layer | 14,629 tokens/s | 4.1 h | 0.98B | 18.7 h |
| 8-layer | 11,357 tokens/s | 5.3 h | 1.27B | 31.0 h |
| 12-layer | 7,847 tokens/s | 7.6 h | 1.83B | 64.9 h |

A run on the existing ~215M-token replay mix is a short research continuation
(about 5-8 hours at this rate). A Chinchilla-style 20-tokens-per-parameter
budget is 1.3-1.8B tokens and takes roughly 1.3-2.7 days on this GPU.

The data budget is the binding constraint. The existing replay mix is only
about 2.3 tokens per parameter for the 12-layer model, far below the
20-tokens-per-parameter planning target. More clean, licensed data is needed
before copied depth can be judged.

## Research boundary

- This is **not** the product 0.8.0 checkpoint and it does not replace it.
- No training was run; `weights_changed: true` means the research checkpoint
  differs in architecture/weights from the base, not that product weights
  changed.
- Do not write this checkpoint into `data/active_adapter.json`, the promotion
  path or product defaults.
- Do not promote it without held-out loss, behavior, safety, citation/content
  audits, latency/memory measurements and the product benchmark. Copied depth
  alone is not evidence of capability.
- The utility refuses to write inside the base directory and refuses to
  overwrite an existing checkpoint without `--force`.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python -m pytest tests/test_upcycle_model.py -q
```

The tiny-model tests cover layer expansion, the copied layer mapping,
interpolation, forward shape, research metadata flags, dry-run (no writes),
refusing to overwrite the base, `--force`, report contents and the missing
tokenizer error.
