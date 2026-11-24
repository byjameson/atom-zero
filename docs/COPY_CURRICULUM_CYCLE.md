# Copy-curriculum cycle - value copy without key copy

Date: 2026-11-24

## Setup

The fixed key-pair curriculum was generated with the new builder:

```text
builder:        research/build_diverse_teacher_queue.py --dry-run
key pairs:      12 (10 train, city/population and river/km held out)
split:          entity-first, zero entity overlap
train:          576 rows, 18 families
in-dist eval:   192 rows, 24 families, unseen entities
cross eval:     200 rows (city_json/city_kv/city_yaml, river_json/river_kv)
replay:         192 legacy person/product/arithmetic rows
answers:        canonical compact JSON / KV / YAML
```

Two adapters were trained for 2,000 steps with replay weight `0.2`,
content weight `5`, LR `3e-4`, MLP rank 8 / alpha 8, seed 23:

- MLP-only (`368,640` LoRA parameters);
- `out_proj` + MLP (`405,504` LoRA parameters, attention rank 4 / alpha 8).

## Cross-family results (200 rows)

```text
                 base      MLP       +out_proj
full loss        4.5012    1.8799    1.7637
content loss     4.2526    2.3629    2.2902
syntax loss      4.8433    1.2155    1.0396
exact match      0/200     0/200     0/200
```

Per-family content loss:

| family | base | MLP | +out_proj |
|---|---:|---:|---:|
| city_json | 5.6720 | 3.1385 | 2.9985 |
| city_kv | 3.5816 | 2.3138 | 2.1283 |
| city_yaml | 3.4757 | 3.0017 | 2.7850 |
| river_json | 5.7269 | 1.5173 | 1.6627 |
| river_kv | 2.5945 | 1.5374 | 1.6379 |

Attention `out_proj` lowers city content loss and overall loss, but slightly
raises river content loss. Neither adapter produces a single exact answer.

## What the generations show

Greedy generations on held-out families copy the entity and the number but
replace the prompt keys with training keys. Representative shape:

```text
prompt keys:   river, km
generated:     "river": "Volga", "price": 5784      (expected km)

prompt keys:   city, population
generated:     "city": "Istanbul", "length": 15112  (expected population)

prompt keys:   city, population
generated:     "name": "Trabzon", "price": 1363     (expected city/population)
```

This is direct evidence of **value copy without key copy**. The model has
learned that an answer contains a name-shaped value and a number, but it
selects key names from the training key vocabulary instead of copying the
keys from the prompt.

The outputs also expose a syntax regression: JSON generations sometimes start
with `elif`/`el`/`if` instead of `{`, and numbers are truncated or merged with
neighbouring tokens. Content weighting improved value tokens at the cost of
some format stability.

## Why the curriculum did not teach key copying

- The training key set is still a closed vocabulary of 10 pairs. Each key
  appears ~32-64 times, so memorizing keys is easier than learning to copy.
- The held-out keys (`city`, `population`, `river`, `km`) are new, but the
  model has no incentive to treat prompt keys as copy targets.
- MLP-only LoRA cannot directly change the attention copy operation; adding
  `out_proj` helps content but does not create a key-copy circuit.
- Single-example updates and 2,000 steps over 576 rows are noisy; the final
  format can drift.

## Next experiment

Use **randomized key strings** (e.g. `k_7f3a91c2`, `field_b4`) in every
training row, with disjoint random keys in evaluation. A closed key vocabulary
is then impossible, so the only generalizing solution is to copy the keys from
the prompt. Add key-token loss and key-copy-rate metrics before judging the
result. Gradient accumulation and a syntax-weight floor are supporting fixes,
not the main hypothesis.

## Evidence

- `research/results/copy_curriculum_summary.json`
- `data/copy_curriculum_v1/` (local, gitignored)
- `checkpoints/adapters_copy_mlp/v0001` (VM local)
- `checkpoints/adapters_copy_attn/v0001` (VM local)

The comparison JSONs contain local prompt text and are not committed.
