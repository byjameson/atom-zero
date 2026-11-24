# Randomized-key copy cycle - partial key copying

Date: 2026-11-24

## Why randomized keys

The fixed key-pair curriculum has a closed key vocabulary (10 training pairs,
each key seen 32-64 times). The adapter learned to emit those keys and got
0/200 exact match on held-out city/river families. Randomized keys remove that
shortcut: every row uses fresh keys such as `field_sk4vdh` / `k_ffce7cd6`, and
train/eval keys are disjoint. The only generalizing solution is to copy the
keys from the prompt.

## Dataset

`research/build_random_copy_queue.py` (stdlib only):

```text
train:        2,000 rows
eval:         200 rows
formats:      JSON 1,000 / KV 1,000 (train); JSON 100 / KV 100 (eval)
keys/row:     2-4, mean 2.98
unique keys:  5,959 train / 595 eval
key overlap:  0
value overlap: 0
rejected:     0
```

Every row is round-trip validated: parse the answer, compare the exact key set
and values, and reject/regenerate on failure.

## Training

```text
base:           0.8.0
LoRA:           MLP rank 8 / alpha 8 + out_proj rank 4 / alpha 8
parameters:     405,504
steps:          1,000 sampled rows  -> 250 optimizer steps
                4,000 sampled rows  -> 1,000 optimizer steps
batch size:     4
replay:         legacy copy-curriculum replay, weight 0.2
content weight: 2
lr:             3e-4
seed:           23
```

## Held-out randomized-key eval (200 rows)

| metric | base | 250 opt steps | 1,000 opt steps |
|---|---:|---:|---:|
| full loss | 4.9664 | 2.2575 | 1.3687 |
| content loss | - | 2.2311 | 1.5299 |
| key loss | - | 1.9575 | 1.2804 |
| value loss | - | 2.4432 | 1.6922 |
| syntax loss | - | 2.5397 | 1.1408 |
| exact match | 0/200 | 1/200 (0.5%) | 3/200 (1.5%) |
| key-copy rate | 0% | 1.0% | 4.5% |
| key recall | 0% | 4.8% | 19.0% |

The trend is real but slow: four times more optimizer steps roughly
quadrupled key-copy rate and quadrupled key recall, and full loss halved.
Generations now preserve some `field_` prefixes and complete values, but they
still merge keys, truncate random suffixes, and drift into neighboring
training patterns.

## Interpretation

- Randomized keys are a much harder but much cleaner test than fixed key pairs.
- The adapter is beginning to learn a copy behavior; it is not yet reliable.
- The remaining gap may be capacity/locus (LoRA cannot change Q/K/V), data
  difficulty (random token sequences), or optimization length.
- A full fine-tune control (`research/finetune_copy_control.py`) will separate
  architecture limits from LoRA limits. If full fine-tune also fails, the base
  or the task format is the limit. If it succeeds, the adapter locus is the
  limit.

## Next steps

1. Run the full fine-tune control on the same randomized-key data, short run
   first (100-300 optimizer steps), then a longer run if loss is still falling.
2. Add a staged curriculum: single-key KV echo -> single-key JSON -> two keys ->
   reordered keys. Measure key recall after each stage.
3. Consider a lower-entropy key pool (common single tokens) to separate
   tokenization difficulty from copy-circuit difficulty.
4. Keep the behavior gate: a candidate is not promotable until held-out exact
   match and key-copy rate clear an explicit threshold.

## Evidence

- `research/results/random_copy_long_audit.json` (local, contains prompts; not
  committed)
- `research/results/random_copy_audit.json` (local)
- `data/random_copy_v1/` (local, gitignored)
- `checkpoints/adapters_random_copy_long/v0001` (VM local)
- `docs/FULL_FINETUNE_CONTROL.md`
