# 0.80.0 batch plan - new-task distillation and a larger research base

Date: 2026-11-25

## Why 0.80.0

0.70.0 showed a partial distillation win on 2-3 key copying (2short 32% exact,
3short 12%) but 4-key copying stayed at 0%. The hard-label teacher data was
mostly canonicalized source answers, so it added little new content. 0.80.0
tests two different doors:

1. **New-task distillation**: Qwen generates fresh 2/3/4-key tasks with new
   keys/values; the 49M student trains on verified new content.
2. **Depth upcycling**: expand the existing 6-layer 49M checkpoint to 8/12
   layers for a larger research base, without touching the product 0.8.0
   weights.

## New-task distillation

- `research/build_distill_new_tasks.py` asks Qwen for batches of independent
  flat JSON tasks, validates keys/values, deduplicates and canonicalizes to
  flat JSON/KV.
- The eval files are never used to generate tasks.
- Target: 1000 new tasks, 2/3/4 keys, balanced formats.
- Train the 49M from `multi_recover` on new tasks + replay; evaluate on the
  original held-out 2/3/4-key sets.

## Depth upcycling

- `research/upcycle_model.py` duplicates/interpolates transformer blocks,
  updates `cfg["layers"]`, saves a research checkpoint and a parameter report.
- No training in this branch; the next step is continued pretraining or
  distillation on cached data.
- The product base remains `0.8.0`.

## Success criteria

- New-task distillation improves 4short exact above 0% without losing >5
  points on 2-3 key stages, or gives a clear negative result.
- Upcycle utility produces a valid 8/12-layer research checkpoint and a
  realistic compute/data plan.
- Full VM suite + benchmark.
- Release only at the 0.80.0 boundary with verification.


## Progress log

- `feat/depth-upcycle` merged: `research/upcycle_model.py`, report utility and tests
  (12 passed).
- 8-layer research checkpoint built from the 49M base with interpolate mode:
  **63,389,184 params**, mapping `[0,0,1,2,3,3,4,5]`, marked
  `research_control: true`, `product_weights: false`. Untrained 4short copy is
  0% with loss 6.90, as expected for a copied-depth base.
- `data/multi_short_v1/train_all_plus_distill.jsonl` combines the three
  curriculum stages (6000 rows) with the 153 verified teacher rows.
- Full fine-tune of the 8L research base is running on the host GPU
  (`fullft_upcycle8l_v1`, 3000 steps, lr 5e-5, content weight 2).

## Publication

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
New posts only; no edits of the 0.70.0 posts.

## Final result

The capacity experiment succeeded and the data-only control failed, which is
the useful combination:

- 8L upcycle + full fine-tune (63,389,184 params, 3000 steps on the host GPU):
  **2short 89.5% exact / 91.5% key-copy, 3short 77.0% / 83.5%, 4short 65.5% /
  74.0%** on 200 held-out rows per stage.
- 49M trained on 656 new teacher tasks (246 four-key): **4short 0% / 5%**,
  2-3 key flat. Data alone did not break the wall.
- Out-of-distribution key styles (`field_`, `k_` prefixes): **0/30**. The gain
  is inside the trained 3-6 lowercase alphanumeric key grammar.

Product base stays `0.8.0`. The 63M checkpoint is a copy-specialized research
artifact. Evidence: `research/results/upcycle_8l_capability_summary.json`.
