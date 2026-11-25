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

## Publication

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
New posts only; no edits of the 0.70.0 posts.
