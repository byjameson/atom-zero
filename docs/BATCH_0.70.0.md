# 0.70.0 batch plan - teacher distillation into the 49M model

Date: 2026-11-25

## Why 0.70.0

0.60.0 proved the copy wall is capacity-related: a pretrained Qwen2.5-0.5B
copies 2-4 keys at 55-85% exact, while the from-scratch 49M adapter stays at
0-6%. The next step is not to replace the product with Qwen. It is to use Qwen
as a **teacher** and distill verified copy behavior into our own 49M model.

## Pipeline

```text
staged train prompts
  -> Qwen2.5-0.5B teacher (host GPU)
  -> tolerant parse + exact key/value verification
  -> canonical flat JSON / KV / YAML
  -> 49M student training (host GPU)
  -> held-out 2/3/4-key evaluation
  -> behavior gate in VMware (CPU)
```

The teacher is used only for training data. Eval prompts and held-out keys are
never sent to the teacher; the student is evaluated on the original disjoint
eval files.

## Data

- Source: `data/multi_short_v1/stage_{2short,3short,4short}/train.jsonl`.
- Teacher: `Qwen/Qwen2.5-0.5B-Instruct` from the local cache, greedy decoding.
- Verification: parsed teacher keys/values must equal the source pairs exactly.
- Invalid teacher output is rejected or replaced by the deterministic oracle
  with `--fallback-oracle`; counts are recorded.
- Output: `data/distill_copy_v1/train_distill.jsonl` plus a content-free
  manifest.

## Training

- Base: 49M `0.8.0`.
- Train on teacher-canonicalized copy data, with a replay mix of the original
  curriculum to reduce forgetting.
- GPU on the RTX 3050 6GB; evaluate on held-out 2short/3short/4short and the
  mixed set.
- Success is measured, not assumed: exact/key-copy before/after, per-stage
  retention, and behavior gate status.


## Current result

- 300 teacher attempts produced **153 verified rows** (json 66, kv 87, yaml 0);
  147 were rejected. The 4short stage contributed only 32 rows.
- A 49M student started from `multi_recover` and trained on the 153 rows plus
  50% rehearsal. Mixed loss moved `0.7769 -> 0.7683` (essentially flat).
- On a 20-row-per-stage check, the distilled adapter scored **20% exact on
  2short, 15% on 3short, 0% on 4short**. The same subset gave `multi_recover`
  0/0/0, but that subset is small and not the earlier 50-row split; a 50-row
  comparison is running before any claim.
- The honest hypothesis is **partial 2-3 key copy improvement, 4-key wall
  unchanged**. Do not claim a general distillation win without the 50-row
  result and a VM behavior gate.

## Release gate

- Distillation cycle ledger complete.
- Student improves at least one held-out 3-4 key stage without losing >5 points
  on any retained stage.
- Full VM suite + benchmark.
- Build + twine + verify_release 0.70.0 with 0 failures.

## Publication

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
New posts only; no edits of the 0.60.0 posts.
