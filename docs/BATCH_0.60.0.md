# 0.60.0 batch plan - controlled attention capacity and capability scorecard

Date: 2026-11-25

## Why 0.60.0

0.50.0 shipped copy routing and QKV LoRA, but the first QKV run was
uncontrolled: 6% exact / 18% key copy from base on multi-short. The project
needs a fair attention-capacity experiment and a six-criterion capability
scorecard before the 0.6 boundary.

## Work items

1. **Controlled QKV ablation**
   QKV LoRA vs matched MLP-only baseline, same data/order/steps/LR/geometry,
   three seeds, longer training. Aggregate exact/key-copy mean/std. Do not claim
   a win if confidence intervals overlap.

2. **Six-criterion capability scorecard**
   - gap discovery;
   - learning policy;
   - retention;
   - transfer;
   - improving its own learning algorithm;
   - low hardware cost.
   Each criterion gets met/partial/not_met/unknown, evidence paths and a next
   measurable target. Tool use is not counted as model transfer.

3. **Held-out task family evidence**
   Arithmetic, city-YAML and city/river, 30 rows each. Current result:
   base and `multi_recover` both **0% exact** on all three. This is a
   no-transfer result and must be in the scorecard.

## Release gate

- QKV ablation complete on VM with 3 seeds × 2 arms.
- Scorecard generated from real result files and reviewed for honest status.
- VM full suite + benchmark.
- Build + twine + verify_release 0.60.0 with 0 failures.

## Publication

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
New posts only; no edits of the 0.50.0 posts.
