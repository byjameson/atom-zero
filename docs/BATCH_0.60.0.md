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

## Capacity decision

The RTX 3050 6 GB host can run Qwen2.5-0.5B-Instruct from the local cache. A
20-row-per-stage probe with a tolerant parser (accepting `{"pairs": ...}` and
flat JSON) measured:

```text
2short: 65% baseline / 85% flat prompt
3short: 55% / 65%
4short: 55% / 85%
```

That is far above the 49M adapter's 0-6% on 3-4 keys. The decision is:

- keep the from-scratch 49M `0.8.0` as the low-cost default;
- treat the larger pretrained base as a separate research tier;
- do not replace product weights without a behavior gate, citation/RAG
  integration and a hardware budget review.

The larger base is capacity evidence, not a product claim or a from-scratch
model.

## Controlled QKV vs MLP result

Three seeds, 2000 sampled rows per run, 30-row eval subset:

| arm | exact mean | key-copy mean | exact 95% CI | key-copy 95% CI |
|---|---:|---:|---|---|
| qkv4 (QKV r4/a8) | 3.33% | 8.89% | 3.33-3.33% | 4.11-13.67% |
| mlp (QKV off) | 3.33% | 5.56% | 3.33-3.33% | 0.77-10.34% |

Verdict: **inconclusive**. Exact match is identical across both arms; the
key-copy difference is 3.3 points with overlapping 95% intervals. QKV LoRA
does not establish a win and does not break the 3-4 key wall.

## Release gate

- QKV ablation complete on VM with 3 seeds × 2 arms.
- Scorecard generated from real result files and reviewed for honest status.
- VM full suite + benchmark.
- Build + twine + verify_release 0.60.0 with 0 failures.

## Publication

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
New posts only; no edits of the 0.50.0 posts.
