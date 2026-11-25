# 1.1.0 batch plan - safety-replay ratio sweep

Date: 2026-11-26

## Problem

The 10x safety-replay model passes the safety gate (40/40) but refuses benign
open prompts: on the 25-prompt benign suite it scores **4% useful / 96%
refusal**, while the 49M base scores **100% useful / 0% refusal**. The
safety-replay ratio is the variable.

## Sweep

Base: 8-layer 63M upcycle. Copy + content mix is fixed at 6000 + 492 rows.
Safety replay rows are 40 templates, repeated at:

| ratio | safety rows | total rows | share |
|---|---:|---:|---:|
| 1x | 40 | 6532 | 0.6% |
| 2x | 80 | 6572 | 1.2% |
| 5x | 200 | 6692 | 3.0% |
| 10x | 400 | 6892 | 5.8% |

Training: 3000 steps, batch 4, lr 5e-5, content weight 2, seed 71, pretokenized
cache.

## Evaluation per ratio

1. `research/product_suite.py`: benign useful/refusal, safety pass, copy exact,
   content ratio, `product_ok`.
2. `research/checkpoint_gate.py`: full gate on content, 4short copy and
   held-out safety.
3. `research/compare_adapters.py`: 200-row 2short/3short exact and key-copy.
4. `research/retention_eval.py`: city content ratio.

## Selection rule

- Safety pass rate >= 0.9 on held-out probes.
- Benign useful rate >= 0.8, refusal rate <= 0.2.
- 4short exact >= 0.25.
- Content ratio reported; a candidate above 1.10 is rejected.

Pick the **lowest** safety ratio that satisfies all four. If no ratio does,
report the frontier (best safety at each benign level) and keep `0.8.0`.

## Current evidence

| ratio | benign useful | benign refusal | safety | copy exact | content |
|---|---:|---:|---:|---:|---:|
| 10x | 4% | 96% | 100% | 57.0% | 0.722 |
| 1x/2x/5x | training | training | - | - | - |

The frontier is expected between 1x and 10x. The release only changes the
default base if a candidate passes the product suite; otherwise 1.1.0 records
the frontier and keeps `0.8.0`.
