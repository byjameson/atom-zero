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
| 1x | 32% | 68% | 95% | 59.0% | 0.802 |
| 2x | 4% | 96% | 100% | 66.0% | 0.754 |
| 10x | 4% | 96% | 100% | 57.0% | 0.722 |

**The ratio sweep failed.** Even 1x (40 safety rows, 0.6% of the mix) pushes
benign refusals to 68%; 2x is already at 96%. More safety data does not buy a
better frontier ? it moves the model into a refusal mode for any prompt that is
not grounded or copy-routed.

The missing variable is **benign chat replay**: the mix has copy rows and
factual content rows, but almost no ordinary conversational examples. The next
branch (`feat/benign-chat-replay`) adds short helpful answers for identity,
greeting, explanation, cooking, travel, coding, planning and general knowledge,
so the model can learn the boundary instead of treating every open prompt as
unsafe.

The release only changes the default base if a candidate passes the product
suite; otherwise 1.1.0 records the frontier and keeps `0.8.0`.

## Final result: benign chat replay fixes the collapse

Two variants, same 8L base, same copy/content mix, same seed:

| variant | safety | benign | 4short | retention | benign useful | gate |
|---|---:|---:|---:|---:|---:|---|
| A | 1x | 3x | **61.5%** | **0.759** | **92%** | 39/40 (fail) |
| B | 5x | 3x | 38.0% | 0.901 | 88% | **40/40 (pass)** |

Both pass the product suite (`benign_useful >= 0.8`, `safety >= 0.9`,
`copy_exact >= 0.25`). B also passes the strict checkpoint gate at 100% safety.
A has the best product profile ? 92% benign, 61.5% four-key exact, content loss
0.759 ? and fails only one of 40 safety probes.

The missing lever was never the safety ratio alone. It was **benign chat
replay**: 144 ordinary prompts with helpful answers, repeated 3x, teach the
model that non-grounded conversation gets an answer. Without it, even 1x
safety data collapses benign useful to 32%.

Decision: publish B as the first optional base that passes both the strict gate
and the product suite. The default remains `0.8.0` until one more product-suite
expansion and human review; A is the candidate for the next iteration.
