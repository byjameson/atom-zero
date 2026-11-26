# 1.5.0 batch plan - benign corpus expansion

Date: 2026-11-26

## Why

1.4.0 proved that capacity is not the conversation/safety lever: 24L F reached
77.5% four-key copy but benign useful stayed 83.8% and the self-harm gate fell
to 9/16. The 63M and 177M models both trade conversation for refusal when the
mix contains safety/self-harm rows.

The bottleneck is the benign corpus: 144 authored rows are too few to offset
the refusal bias.

## Data

1. **Benign chat replay 480 rows** (60 per category) across identity, greeting,
   explanation, cooking, travel, coding, planning and general knowledge. This
   is the main lever.
2. **Contrastive boundary pairs**: for injection, secret exfiltration, privacy
   and harmful/illegal, an unsafe prompt with a short refusal plus a
   near-neighbor benign request with a helpful answer. Kept small and tested
   separately so a regression is attributable.

## Variants

| variant | mix | purpose |
|---|---|---|
| I | copy 6000 + content 492 + safety 40 + benign 480x3 + selfharm 16 | test benign expansion alone |
| J | I + boundary pairs 96 | test whether boundary data helps without hurting benign |

Training: 8L 63M, 3000 steps, batch 4, lr 5e-5, content weight 2, seed 71,
pretokenized cache. If I or J passes, optionally repeat on 24L.

## Promotion bar

- benign useful (136) >= 90%
- safety (40) = 40/40
- self-harm gate = 16/16
- 4short copy >= 25%
- retention <= 1.10
- server suite pass
- full VM suite and human review

If a variant passes, promote it as the 1.5.0 default with the 49M fallback.
Otherwise record the frontier and keep `0.8.0`; the next lever is teacher-
generated benign conversation, not more templates.

## Evidence so far

| model | benign | safety | self-harm | 4short | retention |
|---|---:|---:|---:|---:|---:|
| A (8L, no self-harm) | 95.6% | 39/40 | - | 61.5% | 0.759 |
| F (8L) | 83.8% | 40/40 | 16/16 | 58.0% | 0.841 |
| 24L F | 83.8% | 40/40 | 9/16 | 77.5% | 0.790 |

The target is the A conversation profile with the F safety profile.
