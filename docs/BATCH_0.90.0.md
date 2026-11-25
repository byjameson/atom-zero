# 0.90.0 batch plan - capacity scaling and product viability

Date: 2026-11-25

## Question

0.80.0 proved the 4-key copy wall was capacity-bound inside the trained key
grammar: an 8-layer 63M checkpoint reached 89.5% / 77.0% / 65.5% exact on
held-out 2/3/4-key copy, while the 49M stayed at 32% / 12% / 0%. The open
question for 0.90.0 is whether that result is a one-off or scales, and what it
costs in content retention.

## Reference: 8L / 63M

- Parameters: 63,389,184 (8 layers, interpolated upcycle).
- Copy, 200 held-out rows: **2short 89.5%, 3short 77.0%, 4short 65.5%** exact
  (key-copy 91.5% / 83.5% / 74.0%).
- Content retention vs the 49M base on city YAML/river: **ratio 1.0889**,
  threshold 1.10, `retention_ok: true` (90 scored rows). This is content-token
  loss only; exact match is 0 for both models on that set.

## Paid run: 24L / ~177M

Rented on Vast.ai (RTX 3090 24 GB) because the local 6 GB card is borderline
for full AdamW at this size (planning estimate 3.7-5.9 GB).

```text
upcycle:   6 -> 24 layers, interpolate, 176,795,136 params
training:  copy curriculum + 153 teacher rows, 4000 steps, batch 16,
           lr 5e-5, content weight 2
seeds:     101, 102
eval:      200 held-out rows for 2/3/4-key copy, GPU
retention: 49M baseline vs 24L candidate, city YAML/river, 200 rows, GPU
```

The instance is destroyed automatically at the end; a 3.5-hour failsafe
automation exists in case the launcher process dies.



## Preliminary 24L result (seed 101)

```text
2short: 90.5% exact / 92.5% key-copy
3short: 84.5% / 87.0%
4short: 67.5% / 75.0%
retention ratio: 1.647  (threshold 1.10 -> fail)
```

Against the 8L reference (89.5 / 77.0 / 65.5, retention 1.089), depth adds
**+2.0 points** on 4short and makes content retention clearly worse. Depth is
not the lever. Seed 102 is running for variance, but the direction is already
clear: the next experiment must change the data mix, not the layer count.

Safety gate on the 8L copy specialist: baseline 12.5% (5/40), candidate 0%
(0/40). The copy-only fine-tune regressed refusal behavior as well as content.

## Gate result: 8L is not product-viable

The full checkpoint gate on the 90-row city set (not the 20-row preview) is
decisive:

```text
content loss: 4.153867 -> 4.522974  (delta +0.369107, +8.9%)
exact match:  0.0 -> 0.655
key copy:     0.0 -> 0.74
status:       fail   (reason: content_worse)
safety:       skipped (no probe file yet)
```

The 20-row preview had shown a content improvement; the full set reversed it.
Small subsets are not evidence. The 8L copy specialist therefore stays a
research artifact. The 0.90.0 question is now sharper: can **mixed
copy+content training** keep the copy gain while removing the content
regression? That is what `feat/mixed-replay` prepares.

## Decision tree

1. **4short exact rises well above 65.5% and retention stays <= 1.10**
   -> a larger copy-capable base is viable; next step is packaging and the
   full checkpoint gate.
2. **4short rises but retention exceeds 1.10**
   -> capacity is not enough; the next experiment mixes content replay into
   full fine-tuning instead of copy-only data.
3. **4short does not rise**
   -> 8L was enough and depth is not the lever; redirect to data quality or a
   mixed-objective training run.
4. **Run fails or is inconclusive**
   -> keep the 8L result as the 0.80.0 evidence and do not make a 0.90.0 claim.

## Success criteria

- Two seeds both evaluated; report mean and spread, not a single lucky run.
- 200-row held-out copy numbers for 2/3/4-key.
- Retention ratio with the 49M baseline on the same rows.
- Checkpoint gate report for any candidate that might replace or supplement
  the product base.
- Full VM suite + benchmark before release.

## Release

0.90.0 is the next 10-minor boundary. Publish only if the evidence is complete.
Surfaces: PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn, X
with new posts. Donation addresses and the product base stay unchanged unless
a new base explicitly passes the gates.
