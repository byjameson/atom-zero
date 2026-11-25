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
