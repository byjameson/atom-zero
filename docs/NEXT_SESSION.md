# Next session - 1.5.0 planning

Snapshot: 2026-11-26, after the 1.4.0 release.
Tag `v1.4.0`; main includes the publication commit.
No Vast.ai instance is running; check the credit before the next rental.
Sponsor goal: **$10,000/month** (<https://github.com/sponsors/byjameson>).

## Released

`1.4.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 1.4.0` reports
**0 failures**. CI run `45785671641` is green.

The 24L capacity run on the F recipe:

| area | 24L |
|---|---|
| 2short / 3short / 4short | 95.5% / 86.5% / **77.5%** exact |
| retention | 0.790 |
| benign useful | 83.8% |
| safety (40) | 40/40 |
| gate / product / server | pass / pass / 21/21 |
| self-harm gate | **9/16** |

The 8L F model scores 58.0% on 4short but **16/16** on the same self-harm set.
Capacity bought copy, not conversation or judgment. No default promotion.

The 24L checkpoint download was interrupted at 31 MB; no 24L artifact is
published. The measured result is the evidence.

## The 1.5.0 experiment

The bottleneck is data, not depth. Expand it:

1. **Benign corpus**: move from 144 authored rows to 400-600 rows across the
   eight categories, with paraphrases, multi-turn follow-ups and "I don't know"
   cases.
2. **Contrastive boundary data**: keep the crisis/benign pairs, add
   near-neighbor safety/benign pairs for injection, secrets, privacy and
   harmful requests.
3. Train the 8L and 24L on the expanded mix (same copy/content core), then run
   the full 136-prompt product suite, 40 safety probes, self-harm gate, server
   suite and retention.
4. Target: benign >= 90%, safety 40/40, self-harm 16/16, copy >= 25%,
   retention <= 1.10.
5. If a model passes, promote it as the 1.5.0 default with the 49M fallback;
   otherwise publish the frontier and keep `0.8.0`.

## Infrastructure notes

- Vast.ai: the job now bundles the safety eval, runs gate/self-harm gate/
  136-prompt product suite/server suite, copies the checkpoint back, and uses
  LF shell scripts and non-self-matching process checks.
- Checkpoint transfer: the 24L scp dropped at 31 MB. For the next run use
  `--copy-checkpoint` and verify the file size before destroying, or upload to
  Hugging Face from the instance with an explicit token.
- Tests and gates on the VMware Ubuntu box; training on host or rented GPU.

## Rules

- Release at the next boundary (1.5.0) with fresh posts on every surface.
- Check for leftover Vast.ai instances and destroy them.
- Donation addresses stay unchanged. Default weights change only after the
  full product suite and human review.
