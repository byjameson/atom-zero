# Next session - 1.4.0 planning

Snapshot: 2026-11-26, after the 1.3.0 release.
Tag `v1.3.0`; main includes the publication commit.
No Vast.ai instance is running; credit about **$9.54**.
Sponsor goal: **$10,000/month** (<https://github.com/sponsors/byjameson>).

## Released

`1.3.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 1.3.0` reports
**0 failures**. CI run `45763599533` is green.

The finding: at 63M, contrastive self-harm replay fixes the crisis probe but
costs 7–20 points of benign conversation.

| variant | self-harm rows | benign | safety | 4short | gate | product |
|---|---:|---:|---:|---:|---|---|
| A | 0 | 95.6% | 39/40 | 61.5% | fail | pass |
| H | 8 | 76.5% | 40/40 | 46.5% | pass | fail |
| E | 48 | 75.7% | 40/40 | 58.0% | pass | fail |
| F | 16 | 83.8% | 40/40 | 58.0% | pass | pass |
| G | 16 + benign 6x | 83.8% | 35/40 | 54.5% | fail | fail |

F is published as `byjameson/atom-zero-63m-selfharm`; default stays `0.8.0`.
The deterministic safety filter gained a narrow crisis category (82 safety
cases pass), so the product path can catch crisis input without training.

## The 1.4.0 experiment

The 63M model cannot separate the crisis boundary from ordinary conversation.
Test whether **capacity** fixes it:

1. Build the 24L / 176.8M upcycle.
2. Fine-tune on the A recipe + 16 contrastive self-harm rows (F recipe).
3. Evaluate: 136-prompt product suite, 40 safety probes, server suite,
   2/3/4-key copy, retention.
4. Target: **benign >= 90%, safety 40/40, copy >= 25%, retention <= 1.10**.
5. If the 24L passes, promote it as the 1.4.0 default with the 49M fallback.
   If not, the next lever is a wider benign corpus and a larger rented GPU run.

The 24L training fits a 24 GB Vast.ai 3090/4090; use the existing
`scripts/vastai/` pipeline with `--layers 24`, the mixed data, and the
pretokenized cache. Destroy the instance after the run.

## Infrastructure

- Product suite: `research/product_suite.py` (136 prompts).
- Server suite: `research/server_product_suite.py`.
- Self-harm replay: `research/build_selfharm_replay.py`.
- Crisis filter: `src/atomzero/safety.py` (new `crisis` category).
- Mixed replay: `research/mix_training_data.py`.
- Retention: `research/retention_eval.py`; gate:
  `research/checkpoint_gate.py`.
- Optional bases: `atomzero-quickstart --model-dir` / `--model-repo`.
- Rented GPU: `scripts/vastai/` + `docs/VASTAI.md`; destroy after every run.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Check for leftover Vast.ai instances and destroy them.
- Release at the next boundary (1.4.0) with fresh posts. Donation addresses stay
  unchanged. Default weights change only after the full product suite and human
  review.
