# Next session - 1.3.0 planning

Snapshot: 2026-11-26, after the 1.2.0 release.
Tag `v1.2.0`; main includes the publication commit.
No Vast.ai instance is running; credit about **$9.54**.
Sponsor goal: **$10,000/month** (<https://github.com/sponsors/byjameson>).

## Released

`1.2.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 1.2.0` reports
**0 failures**. CI run `45743106014` is green.

Variant C (safety 2x + benign 3x) is the first base to pass both the strict
checkpoint gate and the expanded 136-prompt product suite:

| area | result |
|---|---|
| 2short / 3short / 4short | 88.0% / 72.0% / 56.0% exact |
| retention | 0.785 |
| benign useful | 80.9% |
| safety | 40/40 |
| server suite | 20/21 |
| gate | pass |

It is published as `byjameson/atom-zero-63m-gated`; the product default stays
`0.8.0` because one in five ordinary prompts is still refused.

## The 1.3.0 experiment

Fix the two remaining blockers with **contrastive self-harm replay**:

1. Variant A (95.6% benign, 61.5% 4-key) fails one crisis probe by answering
   with unrelated coding advice. The current safety rows are generic; A needs
   paired examples:
   - crisis prompt -> supportive safe answer with a refusal marker and crisis
     guidance,
   - near-neighbor benign support question -> helpful, non-refusal answer.
2. Build `research/build_selfharm_replay.py` (branch
   `feat/contrastive-selfharm`) and mix it into the C/A recipes.
3. Train the next variant (E) on: copy 6000 + content 492 + safety 40x2 +
   benign 144x3 + contrastive self-harm pairs (x3).
4. Run the full 136-prompt product suite, the 40-probe safety gate and the
   server suite. Target: **benign useful >= 90% and safety 40/40**.
5. If E passes, prepare the default-base promotion in 1.3.0 with the 49M
   fallback. If not, publish the frontier and keep `0.8.0`.

## Infrastructure

- Product suite: `research/product_suite.py` (136 prompts).
- Server suite: `research/server_product_suite.py` (RAG, copy, multi-turn).
- Benign chat: `research/build_benign_chat.py`.
- Safety: `research/build_safety_probes.py`, `research/build_safety_replay.py`,
  and the new contrastive builder.
- Mixed replay: `research/mix_training_data.py`.
- Retention: `research/retention_eval.py`; gate:
  `research/checkpoint_gate.py`.
- Optional bases: `atomzero-quickstart --model-dir` / `--model-repo`.
- Rented GPU: `scripts/vastai/` + `docs/VASTAI.md`; destroy after every run.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Check for leftover Vast.ai instances and destroy them.
- Release at the next boundary (1.3.0) with fresh posts. Donation addresses stay
  unchanged. Default weights change only after the full product suite and human
  review.
