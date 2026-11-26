# Next session - 1.6.0 planning

Snapshot: 2026-11-26, after the 1.5.0 release.
Tag `v1.5.0`; main includes the publication and verifier commits.
No Vast.ai instance is running; credit about **$8.88**.
Sponsor goal: **$10,000/month** (<https://github.com/sponsors/byjameson>).

## Released

`1.5.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 1.5.0` reports
**0 failures**. CI on the release commit is green.

The default base is now the **63M balanced** checkpoint (weights `0.9.0`):

| area | result |
|---|---|
| 2short / 3short / 4short | 86.0% / 75.5% / **48.5%** exact |
| retention | **0.703** |
| benign useful (136) | **93.4%** |
| safety (40) | **40/40** |
| self-harm (16) | **16/16** |
| gate / product / server | pass / pass / 20/21 |

The 49M `0.8.0` base remains the fallback.

## The 1.6.0 work

1. **Re-evaluate the 24L F model with the fixed safety eval.** Its 1.4.0
   self-harm score (9/16) is provisional because benign eval rows were scored
   as missing refusals. Rerun the gate with `kind`-aware `expect`. If the 24L
   self-harm is actually healthy, capacity may still be a useful copy lever.
2. **Broaden the product suite**: multi-turn follow-ups, ambiguity, "I don't
   know" cases, and RAG/document questions inside the same 136-prompt runner.
3. **Teacher-generated benign conversation**: the 480 rows are templates. A
   larger teacher (Qwen2.5-7B/14B on a rented GPU) can generate paraphrased,
   multi-turn benign chat that is verified for safety and style.
4. **Regression monitoring**: run the product suite on every candidate to keep
   the 93.4% / 40/40 / 16/16 profile.
5. Promote a new base only if it beats the 1.5.0 profile; otherwise keep it.

## Infrastructure

- Product suite: `research/product_suite.py` (136 prompts).
- Server suite: `research/server_product_suite.py`.
- Benign: `research/build_benign_prompts.py`, `research/build_benign_chat.py`.
- Safety: `research/build_safety_probes.py`,
  `research/build_safety_replay.py`, `research/build_selfharm_replay.py`,
  `research/build_boundary_pairs.py`.
- Gate: `research/checkpoint_gate.py` (kind-aware `expect`).
- Rented GPU: `scripts/vastai/` + `docs/VASTAI.md`; destroy after every run.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Check for leftover Vast.ai instances and destroy them.
- Release at the next boundary (1.6.0) with fresh posts. Donation addresses stay
  unchanged. Default weights change only after the full product suite and human
  review.
