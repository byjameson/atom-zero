# Next session - 1.0.0 planning

Snapshot: 2026-11-26, after the 0.90.0 release.
Main HEAD at release: `174d562`; tag `v0.90.0`.
No Vast.ai instance is running (verified empty); credit about **$9.54**.

## Released

`0.90.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 0.90.0` reports
**0 failures**. CI run `45705068427` on `174d562` is green.

Key results:

- 24-layer / 176.8M upcycle: 4short **69.5%** exact (two seeds) but retention
  **1.664** (fail). Depth rejected.
- 8-layer 63M + copy/content/safety replay: two seeds pass all four checkpoint
  gates.

| seed | 4short exact | key-copy | retention | safety |
|---|---:|---:|---:|---:|
| 71 | 57.0% | 67.0% | 0.722 | 40/40 |
| 72 | 58.0% | 63.5% | 0.809 | 40/40 |

- Candidate: <https://huggingface.co/byjameson/atom-zero-63m-mixed>
- Evidence: `research/results/mixed8_safety_summary.json`
- Product base remains `0.8.0`.

## What 1.0.0 should decide

The 63M candidate is marked **promotable** by the checkpoint gate. The next
boundary is the first real product-base decision:

1. **Validate general behavior, not only loss.** Run the candidate through the
   chat, RAG, documents, answerability and safety suites. Content loss is
   better than the base, but exact QA is still 0 for both on city rows.
2. **Package a second base.** The download path, cache layout, quickstart,
   doctor and model card must support a 63M default without breaking the 49M
   fallback.
3. **Widen safety evaluation.** The 40/40 result uses disjoint synthetic
   templates. Add paraphrased, multilingual and multi-turn probes, plus human
   review of self-harm cases.
4. **Grow the content replay corpus.** 492 rows from 164 English facts is a
   proof of concept. The mixed recipe needs a larger, licensed corpus.
5. **Decide the default.** Keep `0.8.0` unless the candidate passes the full
   product suite and a human review; otherwise publish it as an optional base.

## Infrastructure

- Vast.ai pipeline: `scripts/vastai/` + `docs/VASTAI.md`. Preflight, bundle,
  detached job, automatic destroy. The 24L run cost $0.42 for 2.2 hours.
- Pretokenized cache: `research/pretokenize.py`, `--cache-dir` in the trainer.
  Use it for any paid run.
- Mixed replay: `research/mix_training_data.py`.
- Content rows: `research/build_content_rows.py`.
- Safety: `research/build_safety_probes.py`, `research/build_safety_replay.py`.
- Retention: `research/retention_eval.py`.
- Gate: `research/checkpoint_gate.py` (supports `--device cuda`).

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Destroy rented instances even on failure; check for leftovers.
- Release only at the 10-minor boundary, with fresh posts on every surface.
- Donation addresses stay unchanged.
