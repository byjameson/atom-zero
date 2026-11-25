# Next session - 1.1.0 planning

Snapshot: 2026-11-26, after the 1.0.0 release.
Tag `v1.0.0`; main includes the publication commit.
No Vast.ai instance is running (verified empty); credit about **$9.54**.

## Released

`1.0.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 1.0.0` reports
**0 failures**. CI run `45707191185` is green.

Core result: the 63M mixed-replay candidate passed every checkpoint gate but
failed the product suite. Same server, same corpus, ten prompts:

| area | 49M base | 63M candidate |
|---|---|---|
| grounded document questions | 3/3 | 3/3 |
| copy tool | 1/1 | 1/1 |
| benign open prompts | 1/5 useful | **0/5, 4 refusals** |
| safety redteam | 72/72 | 72/72 |

The candidate is an optional research base. The product default stays `0.8.0`.

## The 1.1.0 experiment

Find the safety-replay ratio that passes the safety gate **without** refusing
benign open prompts.

1. Generate the safety replay split with ratios 2x, 5x, 10x (current), 20x and
   a template-diverse variant.
2. For each: train the 8L mixed model, run the checkpoint gate AND a benign
   prompt suite (20-30 ordinary prompts: identity, explanation, greeting,
   cooking, travel, coding help).
3. Score two axes:
   - safety gate pass rate on held-out probes,
   - benign-prompt useful rate (no refusal, on-topic, non-empty).
4. Pick the smallest safety ratio that keeps the gate pass and restores at
   least 80% useful benign answers.
5. Only then consider a default-base change; otherwise publish the finding and
   keep `0.8.0`.

Add a `research/benign_prompts.py` builder or a fixed JSONL with 25 prompts and
a `research/product_suite.py` runner that reports both axes content-free. The
current ad-hoc product test lives in `research/results/product_suite_1.0_summary.json`.

## Infrastructure

- Vast.ai: `scripts/vastai/` + `docs/VASTAI.md`; preflight, bundle, detached
  job, automatic destroy. Use only for jobs that do not fit locally.
- Pretokenized cache: `research/pretokenize.py` + `--cache-dir`.
- Mixed replay: `research/mix_training_data.py`.
- Content rows: `research/build_content_rows.py`.
- Safety probes/replay: `research/build_safety_probes.py`,
  `research/build_safety_replay.py`.
- Retention: `research/retention_eval.py`; gate:
  `research/checkpoint_gate.py` (supports `--device cuda`).
- Optional bases: `atomzero-quickstart --model-dir` / `--model-repo`;
  `docs/OPTIONAL_BASES.md`.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Check for leftover Vast.ai instances and destroy them.
- Release only at the next 10-step boundary (1.1.0), with new posts on every
  surface. Donation addresses and product weights stay unchanged unless a new
  base passes the product suite.
