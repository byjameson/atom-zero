# Next session - 1.2.0 planning

Snapshot: 2026-11-26, after the 1.1.0 release.
Tag `v1.1.0`; main includes the publication commit.
No Vast.ai instance is running (verified empty); credit about **$9.54**.

## Released

`1.1.0` is published and verified on PyPI, GitHub, Hugging Face, ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 1.1.0` reports
**0 failures**. CI run `45729474401` is green.

Key result: benign chat replay fixes the 1.0.0 refusal collapse.

| variant | safety | benign | 2short | 3short | 4short | retention | benign useful | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 1x | 3x | 85.0% | 79.5% | **61.5%** | **0.759** | **92%** | 39/40 |
| B | 5x | 3x | 86.5% | 71.5% | 38.0% | 0.901 | 88% | **40/40** |

B is the first candidate to pass both the strict gate and the product suite.
It is published as `byjameson/atom-zero-63m-benign`, an optional base.
Default remains `0.8.0`.

## 1.2.0 decision

The default-base decision now requires a broader product suite, not a new
architecture:

1. **Expand benign coverage** from 25 to 100+ prompts, including multi-turn
   greetings, follow-ups, ambiguity and "I don't know" cases.
2. **Add RAG/doc and tool scenarios** to the product runner (currently the
   document benchmark is model-independent and the copy tool answers before
   generation).
3. **Fix variant A's one safety probe** without losing its 92% benign / 61.5%
   four-key profile. Check which family the probe belongs to and whether the
   replay templates cover it.
4. **Human review** of the self-harm and prompt-injection families.
5. If A or B passes the expanded suite, make it the default in 1.2.0 and keep
   the 49M as a fallback. Otherwise keep `0.8.0` and publish the expanded
   evidence.

## Infrastructure

- Product suite: `research/product_suite.py` (`product_ok` = benign >= 0.8,
  safety >= 0.9, copy exact >= 0.25).
- Benign data: `research/build_benign_prompts.py`,
  `research/build_benign_chat.py`.
- Safety: `research/build_safety_probes.py`,
  `research/build_safety_replay.py`.
- Mixed replay: `research/mix_training_data.py`.
- Content rows: `research/build_content_rows.py`.
- Retention: `research/retention_eval.py`; gate:
  `research/checkpoint_gate.py`.
- Optional bases: `atomzero-quickstart --model-dir` / `--model-repo`;
  `docs/OPTIONAL_BASES.md`.
- Rented GPU: `scripts/vastai/` + `docs/VASTAI.md`; destroy after every run.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Check for leftover Vast.ai instances and destroy them.
- Release at the next boundary (1.2.0) with fresh posts. Donation addresses stay
  unchanged. Default weights change only after the expanded product suite and
  human review.
