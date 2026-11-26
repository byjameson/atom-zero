# atomzero 1.1.0 - benign chat replay fixes the refusal collapse

Status: **published on 2026-11-26**. Product base weights remain `0.8.0`; the 63M
benign-replay candidate is the first optional base that passes both the strict
gate and the product suite.

## The problem

At 1.0.0 the 63M mixed-replay model passed every checkpoint gate and then
refused ordinary conversation: 4% useful / 96% refusals on a 25-prompt benign
suite. The safety-replay ratio alone did not fix it:

| safety ratio | benign useful | safety |
|---|---:|---:|
| 1x | 32% | 95% |
| 2x | 4% | 100% |
| 10x | 4% | 100% |

Even 0.6% safety data collapsed benign answers. The mix had copy rows and
factual content, but no ordinary conversational examples.

## The fix

Add **benign chat replay**: 144 short, helpful answers across identity,
greeting, explanation, cooking, travel, coding, planning and general knowledge,
repeated 3x in the mix. Two variants:

| variant | safety | benign | 2short | 3short | 4short | retention | benign useful | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 1x | 3x | 85.0% | 79.5% | **61.5%** | **0.759** | **92%** | 39/40 |
| B | 5x | 3x | 86.5% | 71.5% | 38.0% | 0.901 | 88% | **40/40** |

- **A** has the best product profile: 92% benign useful, 8% refusal, 61.5%
  four-key exact, content loss 0.759x the 49M base. It fails one of 40 safety
  probes under the strict 100% gate.
- **B** passes everything: 88% benign useful, 12% refusal, 40/40 safety, 38.0%
  four-key exact, content 0.901. It is the first candidate to pass both the
  strict checkpoint gate and the product suite.

## Decision

B is published as `byjameson/atom-zero-63m-benign`, the first fully validated
optional base. The product default stays `0.8.0` for one more product-suite
expansion and human review; A is the candidate for the next iteration.

## New tooling

- `research/build_benign_prompts.py`: deterministic 25-prompt benign suite.
- `research/product_suite.py`: content-free runner with `benign_useful_rate`,
  `benign_refusal_rate`, `safety_pass_rate`, `copy_exact`, `content_loss_ratio`
  and `product_ok`.
- `research/build_benign_chat.py`: 144 benign chat replay rows across eight
  categories.
- `docs/PRODUCT_SUITE.md`, `docs/BENIGN_CHAT.md`.

## Verification

- VM: **1631 tests passed in 47.30s** (before the final merge; rerun below).
- Benchmark: 20/20.
- Gate B: content 4.154 -> 3.740, exact 0.38, key-copy 0.46, safety 40/40.
- Product B: benign 88%, safety 100%, copy 38%, content 0.901.

## Install

```powershell
pip install atomzero==1.1.0
```

The optional base:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-benign
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.1.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-110-the-missing-data-was-a-conversation-224c>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-1-1-0-the-missing-data-was-a-conversation-bffa1319a0f7>
- 63M benign base: <https://huggingface.co/byjameson/atom-zero-63m-benign>
