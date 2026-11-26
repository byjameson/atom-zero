# Atom Zero 1.1.0: the missing data was a conversation

Atom Zero 1.1.0 fixes the failure that 1.0.0 made visible. The 63M model that
refused ordinary conversation now has a variant that passes both the strict
checkpoint gate and the product suite.

The fix was not more safety data. It was ordinary conversation.

## The ratio trap

At 1.0.0 the 63M model passed every gate and then refused 24 of 25 benign
prompts. The obvious hypothesis was too much safety replay. So the ratio was
swept:

| safety ratio | benign useful | safety |
|---|---:|---:|
| 1x | 32% | 95% |
| 2x | 4% | 100% |
| 10x | 4% | 100% |

Even 1x — 40 safety rows, 0.6% of the mix — was enough to collapse conversation.
The problem was not the amount. The mix had 6000 copy rows and 492 factual
content rows, but almost no examples of an ordinary helpful answer. The model
learned a simple rule: if the prompt is not grounded or copy-routed, refuse.

## The fix: benign chat replay

144 short, helpful answers across eight categories: identity, greeting,
explanation, cooking, travel, coding, planning and general knowledge. Repeated
three times in the mix. Two variants:

| variant | safety | benign | 2short | 3short | 4short | retention | benign useful | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| A | 1x | 3x | 85.0% | 79.5% | **61.5%** | **0.759** | **92%** | 39/40 |
| B | 5x | 3x | 86.5% | 71.5% | 38.0% | 0.901 | 88% | **40/40** |

**A** is the better product: 92% benign useful, only 8% refusal, 61.5% four-key
exact, content loss 0.759x the 49M base. It misses one of 40 safety probes.

**B** passes everything: 88% benign useful, 40/40 safety, 38.0% four-key exact,
content 0.901. It is the first candidate to pass both the strict gate and the
product suite.

## What this says about the loop

The 1.0.0 result was not a dead end. It identified the exact missing variable.
The 1.1.0 experiment changed one part of the data, kept everything else fixed,
and measured both axes — safety and usefulness — instead of one.

That is the self-improvement loop working at the level the project can currently
support: a failure becomes a measurable hypothesis, a branch, an experiment and
a gate decision.

## Decision

B is published as `byjameson/atom-zero-63m-benign`, the first fully validated
optional base:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-benign
```

The product default stays `0.8.0` for one more product-suite expansion and human
review. A is the candidate for the next iteration.

## New tooling

- A deterministic 25-prompt benign suite.
- A content-free product runner with benign, safety, copy and content metrics.
- A 144-row benign chat replay builder.
- `product_ok` combines benign useful rate, safety pass rate and copy exact —
  three axes that a single loss number cannot see.

## Verification

- VM Ubuntu: **1631 tests passed in 47.30s**.
- Benchmark: 20/20.
- Gate B: content 4.154 -> 3.740, exact 0.38, key-copy 0.46, safety 40/40.
- Product B: benign 88%, safety 100%, copy 38%, content 0.901.
- Product base: `0.8.0`, unchanged.

## Install

```powershell
pip install atomzero==1.1.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/1.1.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- 63M benign research: https://huggingface.co/byjameson/atom-zero-63m-benign
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The model did not need to be bigger. It needed to be taught that a conversation
is not a threat.
