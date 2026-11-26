# atomzero 1.2.0 - first base to pass the strict gate and the expanded product suite

Status: **release candidate**. Product base weights remain `0.8.0`.

## The result

Variant C — uniform safety replay 2x + benign chat replay 3x on the 8-layer 63M
base — is the first candidate to pass both:

- the strict checkpoint gate (content not worse, exact match, key-copy, safety
  40/40),
- the expanded 136-prompt product suite (120 single-turn + 16 multi-turn),
- and the server-level suite (RAG, copy tool, multi-turn; 20/21).

| area | result |
|---|---|
| 2short copy (200 rows) | 88.0% exact / 88.0% key-copy |
| 3short copy (200 rows) | 72.0% / 76.0% |
| 4short copy (200 rows) | **56.0% / 63.5%** |
| content retention | **0.785** (better than the 49M base) |
| benign useful (136) | **80.9%** |
| benign refusal | 19.1% |
| safety (40 held-out) | **40/40** |
| server suite | **20/21**, 5/5 source-cited RAG |
| checkpoint gate | **pass, promotable** |

## The expanded suite

1.1.0 used 25 benign prompts. 1.2.0 uses 136: 120 single-turn across eight
categories and 16 two-turn conversations, plus RAG/document questions, the copy
tool and held-out safety probes.

The wider suite changed the ranking:

| variant | benign useful | safety | 4short | gate | product |
|---|---:|---:|---:|---|---|
| A (safety 1x) | **95.6%** | 39/40 | **61.5%** | fail | pass |
| B (safety 5x) | 78.7% | 40/40 | 38.0% | pass | fail |
| D (per-family) | 68.4% | 40/40 | 53.5% | pass | fail |
| **C (safety 2x)** | **80.9%** | **40/40** | **56.0%** | **pass** | **pass** |

A still has the best conversation profile but answers one self-harm crisis
prompt with unrelated coding advice. That is a data problem, not a threshold
problem; the next branch is a dedicated contrastive self-harm replay set.

## Decision

C is published as `byjameson/atom-zero-63m-gated`, the first fully validated
optional base:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-gated
```

The product default stays `0.8.0`. C refuses about one in five ordinary
prompts; the promotion bar is benign useful >= 90% plus human review. The 49M
base remains the safest default.

## New tooling

- 136-prompt benign suite with multi-turn rows (`research/build_benign_prompts.py`).
- Content-free product runner across benign, safety, copy and content axes.
- Server-level suite for RAG, copy tool and multi-turn (`research/server_product_suite.py`).
- Variant C training mix: 6000 copy + 492 content + 40 safety x2 + 144 benign x3.

## Verification

- VM: 1663 tests passed in 46.67s; benchmark 20/20.
- Gate C: content 4.154 -> 3.259, exact 0.56, key-copy 0.635, safety 40/40.
- Server C: 20/21, 0 errors, 5 source citations.
- Sponsor goal: $10,000/month for full-time development.

## Install

```powershell
pip install atomzero==1.2.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.2.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Sponsor: <https://github.com/sponsors/byjameson>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
