# Atom Zero 1.2.0: a base that passes the gate and the conversation test

Atom Zero 1.2.0 has a real first: a 63M base that passes the strict checkpoint
gate and the expanded product suite at the same time.

It is not the default yet. The reason is in the numbers.

## The wider suite changed the ranking

1.1.0 used 25 benign prompts. It was enough to expose the refusal collapse, but
not enough to choose a base. 1.2.0 uses **136**: 120 single-turn prompts across
eight categories and 16 two-turn conversations, plus RAG/document questions,
the copy tool and 40 held-out safety probes.

| variant | benign useful | safety | 4short | content | gate | product |
|---|---:|---:|---:|---:|---|---|
| A (safety 1x) | **95.6%** | 39/40 | **61.5%** | **0.759** | fail | pass |
| B (safety 5x) | 78.7% | 40/40 | 38.0% | 0.901 | pass | fail |
| D (per-family) | 68.4% | 40/40 | 53.5% | 0.786 | pass | fail |
| **C (safety 2x)** | **80.9%** | **40/40** | **56.0%** | **0.785** | **pass** | **pass** |

C is the first candidate to pass both. Its full profile:

- 2short 88.0% exact
- 3short 72.0%
- 4short **56.0%** exact / 63.5% key-copy
- content retention **0.785** (better than the 49M base)
- benign useful **80.9%**, refusal 19.1%
- safety **40/40**
- server suite **20/21**, 5/5 source-cited RAG answers

## Why A still matters

A has the best conversation profile: 95.6% benign useful, only 4.4% refusals,
61.5% four-key exact, and content 0.759. It fails exactly one safety probe —
and that failure is serious: a self-harm crisis prompt gets unrelated coding
advice.

That is not a threshold problem. It is a data problem. The next branch is a
**contrastive self-harm replay set**: crisis prompts must get a supportive,
safe response; near-neighbor benign support questions must get a helpful
answer. Generic safety rows do not teach that boundary.

## The decision

C is published as `byjameson/atom-zero-63m-gated`, the first fully validated
optional base:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-gated
```

The product default stays `0.8.0`. C refuses about one in five ordinary
prompts. The promotion bar is **benign useful >= 90%** plus human review. The
49M base still answers every benign prompt.

This is the line between "passes the gate" and "good enough to be the default".
1.2.0 makes the line explicit.

## What ships

- 136-prompt benign suite with multi-turn rows.
- A content-free product runner across benign, safety, copy and content axes.
- A server-level suite for RAG, copy tool and multi-turn.
- Variant C's training recipe: 6000 copy + 492 content + 40 safety x2 +
  144 benign x3.

## Sponsor

The project is independent and measured in public. The sponsor goal is
**$10,000/month** to fund full-time development, rented GPU time, data,
release infrastructure and safety review:

https://github.com/sponsors/byjameson

Tiers start at $5/month; higher tiers cover labs and compute partners.

## Verification

- VM Ubuntu: **1663 tests passed in 46.67s**.
- Benchmark: 20/20.
- Gate C: content 4.154 -> 3.259, exact 0.56, key-copy 0.635, safety 40/40.
- Server C: 20/21, 0 errors, 5 source citations.
- Product base: `0.8.0`, unchanged.

## Install

```powershell
pip install atomzero==1.2.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/1.2.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- 63M gated: https://huggingface.co/byjameson/atom-zero-63m-gated
- Sponsor: https://github.com/sponsors/byjameson
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

A gate pass is necessary. A conversation test is necessary too. 1.2.0 is the
first release where both agree on a candidate.
