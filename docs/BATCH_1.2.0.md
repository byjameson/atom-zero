# 1.2.0 batch plan - expanded product suite and default-base decision

Date: 2026-11-26

## Goal

1.1.0 found a candidate that passed the 25-prompt product suite. 1.2.0 asks
whether it survives a broader product suite: 120 benign single-turn prompts,
16 multi-turn conversations, RAG/document cases, the copy tool and the strict
safety gate.

## Expanded suite results

| variant | benign useful | refusal | safety | 4short | content | product | server |
|---|---:|---:|---:|---:|---:|---:|---:|
| A (safety 1x) | **95.6%** | 4.4% | 39/40 | **61.5%** | **0.759** | pass | **21/21** |
| B (safety 5x) | 78.7% | 21.3% | **40/40** | 38.0% | 0.901 | fail (benign) | 20/21 |
| D (per-family) | 68.4% | 31.6% | **40/40** | 53.5% | 0.786 | fail (benign) | - |
| C (uniform 2x) | pending | pending | pending | pending | pending | pending | pending |

No candidate yet passes both the expanded product suite and the strict 100%
safety gate. A has the best product profile but misses one genuine self-harm
probe: it answers a crisis prompt with unrelated coding advice. The only
failure is a data problem, not a threshold problem.

## Decision rule

- C must pass safety 40/40 and benign >= 80% on the 136-prompt suite.
- If C passes, prepare 1.2.0 as the first default-base candidate, with the
  49M fallback.
- If C fails, record the A/B/C frontier and keep `0.8.0`; the next branch is a
  dedicated contrastive self-harm replay set (crisis prompts vs. benign
  support questions), not more generic safety rows.

## Evidence

- `research/results/expanded_suite_summary.json`
- `research/results/server_suite_a.json`, `server_suite_b.json`
- `docs/SERVER_PRODUCT_SUITE.md`, `docs/BENIGN_SUITE.md`

## Final result: variant C passes both

| variant | benign useful | refusal | safety | 4short | content | product | gate | server |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 95.6% | 4.4% | 39/40 | 61.5% | 0.759 | pass | fail | 21/21 |
| B | 78.7% | 21.3% | 40/40 | 38.0% | 0.901 | fail | pass | 20/21 |
| D | 68.4% | 31.6% | 40/40 | 53.5% | 0.786 | fail | pass | - |
| **C** | **80.9%** | 19.1% | **40/40** | **56.0%** | **0.785** | **pass** | **pass** | **20/21** |

C (uniform safety 2x + benign 3x) is the first candidate to pass the strict
checkpoint gate and the expanded 136-prompt product suite. Full profile:
2short 88.0%, 3short 72.0%, 4short 56.0%, retention 0.785, server suite 20/21.

It is published as an optional base. The product default stays `0.8.0` because
one in five ordinary prompts is still refused; the promotion bar is benign
useful >= 90% plus human review. A remains the best conversational profile and
needs a dedicated contrastive self-harm replay set.

Evidence: `research/results/variant_c_summary.json`.
