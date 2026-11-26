# 1.3.0 batch plan - contrastive self-harm replay and the default decision

Date: 2026-11-26

## Why

1.2.0 produced the first base (variant C) that passes the strict gate and the
136-prompt product suite. It is still not the default because one in five
ordinary prompts is refused (benign 80.9%).

Variant A is the best product profile — 95.6% benign, 61.5% four-key exact,
retention 0.759 — but fails one safety probe: a self-harm crisis prompt gets
unrelated coding advice. The failure is a data problem, not a threshold problem.

## Experiment

Add **contrastive self-harm replay**:

- crisis prompt -> supportive safe answer with an explicit refusal marker and
  crisis guidance (no methods, no graphic content),
- near-neighbor benign support prompt -> helpful, non-refusal answer.

Mix into the A recipe:

```text
copy 6000 + content 492 + safety 40 x1 + benign 144 x3 + self-harm pairs x3
```

Train the 8L 63M base, 3000 steps, batch 4, lr 5e-5, content weight 2, seed 71,
pretokenized cache.

## Evaluation

1. `research/checkpoint_gate.py` — content, 4short copy, 40 held-out safety.
2. `research/product_suite.py` — 136 benign prompts, safety, copy, content.
3. `research/server_product_suite.py` — RAG, copy tool, multi-turn.
4. 2short/3short copy and retention.

## Promotion bar

- safety **40/40**
- benign useful **>= 90%**
- 4short copy **>= 25%**
- content retention **<= 1.10**
- server suite pass
- full VM suite and human review

If the candidate passes, 1.3.0 promotes it as the default base with the 49M
fallback. If it fails, record the frontier and keep `0.8.0`; the next branch is
a larger benign corpus or a different safety objective, not more generic
safety rows.

## Evidence so far

| variant | benign useful | safety | 4short | retention | gate | product |
|---|---:|---:|---:|---:|---|---|
| A | 95.6% | 39/40 | 61.5% | 0.759 | fail | pass |
| B | 78.7% | 40/40 | 38.0% | 0.901 | pass | fail |
| C | 80.9% | 40/40 | 56.0% | 0.785 | pass | pass |
| D | 68.4% | 40/40 | 53.5% | 0.786 | pass | fail |
| E | pending | pending | pending | pending | pending | pending |

A + contrastive self-harm replay is the shortest path to a full pass with the
best conversation profile.
