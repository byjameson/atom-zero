# 1.4.0 batch plan - 24L capacity on the F recipe

Date: 2026-11-26

## Why

At 63M, contrastive self-harm replay fixes the crisis probe but costs 7-20
points of benign conversation (A 95.6% / 39/40; F 83.8% / 40/40). The 63M model
cannot separate the boundary. 1.4.0 tests whether **capacity** can.

## Experiment

- Base: 24-layer upcycle, 176,795,136 params.
- Data: F recipe — 6000 copy + 492 content + 40 safety + 144 benign x3 +
  16 contrastive self-harm rows. Pretokenized cache.
- Training: 3000 steps, batch 16, lr 5e-5, content weight 2, seed 71.
- Hardware: Vast.ai RTX 3090 24GB, about $0.18/hour, instance destroyed after.

## Evaluation (on the instance)

1. 2/3/4-key copy, 200 held-out rows each.
2. Retention against the 49M base.
3. Checkpoint gate (content, 4short copy, 40 safety probes).
4. Self-harm gate (16 held-out crisis/benign rows).
5. 136-prompt product suite.
6. Server suite (RAG, copy, multi-turn).

## First run (2026-11-26)

Copy results:

| stage | exact | key-copy |
|---|---:|---:|
| 2short | **95.5%** | 95.5% |
| 3short | **86.5%** | 90.0% |
| 4short | **77.5%** | 81.0% |
| retention | **0.790** | pass |

That is the best copy profile so far (8L F: 88.0 / 72.0 / 58.0; 24L copies:
92.3 / 83.5 / 69.5).

The first evaluation run failed because `data/safety_replay_v1/eval.jsonl` was
not in the bundle: the checkpoint gate and product suite returned a
`FileNotFoundError` report, and only the self-harm gate (9/16) and server suite
(pass) ran. The bundle and launcher were fixed to include the file and copy the
trained checkpoint back.

## Promotion bar

benign useful >= 90%, safety 40/40, 4short >= 25%, retention <= 1.10, server
suite pass, full VM suite and human review. If 24L passes, promote it as the
1.4.0 default with the 49M fallback; otherwise expand the benign corpus.
