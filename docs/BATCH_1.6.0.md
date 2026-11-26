# 1.6.0 batch plan - teacher-generated benign conversation

Date: 2026-11-26

## Why

1.5.0 promoted the 63M balanced base: benign 93.4%, safety 40/40, self-harm
16/16, 4-key 48.5%, retention 0.703. The next step is to grow the benign corpus
beyond 480 authored templates with teacher-generated conversation.

## Data

1. **Teacher benign chat**: Qwen2.5-0.5B (cached) generates 400 short helpful
   answers across the eight categories. Validation: 20-300 chars, >= 5 words,
   no refusal/unsafe terms (`SafetyFilter`), no duplicate pairs, no personal
   data. Content-free manifest.
2. If the 0.5B teacher is too weak, rent a 24GB GPU and use Qwen2.5-7B with the
   same pipeline.

## Variants

| variant | mix | purpose |
|---|---|---|
| K | J mix (copy 6000 + content 492 + benign 480x3 + safety 40x5 + selfharm 16x5 + boundary 96) + teacher 400x2 | add teacher diversity |
| L | 24L K | capacity + teacher data, single rented run |

Training: 8L, 3000 steps, batch 4, lr 5e-5, content weight 2, seed 71.

## Evaluation

- Expanded product suite: 136 prompts, multi-turn split, RAG cases.
- Safety 40, self-harm 16, server suite, 2/3/4-key copy, retention.
- Promotion bar: beat the 1.5.0 profile (benign 93.4%, safety 40/40,
  self-harm 16/16, copy 48.5%, retention 0.703) without regressing the
  multi-turn/RAG checks.

## 24L re-evaluation

The 1.4.0 24L self-harm result (9/16) was computed with the old gate; with
kind-aware `expect` it is 8 benign passes + 1 crisis pass = 1/8 crisis refusals.
The 24L was trained on the old F mix (selfharm 1x). L retrains it on the J/K
mix with 5x self-harm and boundary pairs, so it is the correct re-test.

Use `scripts/vastai/launch.py --copy-checkpoint` and verify the checkpoint
size before destroying. If the transfer drops, upload to HF from the instance
with an explicit token or split the transfer.

## Results

| variant | mix | benign | safety | self-harm | 4-key | retention | multi-turn | gate |
|---|---|---:|---:|---:|---:|---:|---|---|
| J | 1.5.0 default | 93.4% | 40/40 | 16/16 | 48.5% | 0.703 | 75% / 50% | pass |
| K | teacher 2x, safety 5x | **98.5%** | 39/40 | 14/16 | 48.5% | 0.808 | **93.8% / 75%** | fail |
| K2 | safety/self-harm 8x | 89.0% | **40/40** | **16/16** | **53.5%** | 0.816 | 62.5% / 56.3% | pass |
| K3 | midpoint 6x, teacher 2x | 95.6% | **40/40** | **16/16** | 45.0% | 0.828 | 68.8% / 37.5% | pass |

K3 passed the checkpoint gate and the product suite but failed the promotion bar
on copy, retention and multi-turn. No variant beat J on the full profile.

## Decision

Keep the 63M balanced default (weights `0.9.0`) and the 49M `0.8.0` fallback.
Publish 1.6.0 as a tooling/frontier release: teacher benign generator, expanded
product suite, and the K/K2/K3 evidence. Do not start the 24L run; it was gated
on a winning 8L mix and no mix won.

## Verification

- Gate K3: content 4.154 -> 3.438, exact 0.450, key-copy 0.555, safety 40/40.
- Self-harm K3: 16/16.
- Product K3: benign 95.6%, safety 100%, copy 45.0%, content 0.828, RAG 100%,
  multi-turn 68.8% / 37.5%.
- VM: 1717 tests passed in 66.77s; benchmark 20/20.
