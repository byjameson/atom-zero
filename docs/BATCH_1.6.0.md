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

## Release

1.6.0 only if K or L beats the 1.5.0 profile; otherwise record the frontier and
keep the 63M balanced default.
