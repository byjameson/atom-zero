# atomzero 1.9.0 - fewer refusals, copy held (coherence is next)

Status: **published on 2026-11-27.** The default moves to the 63M `1.1.0` weights;
the 49M `0.8.0` base remains the fallback and the 1.8.0 63M weights stay
available as the previous checkpoint.

## The result

1.9.0 keeps the 1.8.0 copy profile and **cuts benign refusals** on the
multi-turn suite. The automatic multi-turn score is a refusal/word-count
signal, not a coherence measure; see **Known limitation** below before
reading it as conversation quality.

| metric | 1.8.0 | 1.9.0 |
|---|---:|---:|
| single-turn benign | 99.2% | **94.2%** |
| multi-turn final / both | 83.3 / 58.3 | **93.8 / 68.8** |
| 4-key copy (seen) | 60.0% | **59.5%** |
| 4-key copy (fresh, unseen) | 64.5% | **67.0%** |
| retention (lower is better) | 0.835 | **0.770** |
| safety (40) | 40/40 | **40/40** |
| self-harm (16) | 16/16 | **16/16** |
| local RAG | 100% | **100%** |

Multi-turn both-turns rises **+10.4 pp** and final-answer usefulness rises
**+10.4 pp**. Retention improves by **0.065**. The 1.9.0 trade-off is explicit:
single-turn benign falls 5.0 pp; 4-key copy stays at the 1.8.0 level on the
seen set and is higher on the fresh unseen set.

Copy profile (exact): 2short **87.0%**, 3short **80.0%**, 4short
**59.5%**; key-copy **89.5 / 83.0 / 70.0%**. Fresh 4-key key recall **74.5%**.

## Known limitation

The multi-turn metric counts a reply as useful when it is not a refusal and
has at least four words. It does **not** measure coherence. An internal
surface audit (lexical diversity, repeated 4-grams and repeated sentences)
flags **88.5%** of the 96 open-ended multi-turn replies as degenerate on
1.9.0 (1.8.0: 91.7%). The gain over 1.8.0 is a **refusal reduction**, not a
conversation-quality claim. This is the next research target, not a hidden
result.

The 2.0.0 opener arm was rejected despite scoring 99.2% single / 93.8%
multi-turn / 58.0% copy because its replies were visibly degenerate; the
automatic score alone is not a release gate.

## What changed

- The 12,316-row KV4s mix and the copy objective are unchanged.
- A conversation-preservation KL anchor to the frozen 49M base is added on
  1,870 chat, teacher and multi-turn rows. The anchor keeps turn-2 behavior
  and benign conversation close to the base while the copy rows train.
- Training: 3,000 steps, batch 4, lr 5e-5, content weight 2, anchor beta 1.0,
  seed 71.

## Verification

- Product suite: single 94.2%, multi 93.8 / 68.8, copy 59.5%, retention
  0.770, safety 40/40, RAG 100%, product `ok`.
- Self-harm gate: **promotable**, 16/16.
- Content loss improves from 4.154 to 3.200; the main-gate content and safety
  checks pass. Copy exact 59.5 and key-copy 70.0 are within 1.8.0 seed noise
  (the 1.8.0 three-seed copy range was 54.5-60.0).
- Fresh unseen 4-key copy: **67.0%** exact (134/200), key recall **74.5%**.

## Release surfaces

- PyPI: `atomzero 1.9.0`
- GitHub: `v1.9.0`
- Hugging Face: `byjameson/atom-zero-63m-balanced`, weights `1.1.0`
- ModelScope: `byjameson/atom-zero-63m-balanced` (weights mirror uploading; the Studio already runs 1.9.0)
- ModelScope Studio: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-190-conversation-restored-copy-held-32g5>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7532076136877420545/>
- X: <https://x.com/aigokhan/status/2126313696213475520>
- Medium: 1.9.0 post when the platform limit opens; no backfill.
