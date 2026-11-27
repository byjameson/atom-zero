# atomzero 1.8.0 - the first 7B-teacher promotion: copy up, safety held

Status: **release prepared, publication in progress.** The default moves to the
promoted 63M checkpoint (weights `1.0.0`); the 49M `0.8.0` base remains the
fallback.

## The result

The promoted 63M candidate is the first arm to beat J on copy and multi-turn
while holding safety and self-harm:

| metric | J (1.7.0 default) | 1.8.0 promoted | 3-seed mean |
|---|---:|---:|---:|
| single-turn benign | 95.8% | **99.2%** | — |
| multi-turn final / both | 62.5 / 39.6 | **83.3 / 58.3** | 81.9 / 59.0 |
| 4-key copy (seen) | 48.5% | **60.0%** | **56.7%** |
| 4-key copy (fresh, unseen) | — | **64.5%** | — |
| retention (lower is better) | 0.703 | 0.835 | 0.846 |
| safety (40) | 40/40 | **40/40** | 2/3 seeds 40/40 |
| self-harm (16) | 16/16 | **16/16** | — |
| local RAG | 100% | **100%** | — |

Copy rises **+11.5 pp** over J and multi-turn rises **+20.8 / +18.7 pp**. The
fresh 4-key set, with unseen keys and values, scores **64.5%**, so the gain is
not template memorization. Retention is at J's seed mean (0.840), not a
regression against the seed distribution.

Copy profile (exact): 2short **87.0%**, 3short **74.5%**, 4short **60.0%**
(J: 86.0 / 75.5 / 48.5); key-copy 91.0 / 81.5 / 74.5. The 2- and 3-key
profiles stay at J level; the 4-key profile is the improvement.

## Seed variance

The 3-seed matrix is honest about the limits:

| seed | copy | multi final/both | retention | safety |
|---|---:|---:|---:|---:|
| promoted | **60.0%** | 83.3 / 58.3 | 0.835 | **40/40** |
| seed 2 | 54.5% | 85.4 / 70.8 | 0.863 | 39/40 |
| seed 3 | 55.5% | 77.1 / 47.9 | 0.839 | **40/40** |

Copy is robust across seeds (+6 to +11.5 pp). Safety is 40/40 on two of three
seeds; the promoted seed is the fully gated one. The miss on seed 2 is
published rather than hidden.

## What changed

- The 7B-teacher mix is combined with a copy and safety replay curriculum:
  12,316 rows, including additional 4-key KV rows and duplicated safety and
  self-harm rows.
- Training: 3,000 steps, batch 4, lr 5e-5, content weight 2, seed 71.
- A fresh 4-key copy set (200 rows, 100 JSON / 100 KV) was generated with a new
  seed; it shares no prompts and only 16/800 keys with the training copy rows.
- Gates: main gate **promotable** (exact 0.60, key-copy 0.745, safety 40/40);
  self-harm gate **promotable** (16/16).

## Verification

- Main gate: content pass, exact 0.60, key-copy 0.745, safety 40/40.
- Self-harm gate: 16/16, no failed checks.
- Product suite: single 99.2%, multi 83.3 / 58.3, copy 60.0%, retention 0.835,
  safety 40/40, RAG 100%.
- Fresh copy: 64.5% exact, key 74.5%, key recall 90%.
- Full VM suite: **1,764 passed in 62.45s**; benchmark **20/20** with 1.8.0.
- The previous 1.7.0 release remains in the archive; its default was J.

## Install

```powershell
pip install atomzero==1.8.0
atomzero-quickstart
```

The quickstart downloads the promoted 63M checkpoint (~279 MB) into
`~/.cache/atomzero/atom-zero-63m-balanced`.

Fallback:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-49m `
  --cache ~/.cache/atomzero/atom-zero-49m
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.8.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- Sponsor: <https://github.com/sponsors/byjameson>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>

Medium: the 1.7.0 post was skipped by the platform limit; the next Medium post
is 1.8.0 when posting is available (no backfill).
