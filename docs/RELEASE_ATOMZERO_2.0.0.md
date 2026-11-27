# atomzero 2.0.0 - history-aware context adapter

Status: **published on 2026-11-28.** The default moves to the 63M `1.2.0` weights;
the 49M `0.8.0` base remains the fallback and the 1.9.0 63M weights stay
available as the previous checkpoint.

## The result

2.0.0 adds a history-conditioning adapter to 1.9.0. On 200 held-out
counterfactual conversations it recalls names, pets, cities, preferences,
days and language corrections **65.0%** of the time, up from **10.0%** on
1.9.0.

| metric | 1.9.0 | 2.0.0 |
|---|---:|---:|
| single-turn benign | 94.2% | **97.5%** |
| multi-turn final / both | 93.8 / 68.8 | **83.3 / 58.3** |
| 4-key copy (seen) | 59.5% | **63.0%** |
| 4-key copy (fresh, unseen) | 67.0% | **75.0%** |
| retention (lower is better) | 0.770 | **0.700** |
| history recall (held-out) | 10.0% | **65.0%** |
| safety (40) | 40/40 | **40/40** |
| self-harm (16) | 16/16 | **16/16** |
| local RAG | 100% | **100%** |

The trade-off is the automatic multi-turn **both-turns** score, 68.8% ->
58.3%. That metric counts non-refusal answers with at least four words and
does not measure coherence. The safety patch also makes the model more
cautious on ambiguous identity and greeting turns. Single-turn, copy,
retention and history recall all improve.

## What changed

- A history-conditioning LoRA is trained on 2,000 counterfactual two-turn
  pairs whose final answer depends on an earlier turn, with 50% replay from
  the KV4s mix.
- A small safety adapter is trained on new crisis phrasings plus benign
  support contrast, so ordinary concern is not mistaken for a crisis.
- The adapters are merged into the 1.9.0 checkpoint; no architecture change.

## Verification

- Product suite: single 97.5%, multi 83.3 / 58.3, copy 63.0%, retention
  0.700, safety 40/40, RAG 100%, product `ok`.
- Main gate: **promotable** (exact 63.0, key-copy 73.0, content improves,
  safety 40/40).
- Self-harm gate: **promotable**, 16/16.
- Fresh unseen 4-key copy: **75.0%** exact (150/200), key recall **92.0%**.
- VM suite: 1764 tests passed, benchmark 20/20.

## Release surfaces

- PyPI: `atomzero 2.0.0`
- GitHub: `v2.0.0`
- Hugging Face: `byjameson/atom-zero-63m-balanced`, weights `1.2.0`
- ModelScope: `byjameson/atom-zero-63m-balanced`
- ModelScope Studio: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-200-a-history-aware-context-adapter-1k0j>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7532210386427703296/>
- X: <https://x.com/aigokhan/status/2126444828980371675>
- Medium: the 24-hour limit is active; no backfill. The current 2.0.0 post is
  scheduled for the next window.
