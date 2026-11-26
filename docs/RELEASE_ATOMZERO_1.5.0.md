# atomzero 1.5.0 - the first default promotion

Status: **release candidate**. The default base moves to the 63M balanced
checkpoint (weights `0.9.0`); the 49M `0.8.0` base remains the fallback.

## The result

Variant J is the first model to pass the full product bar:

| area | result |
|---|---|
| 2short | 86.0% exact |
| 3short | 75.5% |
| 4short | **48.5% exact / 58.5% key-copy** |
| content retention | **0.703** (better than the 49M base) |
| benign useful (136) | **93.4%** |
| benign refusal | 6.6% |
| safety (40) | **40/40** |
| self-harm gate (16) | **16/16** |
| checkpoint gate | **pass, promotable** |
| product suite | **pass** |
| server suite | **20/21**, 5/5 source-cited RAG |

Promotion bar: benign >= 90%, safety 40/40, self-harm 16/16, copy >= 25%,
retention <= 1.10, server pass. All met.

## What changed

- Benign chat replay expanded from 144 to **480 rows** (60 per category),
  repeated 3x.
- Safety replay 5x and self-harm replay 5x, plus **96 contrastive boundary
  pairs** for injection, secrets, privacy and harmful requests.
- The mix keeps the copy/content core: 6000 copy + 492 content rows.

The previous failure was a balance problem: benign 480x3 alone reached 96.3%
benign but safety fell to 31/40; safety 5x/self-harm 5x restores 40/40 and
16/16 while keeping benign at 93.4%.

## Gate fix

The self-harm gate was scoring benign eval rows as missing refusals because the
eval rows had no `expect` field. `read_safety_rows` now infers `expect` from
`kind` (`crisis` -> refusal, `benign` -> safe), and the builder writes the field
explicitly. This changed J's self-harm result from 8/16 to the correct 16/16,
and the 49M baseline from 11/16 to 11/16 (unchanged).

The 24L self-harm result (9/16) also needs re-reading with the fixed gate; the
1.4.0 conclusion "capacity hurts the boundary" may be partly a scoring artifact.
Treat it as provisional until the 24L run is repeated with the fixed data.

## Packaging

- Default download: `byjameson/atom-zero-63m-balanced` (~279 MB).
- Default cache: `~/.cache/atomzero/atom-zero-63m-balanced`.
- Fallback: `atomzero-quickstart --model-repo byjameson/atom-zero-49m
  --cache ~/.cache/atomzero/atom-zero-49m`.
- Weights version: `0.9.0` (63M balanced); package version `1.5.0`.

## Verification

- VM: 1693 tests passed in 46.87s (rerun below with the final number).
- Benchmark: 20/20.
- Gate J: content 4.154 -> 2.920, exact 0.485, key-copy 0.585, safety 40/40.
- Self-harm J: 16/16.
- Product J: benign 93.4%, safety 100%, copy 48.5%, content 0.703.
- Server J: 20/21, 0 errors, 5 source citations.

## Install

```powershell
pip install atomzero==1.5.0
atomzero-quickstart
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.5.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Sponsor: <https://github.com/sponsors/byjameson>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
