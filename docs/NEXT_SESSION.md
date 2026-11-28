# Next session - 2.2.0 released

Snapshot: 2026-11-28, **2.2.0 release**. Weights stay `1.2.0` (63M); the
release adds the deterministic recall checker next to the 2.1.0 budget/time
checker. No training is running; GPU idle. No Vast.ai instance is running.
Sponsor goal: **$10,000/month**.

## Released baseline

`2.2.0` is the current boundary. The default remains the **63M** checkpoint
(`byjameson/atom-zero-63m-balanced`, weights `1.2.0`); the 49M `0.8.0` base is
the fallback and the 1.9.0 63M weights stay available as the previous
checkpoint.

## 2.2.0 result

| metric | 2.1.0 | 2.2.0 |
|---|---:|---:|
| routed context (200 held-out) | 82.5% | **100%** |
| natural held-out (276) | 91.3% | **100%** |
| model-only context (200) | 65.0% | 65.0% |
| single-turn benign | 97.5% | **97.5%** |
| multi-turn final/both | 83.3 / 58.3 | **83.3 / 58.3** |
| 4-key copy (seen) | 63.0% | **63.0%** |
| safety | 40/40 | **40/40** |
| self-harm | 16/16 | **16/16** |
| RAG | 100% | 100% |

- `src/atomzero/recall_router.py` answers explicit name/city/language/
  preference/day/pet questions from the facts stated earlier in the
  conversation; the API marks it `source: recall_tool`.
- All eight context families are **25/25** on the routed path; the model alone
  still scores 65.0%, and the card keeps both numbers separate.
- False positives: **0/168** benign_v2 prompts, **0/16** self-harm prompts.
- Product suite: **pass** (single 97.5, multi 83.3/58.3, copy exact 63.0,
  safety 40/40, content ratio 0.700).
- Weights are unchanged. Recall and arithmetic are runtime tools, not model
  skills, and the model card says so.

### Previous: 2.1.0

2.1.0 added the budget/time checker and lifted routed context 65.0% -> 82.5%
(budget 4/25 -> 25/25, time 11/25 -> 25/25) with 91.3% on 276 natural
phrasings. Weights were unchanged.

## 2.0.0 result (previous boundary)

| metric | 1.9.0 | 2.0.0 |
|---|---:|---:|
| single-turn benign | 94.2% | **97.5%** |
| multi-turn final/both | 93.8 / 68.8 | **83.3 / 58.3** |
| 4-key copy (seen) | 59.5% | **63.0%** |
| 4-key copy (fresh) | 67.0% | **75.0%** |
| retention | 0.770 | **0.700** |
| history recall (held-out) | 10.0% | **65.0%** |
| safety | 40/40 | **40/40** |
| self-harm | 16/16 | **16/16** |

2.0.0 added a history-conditioning LoRA to 1.9.0 plus a small safety adapter
with benign-support contrast. Main gate **promotable** (exact 63.0, key 73.0);
self-harm gate **promotable** (16/16); fresh 4-key copy **75.0%**. The
automatic multi-turn both-turns score falls 68.8 -> 58.3 because the safety
adapter is more cautious on ambiguous identity/greeting turns; the metric is
refusal/word-count only, not coherence.

## 1.9.0 result

| metric | 1.8.0 | 1.9.0 |
|---|---:|---:|
| single-turn benign | 99.2% | **94.2%** |
| multi-turn final/both | 83.3 / 58.3 | **93.8 / 68.8** |
| 4-key copy (seen) | 60.0% | **59.5%** |
| 4-key copy (fresh) | 64.5% | **67.0%** |
| retention | 0.835 | **0.770** |
| safety | 40/40 | **40/40** |
| self-harm | 16/16 | **16/16** |

The 1.9.0 arm adds a frozen-base conversation KL anchor on 1,870 chat,
teacher and multi-turn rows to the unchanged KV4s mix. It buys +10.4 pp
both-turns and -0.065 retention; single-turn pays -5.0 pp. Self-harm gate
**promotable** (16/16). Copy exact 59.5 / key 70.0 sit within the 1.8.0 seed
range (54.5-60.0) and fresh unseen copy is higher (67.0).

| area | J (default) |
|---|---:|
| 2short / 3short / 4short | 86.0% / 75.5% / 48.5% exact |
| retention | 0.703 |
| benign (136) | 93.4% |
| safety (40) | 40/40 |
| self-harm (16) | 16/16 |
| RAG suite | 100% |
| multi-turn | 75% final / 50% both turns |
| gate / product / server | pass / pass / 20/21 |

## 1.8.0 result

| metric | J (1.7.0) | 1.8.0 promoted | 3-seed mean |
|---|---:|---:|---:|
| single-turn benign | 95.8% | **99.2%** | — |
| multi-turn final/both | 62.5 / 39.6 | **83.3 / 58.3** | 81.9 / 59.0 |
| 4-key copy (seen) | 48.5% | **60.0%** | 56.7% |
| 4-key copy (fresh) | — | **64.5%** | — |
| retention | 0.703 | 0.835 | 0.846 |
| safety | 40/40 | **40/40** | 2/3 seeds |
| self-harm | 16/16 | **16/16** | — |

Main gate **promotable** (exact 0.60, key 0.745, safety 40/40); self-harm gate
**promotable** (16/16). s72 was 39/40 on product safety; the promoted seed is
the fully gated one. Medium 1.7.0 was skipped; the next Medium post is 1.8.0
when the platform allows (no backfill).

## 1.7.0 result

| arm | single | multi final/both | 4-key | retention | safety | product |
|---|---:|---:|---:|---:|---:|---|
| J | 95.8% | 62.5 / 39.6 | 48.5% | 0.703 | 40/40 | ok |
| T 7B s71 | 95.8% | **79.2 / 47.9** | 38.5% | 0.868 | 40/40 | ok |
| TA 7B s71 | 97.5% | 77.1 / 45.8 | 38.5% | 0.844 | 40/40 | ok |

T and TA are rejected on the **10 pp copy regression**; the J anchor did not
recover copy. Retention is also worse, not better: 0.868 / 0.844 vs J seed 71's
0.703 (lower is better; J seed mean 0.840). The A anchor arm stays parked (s72
self-harm 14/16). No seeds 72/73 are trained for T/TA because the recipe cannot
close the copy gap.

## 1.6.0 result

| variant | benign | safety | self-harm | 4-key | retention | multi-turn | gate |
|---|---:|---:|---:|---:|---:|---|---|
| J | 93.4% | 40/40 | 16/16 | 48.5% | 0.703 | 75% / 50% | pass |
| K | **98.5%** | 39/40 | 14/16 | 48.5% | 0.808 | **93.8% / 75%** | fail |
| K2 | 89.0% | **40/40** | **16/16** | **53.5%** | 0.816 | 62.5% / 56.3% | pass |
| K3 | 95.6% | **40/40** | **16/16** | 45.0% | 0.828 | 68.8% / 37.5% | pass |

K3 passed every gate but lost copy, retention and multi-turn against J. No
variant beat J on the full profile, so the default did not change.

## Published surfaces

- PyPI: <https://pypi.org/project/atomzero/2.1.0/> (live)
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v2.1.0>
  (wheel + sdist assets attached)
- Hugging Face default: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
  (card/config 2.1.0, weights 1.2.0 verified)
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
  (card 2.1.0 verified)
- ModelScope Studio: 2.1.0, L-7781 verified
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-210-exact-budget-and-time-answers-still-63m-and-offline-18e8>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7532262926141779968/>
- X: <https://x.com/aigokhan/status/2126498628956983641>
- Medium: 2.1.0 article ready; the 24-hour limit is still active (latest post
  10 h old). Publish in the next window, no backfill.
- Reddit: **not used anymore**. Reddit filters remove posts; do not post or reply there. The existing profile/post stay as history only.
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo> (scoped document QA; synthetic corpus)

Verification: VM **1780 tests passed**; benchmark **20/20**; product suite
**pass**; server suite **21/21**; routed context **82.5%**; self-harm **16/16**;
`verify_release.py --version 2.1.0` reports **0 failures** (PyPI, HF,
ModelScope, GitHub release + assets, dev.to).

## Next levers

1. **Model-side arithmetic remains research; the runtime checker is the
   product path.** Do not ship a model arithmetic claim without a held-out
   rule test.
2. **Multi-turn routing is parked.** The larger 200-row open-ended eval shows
   the dialogue adapter at final 88.5% vs base 87.5%, but both-turns 73.0% vs
   77.5%. The gain does not justify routing; keep the 2.1.0 path.
3. **Teacher distillation:** retry only on a host with verified Hugging Face
   download speed (>=50 MB/s) or a preloaded Qwen image; otherwise use the
   synthetic labeled data.
4. Do **not** repeat 24L, cw3, MT-chain, self-teacher-replacement, soup, the
   opener arm or the full-safety-overrefusal variants; each is measured and
   rejected.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Vast.ai only for substantial runs; destroy the instance and check for
  leftovers afterward.
- Release every 0.1 boundary; new posts, not edits.
- Donation addresses stay unchanged unless the user asks.
- Default weights change only after the full product suite and human review.
- A metric-only pass is not a release decision; inspect coherence samples.
