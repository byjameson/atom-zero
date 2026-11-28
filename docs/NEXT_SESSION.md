# Next session - 2.0.0 planning

Snapshot: 2026-11-28, **2.0.0 release prepared**. Head: release commit for
`2.0.0`; tag `v2.0.0`. No training is running; GPU idle. No Vast.ai instance
is running. Sponsor goal: **$10,000/month**.

## Released baseline

`2.0.0` is the current boundary. The default moves to the promoted **63M**
checkpoint (`byjameson/atom-zero-63m-balanced`, weights `1.2.0`); the 49M
`0.8.0` base is the fallback. The 1.9.0 63M weights stay available as the
previous checkpoint.

## 2.0.0 result

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

2.0.0 adds a history-conditioning LoRA to 1.9.0, plus a small safety adapter
with benign-support contrast. Main gate **promotable** (exact 63.0, key 73.0);
self-harm gate **promotable** (16/16); fresh 4-key copy **75.0%**. The
automatic multi-turn both-turns score falls 68.8 -> 58.3 because the safety
adapter is more cautious on ambiguous identity/greeting turns; the metric is
refusal/word-count only, not coherence. History recall is measured on 200
held-out counterfactual pairs.

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

- PyPI: <https://pypi.org/project/atomzero/2.0.0/>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v2.0.0>
- Hugging Face default: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
  (weights 2.0.0 / 1.2.0 verified; `verify_release.py` reports 0 failures)
- ModelScope Studio: 2.0.0, L-7781 verified
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-200-a-history-aware-context-adapter-1k0j>
- Medium: 2.0.0 post is scheduled for 2026-11-28 20:30 (the 24-hour limit is
  active now; no backfill).
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7532210386427703296/>
- X: <https://x.com/aigokhan/status/2126444828980371675>
- Reddit: **not used anymore**. Reddit filters remove posts; do not post or reply there. The existing profile/post stay as history only.
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo> (scoped document QA; synthetic corpus)

Verification: VM **1764 tests passed**; benchmark **20/20**.

## Next levers

1. **2.1.0 is not ready.** A 4-turn dialogue adapter raises held-out dialogue
   recall to **69.8%** and multi both to **81.25%**, but the self-harm gate
   fails 13/16. Adding a safety patch fixes self-harm and dialogue recall
   (86.3%) but drops single to 90.8%, copy to 55.5%, context to 40.5% and
   product safety to 39/40.
2. Next objective: a **joint context + dialogue + safety** adapter with a
   higher context weight and benign-support contrast; require single >= 94%,
   multi >= 60%, context >= 60%, copy >= 55%, safety 40/40, self-harm 16/16,
   and the coherence audit <= 30% flagged.
3. Generate real 4-6 turn teacher dialogues (not templates) through a custom
   dialogue job; the earlier single-turn validator rejects long prompts.
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
