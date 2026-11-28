# Next session - identity in the weights + spillover gates

Snapshot: 2026-11-28 09:40, **2.2.0 remains the default** (weights `1.2.0`).
No training is running; no Vast.ai instance is running. The runtime
`identity_tool` is implemented and tested locally (not released yet).

## User request: put name + creator in the weights

- Goal: the model itself answers "What is your name?" and "Who created you?"
  without `identity_tool`.
- Data: 150-300 paraphrases of name/creator questions with short answers
  ("I am Atom Zero." / "Atom Zero was created by Gökhan Çava."), plus a
  held-out 50-paraphrase identity eval. Check tokenizer handling of "Gökhan
  Çava" first; include ASCII "Gokhan Cava" if needed.
- Train: LoRA on the chosen base with 50% replay, 600-1000 steps; merge and
  gate.
- Gates: held-out identity >=95%, single >=97, copy >=60, safety 40/40,
  self-harm 16/16, 200-row multi both >=80, refusal <=7%.
- Keep `identity_router.py` as a runtime fallback/display layer; weights
  become the primary source.
- Do not ship the weights change without the full product/context/self-harm
  gates on the VM.

## Spillover adapter v2: 2.3.0 candidate

- `checkpoints/atom-zero-63m-2.3.0-spillover-v2`: `adapters_spillover_v2`
  merged into 2.0.0 (600 steps, 75% replay, lr 3e-5, rank 4).
- Held-out spillover: refusal **2.5%** (base 5.71%), useful **96.79%**.
- 200-row multi: final **91.0**, both **81.5**, turn **90.75**; refusals
  **24/400 (6.0%)** vs base 35; zero-overlap **33.75%** vs 34.75%; empties 2
  (vs 0), short 52 (vs 38).
- Product suite: **pass**; single **95.0%** (>=94), copy **62.0%** (key 71.5),
  safety **40/40**, self-harm **16/16**, content ratio **0.694** (2.0.0: 0.700),
  multi 91.67 / 68.75 on the 48-row set.
- Net: multi +3.5/+4.0 on 200 rows, refusals -2.75 pp, retention slightly
  better; single -2.5 (still >=94), copy -1.0 (within seed noise).
  **Promote as 2.3.0 candidate if the identity add-on and VM gates hold.**
- Next: train identity LoRA on top of this checkpoint
  (`data/identity_v1`, 200 train / 38 held-out), gate identity >=95 plus
  single/copy/safety/self-harm/multi; if it regresses, ship spillover-v2 with
  the runtime `identity_tool` and keep identity weights for a later pass.
  **Identity v1 failed:** 600 steps, lr 3e-5, rank 4 -> held-out 0%, train
  name 3.6%, project 0%; underfit (eval loss 3.10, train loss 2.55). The merge
  is fine; the adapter did not learn the facts. **v2 running:** 1,600 steps,
  lr 1e-4, rank 8/alpha 16, 25% replay, on the spillover-v2 checkpoint.
  **Identity v2 result: rejected.** Held-out identity **~92%** (name 14/14,
  creator 15/17, project 5/6) and direct answers are correct, but the adapter
  **over-triggers**: identity-category follow-ups such as "What is the main
  limitation?" and "Does it work offline?" get "I am Atom Zero." / "Atom Zero."
  (200-row zero-overlap 43.5% vs 33.75% for spillover-v2). The word-count
  multi metric rises to 94.5 / 85.0, but the replies are shallow. Keep the
  conservative runtime `identity_tool`; do not put identity in the weights
  with this recipe.
  Future identity-weights branch: contrastive rows (limitation/offline/
  capability questions with full answers) so the model does not answer every
  identity-adjacent question with its name.

## 2.3.0 candidate: spillover-v2 + runtime identity tool

- Weights: `checkpoints/atom-zero-63m-2.3.0-spillover-v2` (spillover adapter
  merged into 2.0.0; weights version -> 1.3.0).
- Code: runtime `identity_tool` (already committed), numeric/recall tools.
- Profile: single 95.0, copy 62.0, multi 200 **91.0 / 81.5**, refusals 6.0%,
  spillover refusal 2.5%, safety 40/40, self-harm 16/16, content 0.694,
  routed context 100% via tools.
- Next: VM tests, routed context check, stage/publish 2.3.0, new social posts;
  Medium should publish **2.3.0** when the limit opens (no 2.2.0 backfill).

# Previous session - 12L rejected, 2.2.0 stays

Snapshot: 2026-11-28 08:45, **2.2.0 remains the default** (weights `1.2.0`,
tools `numeric_tool` + `recall_tool`). No training is running; no Vast.ai
instance is running.

## 12L result: rejected

- `checkpoints/upcycle/atom-zero-63m-12l`: 12 layers, 98.0M params (research).
- `checkpoints/mixed12_j_s71`: 12L full fine-tune on `mixed_benign_j`,
  3,000 steps, lr 5e-5, batch 4 (91.7M params).
- 48-row product suite (`research/results/product_suite_12l_j.json`): pass,
  single **98.3%**, multi **85.4 / 75.0**, copy **60.0 / 66.0**, safety
  **40/40**, self-harm **16/16**, content ratio **0.844**.
- 200-row `multi_turn_v3` (`research/results/multi_v3_12l_j.json`): final
  **82.0%**, both **74.5%**, turn **85.25%** vs 2.0.0 **87.5 / 77.5 / 88.5**.
  Refusals: **56/400** turns vs base **35/400**. The 48-row both-turns gain was
  small-sample noise; the larger set regresses and over-refuses.
- Coherence audit: 12L **95.0% flagged** vs base **82.5%** on the same 200-row
  diagnostic; the audit is blunt (both high), so the refusal counts and the
  qualitative samples are the primary evidence. Samples include an identity
  question answered with a Python `def` explanation and a benign offline
  question answered with a security refusal.
- **12L KV4s+context** (`checkpoints/mixed12_kv4s_context_s71`): full FT on
  the 16,316-row mix, 3,000 steps. Copy exact **45.0%** -> rejected without
  further gates.
- Decision: **reject both 12L runs; keep 2.2.0**. Do not promote on the 48-row
  multi suite; 200 rows is the bar.

## 1.9.0 trade-off (no 2.3.0)

- 200-row multi: final **97.0%**, both **81.5%**, turn **90.75%** vs 2.0.0
  **87.5 / 77.5 / 88.5**; first-turn refusals **30** vs 19, zero-overlap
  **39.0%** vs 34.75%.
- Full gates: pass; single **94.17%**, copy **59.5%** (key 70.0), safety
  **40/40**, self-harm **16/16**, content ratio 0.770, routed context **100%**
  with tools.
- Trade-off vs 2.2.0: multi +10.4/+10.5 on 48 rows and +9.5/+4.0 on 200, but
  single -3.3, copy -3.5, first-turn refusals worse. Not a clear win ->
  **no 2.3.0**; 2.2.0 stays.
- Next branch: **spillover-contrast** adapter on 2.0.0 (or 1.9.0) targeting
  benign prompts that currently trigger self-harm/wrongdoing refusals, using
  held-out prompts and the new relevance audit. Target: first-turn refusals
  halved, single >= 97, copy >= 60, self-harm 16/16, safety 40/40.
  `data/spillover_v1`: 180 train / 280 eval, 12 families, no overlap with
  `multi_turn_v3`; base 2.0.0 useful **92.86%**, refusal **5.71%** (identity 6,
  coding 4, planning 4). Adapter `checkpoints/adapters_spillover_v1`: 800 steps,
  rank 4, alpha 8, lr 5e-5, 50% replay. Gate on spillover refusal, 200-row
  multi refusals/zero-overlap, single/copy/safety/self-harm.
  **Spillover eval after adapter: refusal 0.0% (0/280), useful 99.29%** on the
  held-out set. Full gates running on
  `checkpoints/atom-zero-63m-2.3.0-spillover`.
  **200-row multi:** final **87.0**, both **82.0**, turn **91.0**; refusals
  **10/400 (2.5%)** vs base 35; first-turn 10, second-turn 0. But zero-overlap
  **42.5%** (base 34.75%) and **23 empty second-turn replies** (new failure
  mode), plus 3 short. Product suite + self-harm were **not run** (stopped for
  shutdown); they are the next gate. Checkpoint is ready at
  `checkpoints/atom-zero-63m-2.3.0-spillover`.
  Before promotion: run product suite (single/copy/safety) + self-harm, inspect
  the 23 empty replies (stop/max_new artifact?), and require single >= 97,
  copy >= 60, safety 40/40, self-harm 16/16, both >= 80, zero-overlap <= 35%.
- Optional next model branch: 12L on `mixed_benign_k3_7b_kv4s_context`
  (16,316 rows, cache `data/pretok/mixed_benign_k3_7b_kv4s_context`), but it
  must explain the 200-row multi regression before any promotion.
  **Result: rejected.** `checkpoints/mixed12_kv4s_context_s71` finished with
  copy exact **45.0%** (2.0.0: 63.0), below the 55% bar; no further gates.
  Full fine-tuning the 12L base on the 16k mix disrupts copy more than the J
  mix (60.0). Keep 2.2.0.
- Coherence tool note: `diagnose_multiturn.py --out X.txt` overwrites the text
  with JSON; pass `--out X.json` so the dialogue text is `X.txt`.

## Medium 2.2.0

Unpublished: the 24-hour limit was active and the editor returned "cannot save
your story". Publish manually when the limit is open; do not backfill 2.0.0 or
2.1.0. Text: `C:\Users\biris\AppData\Local\Temp\opencode\medium_220.md`.

# Previous session - 2.2.0 released

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
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-220-exact-recall-for-facts-you-already-told-it-53l3>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7532286044948295680/>
- X: <https://x.com/aigokhan/status/2126520809539973139>
- Medium: 2.1.0 article ready; the 24-hour limit is still active (latest post
  10 h old). Publish in the next window, no backfill.
- Reddit: **not used anymore**. Reddit filters remove posts; do not post or reply there. The existing profile/post stay as history only.
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo> (scoped document QA; synthetic corpus)

Verification: VM **1803 tests passed**; benchmark **20/20**; product suite
**pass**; server suite **21/21**; routed context **100%** (200/200); natural
held-out **100%** (276/276); self-harm **16/16**;
`verify_release.py --version 2.2.0` reports **0 failures** (PyPI, HF,
ModelScope, GitHub release + assets, dev.to).

## Next levers

1. **Model-side arithmetic remains research; the runtime checker is the
   product path.** Do not ship a model arithmetic claim without a held-out
   rule test.
2. **Copy is tool-covered.** The deterministic copy tool scores 1.0 exact on
   the 200-row set; the model-only 63.0% is one seed against a 54.5-60.0
   three-seed range. Do not chase copy without a seed-controlled gain.
3. **Multi-turn routing is parked.** The larger 200-row open-ended eval shows
   the dialogue adapter at final 88.5% vs base 87.5%, but both-turns 73.0% vs
   77.5%. The gain does not justify routing; keep the 2.1.0 path.
4. **Next model-side candidate:** a 12L middle ground or a coherence audit of
   the 200-row multi-turn set, with full gates and seed control.
5. **Teacher distillation:** retry only on a host with verified Hugging Face
   download speed (>=50 MB/s) or a preloaded Qwen image; otherwise use the
   synthetic labeled data.
6. Do **not** repeat 24L, cw3, MT-chain, self-teacher-replacement, soup, the
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
