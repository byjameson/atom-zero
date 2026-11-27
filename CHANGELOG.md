# Changelog

## atomzero 1.7.0 - 2026-11-27

- **A 7B teacher lifts conversation, not the full profile.** T (7B mix, no anchor) reaches **79.2% final-turn / 47.9% both-turn** multi-turn, safety **40/40**, but 4-key copy falls to **38.5%** (J 48.5%) and retention regresses to **0.868** (higher is worse; J seed mean 0.840). TA (7B mix + J anchor, beta 0.3) reaches **77.1 / 45.8**, copy **38.5%**, retention **0.844** (not a gain), safety **40/40**: the anchor does not recover copy, so TA is rejected.
- **No default promotion.** J / 63M balanced (weights `0.9.0`) stays the default; the 49M `0.8.0` base remains the fallback. The A anchor arm is parked after seed 72 failed self-harm **14/16**.
- 7B teacher pipeline: 260 accepted answers; fallback mix 8980 rows (518 7B + 2 base fallback), 854,500 pretokenized tokens. One substituted explanation prompt is recorded in the manifest.
- Scoped ModelScope Studio demo: **Atom Zero 63M — document QA demo**; out-of-corpus questions get a scope message before the model runs, while launch/warranty/travel/visitor questions stay cited.
- Reddit is no longer a release surface (its filters remove posts); the README, release checklist and release body were updated. PyPI metadata now carries the live demo URL.
- VM: **1764 tests passed in 57.97s**; benchmark **20/20** (pre-bump); the final run is repeated after the version bump.

## atomzero 1.6.0 - 2026-11-26

- **Teacher data improves conversation, not the full profile.** K (teacher 2x, safety 5x) reaches **98.5% benign / 93.8% final-turn multi-turn** but fails safety (39/40) and self-harm (14/16). K2 (safety/self-harm 8x) restores **40/40 / 16/16** and lifts copy to **53.5%**, but benign falls to 89.0%. K3 (6x midpoint) passes **40/40 / 16/16**, the checkpoint gate and the product gate, but copy 45.0%, retention 0.828 and multi-turn 68.8% / 37.5% are below J.
- **No default promotion.** The 63M balanced base (weights `0.9.0`) remains the default; the 49M `0.8.0` base remains the fallback.
- Teacher benign chat generator: 400 requested -> **260 accepted**, 437 rejected attempts, 0 duplicates. The expanded product suite adds local RAG cases and separate multi-turn final/both rates; **35 tests**.
- VM: **1727 tests passed in 64.83s**; benchmark **20/20**.

## atomzero 1.5.0 - 2026-11-26

- **First default promotion:** the 63M balanced base (weights `0.9.0`) becomes the default. The 49M `0.8.0` base remains the fallback.
- Variant J passes the full bar: **93.4% benign useful**, **40/40 safety**, **16/16 self-harm**, **48.5% 4-key exact**, retention **0.703**, checkpoint gate **pass**, product suite **pass**, server suite **20/21**.
- Data: benign chat replay expanded to **480 rows x3**, safety 5x, self-harm 5x, plus **96 contrastive boundary pairs** over a 6000 copy + 492 content core.
- **Gate fix:** `read_safety_rows` now infers `expect` from `kind`; the self-harm eval was scoring benign support rows as missing refusals. J's correct score is 16/16 (was 8/16). The 1.4.0 24L self-harm result is provisional until rerun.
- Packaging: default download `byjameson/atom-zero-63m-balanced` (~279 MB), cache `~/.cache/atomzero/atom-zero-63m-balanced`; fallback flags documented.
- VM: **1693 tests passed in 46.87s**; benchmark **20/20**.

## atomzero 1.4.0 - 2026-11-26

- **24L capacity run (176,795,136 params, F recipe):** 2short 95.5%, 3short 86.5%, 4short **77.5%** exact (81.0% key-copy), retention **0.790**, safety **40/40**, checkpoint gate **pass**, product suite **pass**, server suite **21/21**.
- **Capacity bought copy, not conversation:** benign useful stayed **83.8%** and the self-harm gate fell to **9/16**, while the 8L F model passes the same set **16/16**. Depth is a copy lever, not an alignment lever.
- **No default promotion.** The product default stays `0.8.0`. The 24L checkpoint download was interrupted and the instance destroyed; no 24L artifact is published.
- The Vast.ai job now runs the checkpoint gate, self-harm gate, 136-prompt product suite and server suite on the instance; shell scripts are packed with LF and process checks no longer self-match.
- VM: **1693 tests passed in 46.87s in 46.59s**; benchmark **20/20**.

## atomzero 1.3.0 - 2026-11-26

- **The 63M self-harm tradeoff:** contrastive self-harm replay fixes the crisis probe but any amount costs conversation. A: 95.6% benign / 39/40 safety. H (8 rows): 76.5% / 40/40. E (48 rows): 75.7% / 40/40. F (16 rows): **83.8% / 40/40**, 4-key 58.0%, content 0.841, gate and product pass.
- **New crisis filter category:** narrow patterns for "no point in living", "surest way to die", "I want to die", and "thinking about hurting myself" return a supportive refusal with crisis guidance. Tested against 82 safety cases, including benign prevention questions.
- F is published as `byjameson/atom-zero-63m-selfharm`, an optional base. The product default stays `0.8.0`; 83.8% is below the 90% promotion bar.
- The next lever is capacity (24L/177M or a wider benign corpus), not more safety rows at 63M.
- VM: **1693 tests passed in 46.87s in 46.58s**; benchmark **20/20**.

## atomzero 1.2.0 - 2026-11-26

- **First base to pass the strict gate and the expanded product suite.** Variant C (uniform safety 2x + benign chat 3x) reaches **40/40 safety**, **80.9% benign useful** on 136 prompts, **56.0% 4-key exact**, retention **0.785**, and a 20/21 server suite (RAG, copy, multi-turn).
- The 136-prompt suite (120 single-turn + 16 multi-turn) changed the ranking: A has the best conversation profile (95.6% benign, 61.5% 4-key) but fails one self-harm probe with unrelated coding advice; B (78.7% benign) and D (68.4%) pass safety but miss the product bar.
- **Decision:** C is published as `byjameson/atom-zero-63m-gated`, an optional base. The product default stays `0.8.0` because about one in five ordinary prompts is still refused; promotion requires benign useful >= 90% plus human review.
- New tooling: multi-turn benign suite, expanded product runner, server-level suite.
- Sponsor goal raised to **$10,000/month** for full-time development.
- VM: **1663 tests passed in 46.67s**; benchmark **20/20**.

## atomzero 1.1.0 - 2026-11-26

- **Benign chat replay fixes the refusal collapse.** The safety-ratio sweep failed (1x: 32% benign useful; 2x/10x: 4%), because the mix had no ordinary conversational examples. Adding 144 benign chat rows (x3) restored helpful behavior.
- **Two variants:** A (safety 1x) reaches **92% benign useful, 61.5% 4-key exact, retention 0.759**, but misses one of 40 safety probes. B (safety 5x) reaches **88% benign, 40/40 safety, 38.0% 4-key exact, retention 0.901**.
- **B is the first candidate to pass both the strict checkpoint gate and the product suite.** It is published as `byjameson/atom-zero-63m-benign`, an optional base. The product default stays `0.8.0` pending one more product-suite expansion and human review.
- New tools: 25-prompt benign suite, content-free product runner (`benign_useful_rate`, `safety_pass_rate`, `copy_exact`, `content_loss_ratio`, `product_ok`), 144-row benign chat replay builder.
- VM: **1631 tests passed in 47.30s**; benchmark **20/20**.

## atomzero 1.0.0 - 2026-11-26

- **Optional research bases**: `atomzero-quickstart --model-dir PATH` serves a local checkpoint with no download; `--model-repo owner/name` serves a published base. The default remains the 49.2M `0.8.0` base.
- **The 63M candidate failed the product suite.** Same server, same demo corpus, ten prompts: grounded document questions 3/3 on both, copy tool 1/1 on both, safety redteam 72/72 on both, but benign open prompts dropped to **0/5 useful with 4 refusals** on the candidate. RAG and copy answer before generation; open chat is broken.
- **Decision**: the 63M mixed-replay model stays an optional research base. A gate pass is necessary but not sufficient; the product default does not change.
- New `docs/OPTIONAL_BASES.md`, product-suite evidence in `research/results/product_suite_1.0_summary.json`, and quickstart tests for local bases and custom repositories.
- Carried from 0.90.0: retention eval, checkpoint gate, mixed replay, content rows, safety probes/replay, pretokenized cache, Vast.ai pipeline; 24L depth rejected (+4 points, retention 1.66).
- VM: **1612 tests passed in 51.02s**; benchmark **20/20**.

## atomzero 0.90.0 - 2026-11-26

- **Depth is not the lever.** A 24-layer / 176,795,136-param upcycle improved 4-key exact only to **69.5%** (two seeds) while content retention fell to **1.664** (fail). The RTX 3090 run cost about $0.42 and the instance was destroyed.
- **Mixed replay fixes retention.** The 8-layer 63M base fine-tuned on 6000 copy + 492 content + 40 safety rows kept four-key copy and improved content loss over the 49M product base.
- **Two seeds pass all four checkpoint gates**: 4short **57.0% / 58.0%** exact, **67.0% / 63.5%** key-copy, retention **0.722 / 0.809**, safety **40/40** on held-out probes (baseline 12/40). Safety replay costs about 9 points of four-key exact versus copy-only.
- New tools: `retention_eval`, `checkpoint_gate`, `mix_training_data`, `build_content_rows`, `build_safety_probes`, `build_safety_replay`, `pretokenize` plus `--cache-dir`, and the `scripts/vastai/` rent-run-destroy pipeline.
- The 63M candidate is published as a promotable research base; the product default stays `0.8.0`. Donation addresses unchanged.
- VM: **1609 tests passed in 50.45s**; benchmark **20/20**.

## atomzero 0.80.0 - 2026-11-25

- Added **depth upcycling** (`research/upcycle_model.py`): the 6-layer 49.2M checkpoint expands to 8 layers / **63,389,184 params** with interpolated block copies and research-only metadata.
- Full fine-tune of the 8L research base on the copy curriculum + verified teacher rows broke the 4-key wall on held-out data: **2short 89.5% exact / 91.5% key-copy, 3short 77.0% / 83.5%, 4short 65.5% / 74.0%** (200 rows each), versus 49M at 32/12/0 exact.
- Added a **new-task teacher generator** (`research/build_distill_new_tasks.py`): 1000 requested tasks produced **656 verified rows** (246 four-key), 344 rejected, 28 duplicates.
- Data-only control: training the 49M on those 656 new tasks moved 4short to **0% exact / 5% key-copy** and left 2-3 key performance flat. New data alone did not break the wall.
- The gain is bounded: a 30-row out-of-distribution key-style set (`field_`, `k_` prefixes) scored **0/30**. The 63M model copies within the trained key grammar, not generally.
- Product base remains `0.8.0`; the 63M checkpoint is a copy-specialized research artifact, not a promoted assistant.
- VM: **1521 tests passed in 47.52s**; benchmark **20/20**.

## atomzero 0.70.0 - 2026-11-25

- Added a **verified teacher distillation pipeline**: Qwen2.5-0.5B generates copy answers, a tolerant parser recovers the exact pairs, and only verified rows are canonicalized into 49M training data. A 300-prompt run accepted **153 rows** (json 66, kv 87, yaml 0; 4short 32) and rejected 147.
- The distilled student started from `multi_recover` and trained on teacher rows + 50% rehearsal. Mixed loss was essentially flat (`0.7769 -> 0.7683`), but held-out 2-3 key copy improved: 2short **32% exact / 50% key-copy** vs multi_recover 0/0; 3short **12% / 28%** vs 0/0; 4short remained 0%.
- This is a **partial 2-3 key distillation win**, not a general distillation win. The 4-key wall is unchanged; the runtime copy tool remains the exact 4-key path.
- 1493 tests passed on the VM in 50.08s; benchmark 20/20. Base remains 0.8.0.

## atomzero 0.60.0 - 2026-11-25

- Added a **six-criterion capability scorecard** with met/partial/not_met/unknown statuses, evidence paths and next targets. Current status: **met 1, partial 3, not_met 2**.
- Added **held-out task evidence**: arithmetic, city-YAML and city/river are all 0/30 exact for base and the best copy adapter. Runtime copy-tool accuracy is excluded as tool use.
- Added a **large-base capacity probe**: Qwen2.5-0.5B-Instruct reaches 65/85% exact on 2short, 55/65% on 3short and 55/85% on 4short with tolerant parsing, while the 49M adapter is 0-6% on 3-4 keys. Arithmetic/YAML zero-shot is 0/10. Decision: keep the from-scratch 49M 0.8.0 default; use larger models as teachers/research backends.
- Added a **controlled QKV vs MLP ablation** (3 seeds): qkv4 3.33% exact / 8.89% key copy; MLP 3.33% / 5.56%; verdict **inconclusive**, intervals overlap. QKV does not break the 3-4 key wall.
- 1463 tests passed on the VM in 47.20s; benchmark 20/20. Base remains 0.8.0.

## atomzero 0.50.0 - 2026-11-25

- **Copy tool router**: explicit transcription requests now route to `atomzero-copy` in the product chat/agent paths, with `source: "copy_tool"`. Non-copy prompts keep the previous model/RAG path. The router is conservative and returns 2-4 pair JSON/KV/YAML answers.
- **Weak-stage rehearsal loop**: `research/run_weak_stage_rehearsal.py` ranks stages, builds a weak-focused mix and retrains with before/after ledgers. Two real configs did not improve the aggregate: the gentle run moved multi exact 12% -> 15% but echo_json 76% -> 65%; the aggressive run degraded every stage. This is a tool, not yet a win.
- **Fused QKV attention LoRA**: `--qkv-rank`/`--qkv-alpha` adapt Q/K/V `in_proj_weight`, not just out_proj + MLP. The first from-base multi-short run reached only 6% exact / 18% key copy on the mixed set (3short 0/3, 4short 0/0); it needs a proper control and longer training before any conclusion.
- VM disk grown from 20 GB to 40 GB with `growpart` + `resize2fs`.
- 1399 tests passed on the VM in 44.36s; benchmark 20/20. Base remains 0.8.0.

## atomzero 0.40.0 - 2026-11-25

- **Real self-improve curriculum promotion**: a VM run trained `echo_kv,echo_json`, passed the 0.40 exact / 0.50 key-copy / content gate with **53% exact / 60% key copy**, and promoted automatically to `data/self_improve_curriculum_active.json`. A first attempt at a 0.40 exact threshold failed closed on content loss and did not promote.
- **Runtime copy tool**: `atomzero-copy` deterministically parses explicit key=value pairs and emits JSON/KV/YAML. Benchmark: **200/200 exact / 1.0 key-copy** on the real 4-key eval set. This is tool use, not a model capability.
- Curriculum runner `--init-adapter` starts the first stage from an existing adapter; later stages chain from the previous stage.
- 1337 tests passed on the VM in 48.82s; benchmark 20/20. Base remains 0.8.0.

## atomzero 0.30.0 - 2026-11-24

- **Best copy adapter so far**: specialize on multi, then rehearse on the balanced mix. `multi_recover` reached **48.0% exact / 60.4% key copy / 70.0% key recall** on the 250-row mixed held-out set, with content loss `4.6015 -> 0.7789`; behavior gate pass; curriculum registry updated.
- Added **key/value-aware token weighting** (`--key-weight`, `--value-weight`). It did not break the 3-4 key wall: mixed 40.4% / 53.2%, 3short 2/8, 4short 0/0.
- Added a **multi-key short-value curriculum** (2short, 2mixed, 3short, 4short). It helps 2-key rows (42/54 and 44/58) but 3-4 keys remain 0-6% exact.
- Added **self-improve curriculum integration** and VM evidence: `status: completed`, gate pass, exact 0.40, key-copy 0.548, fail-closed on broken commands.
- 1303 tests passed on the host; VM full-suite evidence follows the release prep.

## atomzero 0.22.0 - 2026-11-24 (intermediate patch; next boundary 0.30.0)

- Integrated the staged curriculum runner into `atomzero-self-improve`: `--adapter-curriculum-command` runs a bounded curriculum job, records the gate/promotion block, and fails closed on gate/threshold errors.
- The curriculum ledger now carries exact-match, key-copy, promotion and active-file evidence; raw prompt/answer text is never stored.
- Added `--adapter-curriculum-data-dir`, `--adapter-curriculum-final-eval`, `--adapter-curriculum-min-exact-match` and `--adapter-curriculum-min-key-copy-rate`.
- 1271 tests passed on the VM in 46.34s; benchmark 20/20.

## atomzero 0.21.0 - 2026-11-24

- Added a **curriculum gate/promotion loop**: the staged runner can train, evaluate a final mixed set, run the content/exact/key-copy gate and optionally promote to a separate curriculum registry.
- Added **adapter soup** (weighted LoRA A/B merging) with strict geometry validation and per-split evaluation.
- First behavior-passing curriculum adapter: **40.0% exact / 54.8% key-copy** on a 250-row mixed held-out set; content loss `4.0449 -> 1.0011`; gate pass; promoted to `data/curriculum_active_adapter.json` (product base remains `0.8.0`).
- Pure multi-key training improved multi exact from 8% to 12%, but echo_json fell from 72% to 10%; adapter soup was worse than rehab on every stage. Multi-key copying remains the bottleneck.
- 1264 tests passed on the VM in 50.35s; benchmark 20/20.


Release cadence is now **one 0.1 version per release**. This batch adds the
self-code branch and its planner/runner integration, plus the content/behavior
gate and the first copy-curriculum experiments.

- Added a **content-loss audit** and an **exact-match / key-copy behavior
audit**. The replay-weighted adapter passed the full-loss and content gates but
had 0/48 held-out exact matches, so the active registry was removed and base
`0.8.0` remains the only active model.
- Added **behavior-gated promotion**: `promote` now requires
`content_loss_improved`, `content_ppl_not_worse`, `exact_match_ok` and
`key_copy_ok`, and re-checks the exact-match threshold.
- Added **key-token and key-copy metrics** (`key_loss`, `value_loss`,
`syntax_loss`, `key_copy_rate`, `key_recall`) to `compare_adapters.py`.
- Added a **randomized-key copy curriculum** and a **staged copy curriculum**
(`echo_kv` -> `echo_json` -> `two_kv` -> `two_json` -> `multi`), with disjoint
train/eval keys and zero key/value overlap.
- The staged echo adapter reached **58% exact match / 68% key copy** on its
held-out echo stage; `two_kv` reached **29% / 41%**, but sequential training
forgot the earlier JSON skill (50% -> 3%). A focus rehearsal recovered
one-key JSON to **72% / 82%** and two-key JSON to **28% / 46%**, while keeping
echo_kv at **58% / 68%**. The multi stage remains hard at **8% / 18%**.
- Added **`--init-adapter`** for resumable/staged adapter training,
**gradient accumulation** (`--batch-size`, `--lr-scale`) and a staged
**curriculum runner** with a content-free ledger.
- Added a **full fine-tune copy control** for research only; it is never a
product checkpoint.
- Resized the test VM to **8 vCPU / 16 GB** (from 2 vCPU / 6 GB); the full
suite runs in about 41s there.


- Added **`atomzero-self-code`**: deterministic AST audit of Atom Zero's own
  source for `eval`/`exec` boundaries, `shell=True`, broad exceptions,
  wall-clock tests and TODO markers. Findings are bounded and stable.
- Added bounded proposals: allowed files, suggested checks and a risk level;
  high-risk arbitrary-code findings have no automated checks.
- `atomzero-improve --self-code-json` maps high/medium findings to
  `self_code_high` and `self_code_medium` planner signals.
- `atomzero-self-improve --self-code-root` can audit the repository and derive
  the packet's allowed files and checks from the top self-code proposal.
- Added `docs/SELF_CODE.md`, `docs/aisc/atomzero-self-code.toml` and a
  self-code check in `benchmarks/self_improvement_regression.py`.
- Sixth measured cycle (first self-code-fed cycle): the audit found a broad
  handler in `agent_runner.verify`, an external opt-in agent narrowed it to
  expected failures and added two regression tests; gate
  `cac82f022c879258`; scoreboard `855/0` -> `857/0`.
- Decision notes in `atomzero-self-improve` now bind the action id and gate
  id, so the content-free `note_id` is unique per gated decision.
- Added **`atomzero-self-code-diff`**: compares two self-code reports, pairs
  findings by kind/path/message so line shifts are not fake fixes, and reports
  resolved/introduced/persisting plus new high/medium counts.
- `atomzero-self-improve --self-code-root` now audits the repository before and
  the agent worktree after the run. If the diff introduces a new high or
  medium finding, the loop stops with `self_code_regression` and records no
  decision or scoreboard entry. Both audits use a high per-kind cap.
- Seventh measured cycle: targeted `aisc_coverage.py`, fixed a gate truncation
  false positive, gate `7f0d5be2bce36219`, scoreboard `1033/0 -> 1035/0`.
- The loop now feeds its existing decision ledger to the planner as bounded
  history, so repeated approvals/rejections adjust signal priorities.
- Added **`atomzero-escalate`**: deterministic local/teacher/agent/refuse
  routing with privacy, freshness and budget rules; a content-free growth
  ledger and summary. It never calls a network or starts an agent, and
  parameter growth remains explicitly zero/unchanged.
- Added `docs/GROWTH_LADDER.md` and `docs/ESCALATION.md` describing the honest
  snowball path: retrieval memory -> skills -> distilled adapters -> offline
  checkpoint upcycling -> agentic code improvement.
- Added **`atomzero-growth`**: one content-free report combining escalation
  growth, scoreboard measurements, teacher verification events and an optional
  self-code diff; it blocks on failed tests or a self-code regression and never
  changes weights.
- Added **`atomzero-verify-teacher`**: content-free claim/evidence/conflict
  verification for teacher answers. A verified result is a promotable event,
  not an automatic memory write, adapter or weight change.
- Added **autonomous self-improvement mode** (default): `--opt-in` enables the
  loop; the focused gate, self-code diff, full sandboxed test suite and
  post-merge suite replace per-change human approval. `--require-approval`
  restores the old gate. A failed post-merge suite rolls back.
- Added **`atomzero.sandbox`** and `--sandbox-info`: disposable worktree,
  sanitized environment, hard timeout and explicit `os_vm: false` on this
  machine (no Docker/VM backend is installed).
- Eighth measured cycle: fully autonomous, no per-change human approval;
  sandbox full suite 1066 passed, local merge, post-merge 1066 passed,
  cleanup; gate `da83c0302933d484`, scoreboard `1064/0 -> 1066/0`.
- Added **`atomzero-adapter-candidate`**: verified teacher traces ->
  deterministic train/held-out split -> candidate metadata + local train/eval
  files -> external evaluator -> held-out loss/perplexity gate ->
  content-free candidate event. It never trains, applies or publishes an
  adapter, and `promotable` requires a matching held-out count plus a loss
  improvement and no perplexity regression.
- `atomzero-improve` emits `adapter_candidate_ready` when promotable teacher
  notes exist and no candidate has been exported; `atomzero-self-improve`
  executes that action autonomously when adapter inputs are supplied.
- `atomzero-growth --adapter-jsonl` reports candidate exports, gate passes and
  promotable candidates.
- Added **real teacher queue fixture** and explicit cross-distribution
  candidate export. A local Qwen2.5-1.5B teacher produced 120 JSON and 34
  key-value verified answers; the fixture separates 94/23 in-distribution and
  117/34 cross-distribution splits. CSV was disabled because the 1.5B teacher
  did not reliably copy the requested row.
- Added **adapter task-family leakage protection**: candidate rows may carry a
  validated `family` label, `export-cross` fails hard when an eval family also
  appears in training, and candidate metadata reports `family_separation`,
  `train_families`, `eval_families` and content-free family counts. Random
  splits report `not_applicable`. See `docs/FAMILY_GATE.md`.
- Extended the diverse teacher fixture with reusable `FamilySpec` rows for
  `city_yaml`, `arith_plain` and `arith_json`, dependency-free JSON/key-value/
  YAML/plain validators, per-family CLI counts and a stub-friendly
  `build_rows`. `city_yaml` stays held out of train; tests need no GPU or
  teacher model. See `docs/TEACHER_FAMILIES.md`.
- Added the first **real teacher cross-distribution gate run** in the Ubuntu
  VM: in-distribution loss `6.5393 -> 1.2733` (pass); cross key-value loss
  `3.3621 -> 9.9921` (fail, ppl ratio `757.47`). The gate correctly blocked
  promotion. See `docs/ADAPTER_CROSS_CYCLE.md`.
- Added a **diverse teacher + replay** cycle: 194 verified multi-family teacher
  rows, 264-row training mix with JSON/key-value replay, in-distribution gate
  pass (`5.3949 -> 4.9404`) and cross-city gate fail (`5.2549 -> 7.0586`,
  ppl ratio `6.07`). Replay reduced the cross regression from 757x to 6.07x,
  but did not achieve cross-family transfer. See `docs/DIVERSE_REPLAY_CYCLE.md`.
- Added the first **replay-weighted cross-family gate pass**: 117 primary
  multi-family rows + 154 replay rows at weight `0.35`, 600 steps. Cross-city
  loss `5.2549 -> 2.8027`, ppl ratio `0.086`; the autonomous loop recorded
  `status: completed`, `promotable: true`, `weights_changed: false`. The
  held-out family was never in training. See `docs/DIVERSE_REPLAY_CYCLE.md`.
- Added **content-loss audit + gated adapter promotion**: `promote` now requires
  `content_loss_improved`/`content_ppl_not_worse`, candidate-id validation,
  `family_separation: true` and `adapter.pt`. The unweighted replay candidate
  passed the full-loss gate but its content loss was only `4.4327 -> 4.4123`
  (required `-0.05`), so promotion was blocked. A content-weighted retrain
  (`--content-weight 5`) moved content loss `4.4327 -> 3.4892` and was
  promoted as a local adapter; base remains `0.8.0`. See
  `docs/ADAPTER_PROMOTION.md` and `docs/DIVERSE_REPLAY_CYCLE.md`.
- Added **external adapter training harness** (`research/train_adapter_candidate.py`)
  and `--adapter-trainer-command`/`--base` wiring. A real LoRA run on the local
  0.8.0 base moved held-out loss from 5.9941 to 0.7949 and perplexity from
  401.04 to 2.214; the held-out gate passed with 60 examples.
  See `docs/ADAPTER_TRAINING.md`. The adapter is a local candidate; no product
  weights changed.
- Added **replay-aware adapter training**: `research/replay_sampler.py` mixes a
  deterministic, bounded fraction of older rows into
  `research/train_adapter_candidate.py` via `--replay-jsonl` and
  `--replay-weight` (default `0.25`). The adapter config and result record
  `replay_examples`, `replay_weight` and the actual `train_mix`. With no replay
  file the trainer keeps cycling primary rows. This is a forgetting mitigation,
  not a retention guarantee; the held-out gate and promotion boundary are
  unchanged. See `docs/REPLAY_TRAINING.md`.
- Added **prevalence-corrected isotonic calibration** for the answerability
  signal: hybrid Brier `0.2443 -> 0.2326`, ECE `0.1299 -> 0.0642`; feature-only
  Brier `0.2703 -> 0.2535`, ECE `0.1407 -> 0.0519`; AUC unchanged. The Brier
  target remains unmet and is not claimed.

## atomzero 0.19.0 - autonomous opt-in runner loop (2026-11-24)

This release follows the 0.3 cadence after 0.16.0 and adds the bounded
self-improve loop plus committed-change scope verification.

- Added **`atomzero-self-improve`**: live signals -> planner -> bridge packet
  -> opt-in runner -> gate -> approve/veto -> gated decision -> scoreboard ->
  trend. It requires both `--opt-in` and `--approve`; `--veto` stops before any
  worktree is created.
- The runner scope check now includes `git diff --name-only base...HEAD` plus
  working-tree changes, so an external agent cannot hide an out-of-scope
  committed file from verification.
- Fifth measured cycle executed an actual nested `opencode run` process in the
  isolated worktree; gate `7fb857e6552ae5cc`; approval `gate_ok: true`;
  scoreboard `844/0` -> `846/0`.

## atomzero 0.16.0 - opt-in runner and policy drift (2026-11-24)

This release follows the 0.3 cadence after 0.13.0 and adds the opt-in agent
runner and policy-metric drift alerts.

- Added **opt-in agent runner** (`atomzero-agent-run`): prepares an isolated
  worktree, optionally invokes an external command, checks scope against the
  packet's allowed files, runs the allowlisted gate inside the worktree, and
  reports; it never merges, pushes, tags or publishes.

## atomzero 0.13.0 - agent bridge and calibrated profiles (2026-11-24)

This release follows the 0.3 cadence after 0.10.0 and adds the bounded
agent-handoff bridge plus nested calibration-profile support.

- Added **agent handoff bridge** (`atomzero-agent-bridge`): a human-gated
  improvement action becomes a bounded task packet (allowed files, allowlisted
  checks, constraints) for an external agent harness. It does not spawn agents,
  edit source, run commands, merge, push or change weights.

## atomzero 0.10.0 - calibrated loop (2026-11-23)

This batch adds calibration-drift monitoring and closes two measured
improvement cycles with gated approvals.

- Added **calibration drift monitor** (`atomzero-drift`): compares a current
  answerability/calibration profile with a baseline, reports Brier/ECE deltas,
  raises `brier_increased`, `ece_increased` and `sample_size_dropped` alerts,
  and never runs a benchmark or changes weights. Calibration drift is also a
  prioritized signal in the improvement plan.

## atomzero 0.9.0 - measured self-improvement (2026-11-23)

This batch makes the human-gated loop measured: approval requires a passing
evaluation gate, actions carry suggested checks, and a content-free scoreboard
and trend report show whether the numbers moved.

- Weak-source and single-source actions now carry source-review next steps and
  focused reliability/consensus checks.
- Added **improvement trend report** (`atomzero-trend`): combines the
  content-free decision ledger and scoreboard, surfaces test/benchmark/AISC
  regressions as sorted alerts, and never copies output or text.
- Added a content-free **gate archive** (`atomzero-gate --archive`): gate id,
  counts, method and per-check status/duration only; output, commands and
  source text are rejected.
- The weekly review and its Markdown export include the scoreboard's latest
  measured run and delta.
- Added scoreboard HTTP routes: `POST /v1/scoreboard/record`,
  `/v1/scoreboard/summary` and `/v1/scoreboard/verify`. The record route stores
  counts and a note hash only.
- Added **self-improvement scoreboard** (`atomzero-scoreboard`): content-free
  run records (commit, test/benchmark/AISC counts, note hash), latest/previous
  deltas and a green flag for zero test failures. The writer verifies before an
  atomic rewrite; test output, prompts and source text are rejected.
- **Gated approval**: `record_gated_decision` rejects an `approve` decision
  unless the evaluation gate passed every check. The ledger stores only
  `gate_ok`, `gate_total`, `gate_passed` and `gate_method`; command output is
  not copied. `POST /v1/improvement/decision` now requires a `gate` report for
  approval. Reject/defer may carry a failed gate.
- Every improvement action now carries `suggested_checks` in
  `kind:target` form for `atomzero-gate --check`.

## atomzero 0.8.0 - human-gated self-improvement loop (2026-11-23)

This batch turns local weakness signals into a prioritized plan, requires human
approval, runs an allowlisted evaluation gate and records a content-free
decision. It does not update weights automatically.

- Added **improvement reminders** to proactive conversation: a pending,
  human-gated plan can start a local check-in with the actions `show
  improvements` and `run evaluation gate`; only the action count and top kind
  are stored.
- Added **improvement ledger summary** (`summarize_ledger`): content-free
  counts by decision and kind, with the latest timestamp.
- Added a **human-gated improvement loop** (`atomzero-improve`,
  `POST /v1/improvement/plan`, `POST /v1/improvement/decision`): local signals
  become prioritized actions with `requires_human: True` and an acceptance
  gate; decisions are content-free (action id, kind, decision, note hash).
  There is no auto-apply and no weight update.
- Added **Evaluation gate** (`atomzero-gate`): allowlisted pytest, AISC and
  manual checks for proposed improvements. No generic shell, timeouts fail,
  manual checks are not auto-passed, and `auto_apply`/`weights_changed` are
  always false.

## atomzero 0.7.0 - evidence and temporal loop (2026-11-23)

This is the ten-increment batch accumulated after `0.6.1`: claim and consensus
audits over HTTP, freshness, reliability, temporal validity, claim-aware
proactive check-ins, document version diff, aggregate audit summary and AISC
coverage. No model weights changed; the weights remain `0.8.0`.

- Added **Cross-source consensus** (`atomzero-consensus`,
  `POST /v1/consensus`): deterministic claim clustering across retrieval hits,
  consensus/single-source status, and number/date/identifier/negation conflicts
  kept separate instead of averaged. It makes no model calls and does not
  persist the request.
- Added **AISC coverage** (`atomzero-coverage`, `atomzero-audit --coverage`):
  per-component verification, evidence-file existence and calibration counts
  across the specs. A passing check is not treated as complete coverage.
- Added **claim-aware proactive conversation**: claim and consensus audits can
  record content-free counts on a local session, and proactive check-ins can
  start from unsupported claims, conflicts or single-source clusters. Query,
  claim text, chunks and source labels are not stored.
- Added **Temporal validity / as-of filtering** (`atomzero-temporal`,
  `POST /v1/temporal/filter`): explicit `valid_from`/`valid_until` windows,
  valid/expired/not_yet/undated status and an included/excluded evidence
  report. Undated evidence is configurable and no input is persisted.
- Added **Source reliability ledger** (`atomzero-reliability`,
  `POST /v1/reliability/summary`, `POST /v1/reliability/verify`): hash-only
  observation counts, a support ratio and verification that rejects text,
  query, answer, chunk and session content. The writer verifies the whole
  ledger before an atomic rewrite.
- Added **Source freshness** (`atomzero-freshness`, `POST /v1/freshness`):
  deterministic date extraction from hit text with fresh/aging/stale/future/
  undated status, age days and oldest/newest evidence. It does not filter
  retrieval or use file mtime.
- Added **Document version diff** (`atomzero-doc-diff`,
  `POST /v1/documents/diff`): deterministic block-level added/removed/changed
  blocks and a similarity ratio, with labels and no persistence.
- Added `POST /v1/claims/audit` for the local claim/evidence audit. `query` is
  required, `hits` is a bounded list (maximum 50) and only source/title/
  section/page/chunk/snippet fields are passed through.
- Added `atomzero-audit --summary` for a single aggregate pass/fail/error line
  across all specs, with an optional JSON summary.
- Documented AISC 0.5 `batch` and `diff` support, including glob/directory
  expansion and `--fail-on-breaking`, in `docs/AISC_AUDIT.md`.

## atomzero 0.6.1 - claim evidence and generated AISC specs (2026-11-23)

- Added a content-free **memory lifecycle ledger** (`atomzero-memory-ledger`,
  `GET /v1/memory/ledger`, `POST /v1/memory/ledger/verify`). Successful
  document/session deletes append counts/scope only; the ledger validator
  rejects text, query, answer, chunk and session content fields. The AISC
  memory spec now includes `LedgerWriter` after `MemoryStore`.
- Added deterministic **Claim / Evidence Algebra** (`atomzero-claims`):
  sentence-level claim extraction, best-hit evidence linking, supported/weak/
  unsupported status and a transparent number/unit/date/code/negation
  contradiction audit. It is a lexical audit signal, not an NLI model; it makes
  no model calls and does not change retrievals or weights.
- Added `atomzero-aisc-spec`: builds an AISC TOML spec from the measured
  answerability profile, validates the required evidence keys, and supports
  `--check` drift detection. It never imports AISC and never lowers the target
  thresholds; the generated spec preserves the measured failures so strict
  AISC still reports them.
- Added `docs/CLAIMS.md`, `docs/AISC_SPEC_GENERATOR.md`, the
  `AtomZeroClaims` AISC spec and a generated
  `docs/aisc/atomzero-answerability-generated.toml`.
- Added `docs/ORCHESTRATION.md` for the Planner / Implementer / Reviewer /
  Release workflow and the batch-push policy.

## atomzero 0.6.0 - weekly review and task workflow (2026-11-23)

- Answer snippets now score every contiguous window of up to four sentences
  and keep the best query coverage plus a small length bonus (capped at 500
  chars). Public SQuAD v2 visible-snippet accuracy improved from 0.816 to 0.884
  (hybrid) and 0.777 to 0.845 (FTS) on 50 contexts / 200 questions.
- Source comparison now normalizes grouped numbers, magnitude words, decimal
  commas, dates and units before flagging conflicts (`$1,500,000` =
  `1.5 million`, `1,5` = `1.5`, `5 January 2020` = `January 5, 2020`,
  `24 months` = `2 years`, `72 hours` = `3 days`), reducing false alarms.
  Topic-aware alignment also compares claims with a shared title/section and a
  comparable measure, date or code even when the wording differs; conflicts
  record an `alignment` field. Contradiction regression is now 16/16.
- Added Turkish morphology-aware retrieval: light suffix/consonant-mutation
  skeletons, vowel-drop stems (`şehir` / `şehrin`, `burun` / `burnu`) and FTS
  prefix variants. An authored 20-query check improved hybrid top-1 from 0.800
  to 1.000 and FTS from 0.700 to 0.950; English SQuAD results are unchanged.
- Added a deterministic **document timeline**: `atomzero-docs timeline` and
  `GET /v1/documents/timeline` extract full dates and year-only mentions with
  source/title/page/section provenance. Chat (`timeline`, `zaman çizelgesi`),
  the UI Show timeline action and entity filters share the route. The timeline
  regression passes 10/10.
- Added deterministic **typed relation extraction**: `atomzero-docs relations`,
  `atomzero-docs relation-path`, `GET /v1/graph/relations`,
  `GET /v1/graph/relations/path`, chat (`relations for X`, `X ilişkileri`,
  `relation path from X to Y`) and a UI Show relations action return explicit
  `located_in`, `capital_of`, `founded_by`, `ceo_of` and `born_in` triples and
  multi-hop paths with locators; English and Turkish patterns are supported.
  The relations regression passes 12/12.
- Added a calibrated **CitationGate** for local answers: rerank score ->
  validity probability, configurable threshold/profile via
  `ATOMZERO_CITATION_*`, citation dropping and explicit refusal on an empty
  gate. Default is `squad_v2_stress` at 0.5, not the authored 0.6. SQuAD v2
  simulation: 92/103 answerable retained, 18/97 no-answer refused, precision
  0.479.
- Added a local isotonic calibration utility: `python -m atomzero.calibrate`
  reads a score/label CSV, fits a PAVA map, reports Brier/ECE and recommends a
  threshold for `ATOMZERO_CITATION_CALIBRATION`.
- Added local adapter validation: `research/adapter_check.py` and
  `research/compare_adapters.py` load an adapter produced by the external
  `streamft` / `continual-lora` trainer, check the base/LoRA shapes and score an
  assistant-turn holdout without importing the trainer. The accepted adapter
  moved the two-row holdout loss from 3.853 to 3.139 (-18.5%). A format queue
  already known to the base was a negative control: both tested adapters
  regressed the holdout and were rejected.
- Added `benchmarks/answerability_eval.py` and recorded the initial negative
  result: context-grouped 5-fold CV AUC was 0.562 (logistic) / 0.497 (boost).
- Added an independent optional answerability gate
  (`src/atomzero/answerability.py`) with 42 retrieval/answer-type features,
  optional shadow/enforce env modes and CitationGate kept as a separate gate.
  Default is off; CitationGate remains at 0.5.
- Added `benchmarks/answerability_signal.py`,
  `benchmarks/answerability_transformer.py` and
  `benchmarks/answerability_hybrid.py`. The best hybrid (tiny local QA
  confidence + retrieval features) reached held-out AUC 0.663, Brier 0.244,
  ECE 0.130; it still misses the 0.90 refusal recall/precision and 0.80
  coverage targets, so it is not enabled by default. See
  [`docs/ANSWERABILITY.md`](docs/ANSWERABILITY.md).
- Added `benchmarks/graph_path_regression.py` (6/6 checks).
- Added a packaged `atomzero-audit` command with strict warning handling and
  `--require-calibration`, plus `--json`, `--explain`-ready findings and the
  `docs/aisc/` specs for read, task, tool-plan, context-budget and
  answerability paths.
- Added the **tool plan algebra**: `atomzero-plan` / `POST /v1/tools/audit`
  reports side effects, irreversible steps, confirmation requirements and risk
  before execution (`atomzero.plan_audit`).
- Added the **context budget algebra**: `plan_hits`,
  `grounded_answer(max_context_chars=...)` and
  `atomzero-docs answer --max-context-chars` bound retrieved evidence and
  report dropped hits (`atomzero.context_budget`).
- Added **proactive conversation**: an opt-in local scheduler that starts a
  short, useful chat from overdue/due tasks, knowledge gaps or recent sources,
  with cooldowns, quiet hours, a daily cap and a UI toggle
  (`src/atomzero/proactive.py`, `POST /v1/proactive*`).
- Added the **memory lifecycle** first slice: document and session deletion now
  require an explicit server-side `confirm: true` (409 otherwise, 400 for a
  non-boolean), closing the direct-HTTP bypass. The AISC
  `AtomZeroMemoryLifecycle` spec puts `HumanConfirmation` before the
  irreversible `MemoryStore`.
- Added optional AISC audit integration to `atomzero-doctor --aisc` and the
  `AtomZeroProactive` / `AtomZeroContextBudget` / `AtomZeroMemoryLifecycle`
  specs. Strict calibration warnings remain visible and are not hidden.
- Added a **weekly review**: open/recently completed tasks, pending gaps,
  sources indexed within the window and graph highlights.
- Added `atomzero-brief --weekly --days N`, `POST /v1/weekly-review` and a UI
  Weekly review button.
- Added task `created`/`completed` timestamps with an undated-completion
  fallback for older sessions.
- Added `benchmarks/weekly_review_regression.py` (6/6 checks).
- Added chat intents for the weekly review (`weekly review`,
  `haftalık özet`, optional day window) with the full review payload.
- Weekly review text now says “completed in the last N days” for non-default
  windows instead of always “this week”.
- Weekly review now reports active tag counts (open tasks plus completions in
  the window); Markdown exports include a Tags table.
- Added weekly-review tag filters: CLI `--weekly-tag`, HTTP `tag`, chat
  `weekly review tag work` and a UI tag field. Task sections, counts, Markdown
  and archive metadata are scoped; archive listings show `tag=work`.
- Added daily-brief tag filters: CLI `--brief-tag`, GET/POST `/v1/briefing`
  `tag`, chat `brief me tag work` and a UI field. Tasks, tag counts and next
  actions are scoped; gaps, index and graph stay global.
- Added tag filters to overdue and due-soon views: CLI `--overdue-tag` /
  `--due-soon-tag`, HTTP `tag`, chat `overdue tasks tag work` /
  `due soon tag work` and a shared UI tag field, with scoped counts and
  `| tag=work` headers.
- Added exact tag filtering to the weekly report archive: CLI
  `--reports --tag work`, HTTP `?tag=work` and a UI tag field; listings show
  the stored report tag.
- Added report-tag filtering to chat archive commands: `find report tag work`,
  `list reports tag work`, combined query/sort/limit, with `tag` in the tool
  call and reply.
- Added `benchmarks/weekly_chat_regression.py` (6/6 checks).
- Added weekly-review **Markdown export**:
  `atomzero-brief --weekly --markdown`, `GET /v1/weekly-review/export` and a UI
  Download .md button, with task checkboxes, gaps, sources, graph and next
  actions.
- Added `atomzero-brief --weekly --markdown --out PATH` to write the report
  directly to a file and create parent directories.
- `--out` now writes via a sibling temp file and atomic replace, preserving an
  existing report when the write or replace fails.
- Added a local **weekly report archive**: each successful `--out` export is
  indexed with its period, task/source/graph counts, size and SHA-256.
- Added `atomzero-brief --reports [--json]`, `GET /v1/weekly-review/reports`,
  indexed report download and a UI **Recent reports** panel.
- Added `atomzero-brief --reports --prune`,
  `POST /v1/weekly-review/reports/prune` and a UI **Prune missing** action to
  remove index entries whose report files no longer exist.
- Added `atomzero-brief --reports --keep N`,
  `POST /v1/weekly-review/reports/keep` and a UI **Keep newest** control for
  atomic retention of the newest N reports.
- Added `--keep N --dry-run`, `dry_run: true` for the keep endpoint and a UI
  **Preview only** option that lists would-be-removed reports without changing
  the archive.
- Added `atomzero-brief --reports --delete ID`,
  `POST /v1/weekly-review/reports/delete` and a UI **Delete** action for one
  archive entry; the report file is left on disk.
- Added `atomzero-brief --reports --query TEXT`,
  `GET /v1/weekly-review/reports?query=...` and a UI report search box.
- Added chat search for archived reports (`find report weekly`,
  `rapor ara weekly`) with a `search_weekly_reports` tool call, report payload
  and a UI quick prompt.
- Added chat maintenance previews for `prune report archive`,
  `keep reports N`, `keep reports N dry-run` and `delete report ID`; destructive
  actions require `confirm reports`, cancellation uses `cancel reports`.
- Added queryless chat archive listing (`list reports`,
  `show report archive 5`) with an optional recent-entry limit.
- Added archive ordering: `--reports --sort newest|oldest|largest`,
  `GET /v1/weekly-review/reports?sort=...` and a UI sort selector.
- Added chat sort suffixes for archive list/search commands, such as
  `list reports oldest` and `find report weekly largest`.
- Added chat report viewing by id (`show report w0001`, `rapor göster w0001`)
  with metadata, Markdown content and a `show_weekly_report` tool call.
- Added archived-report integrity checks: `--reports --verify`,
  `GET /v1/weekly-review/reports/verify` and a UI **Verify files** action
  comparing stored size and SHA-256 with each file.
- Added chat integrity commands (`verify report archive`,
  `verify report w0001`) and a UI quick prompt.
- Added archived-report metadata repair: `--reports --repair`,
  `POST /v1/weekly-review/reports/repair` and a UI **Repair metadata** action
  with atomic index updates for changed files.
- Added confirmed chat repair (`repair report archive`, `rapor arşivini onar`)
  with a preview and `confirm reports` gate.
- Added single-entry chat repair (`repair report w0001`, `rapor w0001 onar`)
  with the same confirmation flow.
- Added report comparison: `--reports --compare LEFT RIGHT`,
  `GET /v1/weekly-review/reports/compare` and
  `compare reports w0001 w0002` showing task, gap, source, size and SHA-256
  deltas.
- Added comparison Markdown export: `--compare ... --markdown`,
  `GET /v1/weekly-review/reports/compare/export` and UI Download/Copy actions.
- Added task due dates (`due YYYY-MM-DD`, `today`, `tomorrow`, `in N days`)
  with overdue markers in `list tasks`, the daily brief, weekly review and
  Markdown export.
- Added an overdue-task view: `atomzero-brief --overdue`,
  `GET /v1/tasks/overdue` and a UI **Overdue tasks** button.
- Added a due-soon view: `atomzero-brief --due-soon --due-days N`,
  `GET /v1/tasks/due-soon?days=N` and a UI **Due soon** button.
- Integrated due-soon tasks into the daily brief, weekly review, weekly
  Markdown report and chat (`due soon`, `due soon in 14 days`).
- Added task rescheduling (`reschedule task ID due DATE`),
  `POST /v1/tasks/reschedule` and a UI reschedule form.
- Added confirmed task deletion (`delete task ID` → `confirm task delete`),
  `POST /v1/tasks/delete` with `"confirm": true`, and a UI delete form.
- Added task text search (`search tasks TEXT`, `find task TEXT` or
  `task search TEXT`), `GET /v1/tasks/search?q=TEXT`,
  `atomzero-brief --search-tasks TEXT` and a UI search field/button.
- Added task-search status/due filters: chat suffixes (`open`, `done`,
  `overdue`, `today`, `week`, `none`), HTTP `status`/`due` query parameters,
  `--search-status`/`--search-due` CLI flags and UI selectors.
- Added task priorities (`low`, `normal`, `high`), `priority high` on
  `add task`, `set task ID priority LEVEL`, priority-aware list/brief/review
  labels, `priority` search filters in chat/CLI/HTTP and a UI selector.
- Added priority updates outside chat: `--set-task-priority ID LEVEL`,
  `POST /v1/tasks/priority` and a UI task ID/priority form.
- Added priority-first sorting for task search, overdue and due-soon views:
  `--task-sort priority`, HTTP `sort=priority` and a shared UI task-order
  selector.
- Applied priority-first sorting to daily brief and weekly review task lists,
  HTTP briefing/review JSON, Markdown export and the UI sort selector.
- Added chat commands `brief me priority`, `daily brief priority` and
  `weekly review priority` (including `weekly review 14 days priority`), with
  `sort` exposed in tool calls.
- Added priority sorting to chat task views: `search tasks ... sort priority`,
  new `overdue tasks [priority]` commands and `due soon priority`. Tool calls
  expose the selected sort.
- Added `list tasks priority` (and `sort priority` / `sorted by priority`)
  using the same priority, due-date and id ordering as the other task views.
- Added status/due/priority filters to chat `list tasks` (`done`, `all`,
  `due week|overdue|today|none`, `priority low|normal|high`), including
  combined filters and query-free views.
- Added a structured `list_tasks` chat tool call, `GET /v1/tasks/list` and UI
  **Task list** controls backed by the same shared filter/sort view.
- Added filtered task-list CSV/Markdown export:
  `atomzero-brief --task-list --task-format csv|markdown`,
  `GET /v1/tasks/export` and UI Download CSV/.md buttons using the current
  task-list filters.
- Added task tags: `tags work,finance` on `add task`, `tag task ID ...`,
  `list tasks tag work` / `#work`, CLI `--task-tag`, HTTP/UI `tag` filters and
  `tags` columns in CSV/Markdown exports.
- Added tag filters to task text search: chat `search tasks ... tag work` /
  `#work`, CLI `--search-tag`, HTTP `tag` and a UI search tag field.
- Added a task-tag inventory: chat `list tags` / `show tags`, CLI
  `--list-tags`, HTTP `/v1/tasks/tags` and clickable UI tag chips with
  open/done counts.
- Added confirmed global tag rename: chat `rename tag OLD NEW` then
  `confirm tag rename`, CLI `--rename-tag OLD NEW --confirm-rename-tag`,
  HTTP `/v1/tasks/tags/rename` and UI old/new fields with a confirmation
  checkbox.
- Added confirmed global tag delete: chat `delete tag TAG` then
  `confirm tag delete`, CLI `--delete-tag TAG --confirm-delete-tag`, HTTP
  `/v1/tasks/tags/delete` and UI tag field with a confirmation checkbox.
- Added undo for the last confirmed tag delete: chat `undo delete tag`, CLI
  `--undo-delete-tag`, HTTP `/v1/tasks/tags/undo-delete` and a UI undo button,
  restoring from a per-task session snapshot.
- Added a local tag audit trail (last 20 delete/undo events): chat `tag audit`,
  CLI `--tag-audit`, HTTP `/v1/tasks/tags/audit`, a UI panel and a `## Tag
  audit` section in weekly Markdown/archive reports.
- Ported confirmed tag delete, undo and tag audit to the legacy
  `research/chat_human.py` runtime.
- Daily brief now includes recent tag audit events (last three in text,
  `tag_audit`/`tag_audit_total` in JSON, scoped by `--brief-tag`).
- Added tag and action filters to the tag audit view: chat `tag audit work` /
  `action delete`, CLI `--tag-audit-tag` / `--tag-audit-action`, HTTP
  `tag`/`action` and UI fields.
- Added filtered tag-audit CSV/Markdown export through CLI
  `--tag-audit-format` and `/v1/tasks/tags/audit/export`.
- Added UI CSV/Markdown download buttons for the filtered tag audit.
- Weekly review Tag audit section is scoped by `--weekly-tag` and the new
  `--weekly-audit-action`; HTTP `audit_action` and a UI selector mirror it.
- Ported tag/action audit filters to the legacy `research/chat_human.py`
  runtime.
- Ported task tags to the legacy `research/chat_human.py` runtime:
  `add task ... tags ...`, `tag task`, tag-filtered `list tasks`, `list tags`
  and confirmed `rename tag` with persistence.
- Added optional AISC architecture specs for the critical read and
  task-action paths, plus `scripts/aisc_audit.py` for external AISC checks.
- Fixed `add task` priority parsing so task text ending in "priority task" is
  not interpreted as a second priority clause.
- Added `benchmarks/weekly_export_regression.py` (29/29 checks: CLI, HTTP, UI,
  the report archive, pruning, retention, dry-run preview, deletion, query
  filtering, chat search/list/maintenance, chat/HTTP sorting, report viewing,
  integrity verification, all/single-entry metadata repair, comparison and
  comparison Markdown export).
- Fixed BOM-sensitive session loading: `atomzero-brief` and `ChatHuman` now
  accept PowerShell-written UTF-8 session files with a BOM.
- Extended BOM-tolerant reads to feedback, knowledge-gap, research-queue,
  skill and recipe JSON files.
- Added deterministic **source comparison**: `atomzero-docs compare`,
  `GET /v1/documents/compare` and UI Compare sources, flagging numeric,
  identifier and negation conflicts with both snippets/locators.
- Added `benchmarks/contradiction_regression.py` (5/5 checks).
- Added a chat intent for source comparison
  (`compare sources A and B`, Turkish equivalent) with the same reasons,
  snippets and locators, plus `benchmarks/compare_chat_regression.py`
  (6/6 checks).
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.6.0.md`](docs/RELEASE_ATOMZERO_0.6.0.md).

## atomzero 0.5.0 - second brain and skill workflows (2026-11-22)

First batched release under the ~10-increment policy. It includes the work
accumulated after 0.4.2.

- Added a local **second brain graph**: capitalized-phrase/code/year entity
  extraction, chunk co-occurrence edges, source counts and evidence lookup.
- Added `atomzero-docs graph` / `entity`, `GET /v1/graph` and
  `GET /v1/graph/entity`, plus a UI graph panel with entity chips, edges and
  source/snippet cards.
- Chat now answers relation queries such as “what is related to Ankara?” with
  graph evidence and connections.
- Added graph regression tests and `docs/KNOWLEDGE_GRAPH.md`.
- Added declarative **custom skills**: JSON trigger phrases, local-context
  prompts, provider workflows and deterministic templates.
- Added `atomzero-skills list/validate/show`, `GET /v1/skills`, a UI skills
  panel, three example skills and `docs/SKILLS.md`.
- Added skill regression tests for loading, matching, context rendering,
  provider/template modes and the CLI.
- Added a deterministic **daily brief** (`atomzero-brief`, `POST /v1/briefing`)
  combining open tasks, pending gaps, index stats and graph entities.
- Added a cited **goal planner** (`POST /v1/goals/plan`) with suggested tasks;
  task creation requires explicit `confirm: true`.
- Added the Daily brief UI panel and `docs/BRIEFING.md`.
- Added chat goal planning: `plan goal ...`, `confirm plan` and `cancel plan`.
  Task creation requires the explicit confirmation message.
- Added chat `brief me` / `daily brief`, returning open tasks, pending gaps,
  index status and graph entities with source items.
- Added `benchmarks/briefing_regression.py` (9/9 checks) and its test.
- Added `benchmarks/goal_flow_regression.py` (6/6 checks), an end-to-end HTTP
  test for plan/no-task, exactly-once confirmation and cancel/no-mutation.
- Added `benchmarks/brief_me_flow_regression.py` (7/7 checks), an end-to-end
  HTTP chat test for tasks, gaps, index, graph and briefing sources.
- Added `benchmarks/skills_flow_regression.py` (9/9 checks), an end-to-end
  HTTP chat test for provider, template and no-provider skill modes.
- Added explicit skill execution by name/query: `atomzero-skills run`,
  `POST /v1/skills/run` and UI Run skill controls, sharing `execute_skill`.
- Added `benchmarks/skills_run_regression.py` (9/9 checks) for explicit
  provider, template, fallback and unknown-skill paths.
- Added multi-skill chains: `atomzero-skills chain`, `POST /v1/skills/chain`
  and a UI Run chain control, with explicit `{previous}` context passing,
  no-carry mode and merged sources.
- Added `benchmarks/skills_chain_regression.py` (6/6 checks).
- Added `benchmarks/skills_chain_ui_regression.py` (6/6 checks): UI contract +
  the exact chain request through the real handler.
- Added named chain recipes under `data/skills/chains`: `atomzero-skills
  recipes` / `recipe`, `GET /v1/skills/chains`,
  `POST /v1/skills/chains/run` and a UI recipe select/run control.
- Added `benchmarks/skills_recipe_regression.py` (6/6 checks).
- Added `benchmarks/skills_recipe_ui_regression.py` (7/7 checks): recipe UI
  contract + exact request, carry override and unknown-recipe path.
- 286 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.5.0.md`](docs/RELEASE_ATOMZERO_0.5.0.md).

## atomzero 0.4.2 - UI collection scope (2026-11-22)

- Added a collection-prefix input and **Summarize collection** button to the
  document UI. Search and Answer accept the same prefix scope.
- Exact source and prefix scopes are mutually exclusive in the UI.
- Added a UI regression test for the collection-scope contract; 249 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.4.2.md`](docs/RELEASE_ATOMZERO_0.4.2.md).

## atomzero 0.4.1 - scoped regression and code-aware retrieval (2026-11-22)

- Added `benchmarks/scoped_regression.py`, a deterministic regression gate for
  source/prefix/kind-scoped answers, summaries, locators, no-answer behavior
  and exact code identifiers. Latest result: 10/10 checks pass.
- Added exact code/ID awareness to FTS candidates, reranking and snippets:
  `L-7781` and `W-7781` now rank their matching source instead of competing on
  the shared numeric token.
- Added tests for the benchmark and identifier-aware search; 248 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.4.1.md`](docs/RELEASE_ATOMZERO_0.4.1.md).

## atomzero 0.4.0 - hungry mode and install diagnostics (2026-11-22)

- Added **hungry mode**: `atomzero-hungry`, `POST /v1/hungry/run` and a UI
  section. It generates bounded questions, checks local knowledge, asks the
  configured teacher only for unknowns, stores unverified notes and writes an
  audit ledger. It never fetches web pages and never changes weights.
- Added `atomzero-quickstart --teacher`, gap states `known`/`learned` and a
  dry-run/provider-call budget for hungry mode.
- Added `atomzero-docs summary-collection` and collection mode on
  `GET /v1/documents/summary`; every selected sentence keeps its source,
  page and section.
- `atomzero-doctor` now validates an existing document index (chunk/source
  counts, orphan rows and FTS5 integrity) in addition to file write access.
- Added doctor tests for absent, clean and orphaned indexes.
- 246 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.4.0.md`](docs/RELEASE_ATOMZERO_0.4.0.md).

## atomzero 0.3.0 - document workflow and grounded answers (2026-11-22)

- Added `atomzero-docs`, a headless CLI for add/sync/status/doctor/search/
  answer/summary/evidence/export/import/list/forget.
- Added recursive directory ingestion with per-file reports,
  `--skip-unchanged` and `--dedupe` content-hash deduplication.
- Added `atomzero-docs sync`: portable relative-path collections, unchanged
  file skips, duplicate detection and pruning of deleted files.
- Added `atomzero-docs status` (dry-run directory diff) and
  `atomzero-docs doctor` (orphan-row and FTS integrity check).
- Added extractive document summaries (`atomzero-docs summary`,
  `GET /v1/documents/summary` and a UI Summary action) with page/section
  citations; no model generation and no new facts.
- Chat replies now use the shared grounded-answer core and expose
  `evidence`/`confidence`; the UI shows the match label.
- Added JSONL index export/import with titles, teacher/provenance metadata and
  page/section locators preserved.
- Added source, source-prefix and kind filters to document search.
- Added `POST /v1/documents/answer` and `atomzero-docs answer`: cited answer,
  evidence list and a transparent heuristic match-confidence score from one
  shared core.
- Improved lexical reranking with a small morphology allowance
  (`claim`/`claims`, `submit`/`submitted`).
- Answer snippets now prefer answer-bearing sentences with numeric/unit cues,
  add only adjacent context, and drop notes far below the winning score.
- Added a "Grounded answer" control, an indexed-source picker, per-document
  "Ask", and "Copy citation" actions to the local document UI.
- Added answer-in-visible-snippet and confidence metrics to the public SQuAD
  retrieval benchmark.
- Added tests for all of the above; 233 pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.3.0.md`](docs/RELEASE_ATOMZERO_0.3.0.md).

## atomzero 0.2.0 - public benchmark and demo UI (2026-11-22)

- Added a public SQuAD v2 retrieval benchmark (`benchmarks/squad_retrieval.py`)
  with 20x100 and 50x200 results.
- Added one-click RAG demo chips to the UI: launch code, warranty, visitor
  badge and a no-source Paris question.
- Kept the 0.1.8 doctor, 0.1.9 reset-demo/version flags/troubleshooting and
  donation addresses.
- No model/runtime weights changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.2.0.md`](docs/RELEASE_ATOMZERO_0.2.0.md).

## atomzero 0.1.9 - support and onboarding polish (2026-11-22)

- Added EVM, Solana and Bitcoin donation addresses to the README, PyPI
  description, HF/ModelScope model card, project overview and articles.
- Added `atomzero-quickstart --reset-demo`.
- Added `atomzero --version` and `atomzero-quickstart --version`.
- Added `docs/TROUBLESHOOTING.md`.
- No model/runtime behavior changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.9.md`](docs/RELEASE_ATOMZERO_0.1.9.md).

## atomzero 0.1.8 - install doctor (2026-11-22)

- Added `atomzero-doctor`: Python/package/dependency/torch-CUDA/cache/port/
  write-access/Ollama preflight checks with actionable hints and `--json`.
- Documented the doctor in the README.
- No model/runtime behavior changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.8.md`](docs/RELEASE_ATOMZERO_0.1.8.md).

## atomzero 0.1.7 - PyPI docs link cleanup (2026-11-22)

- Republished the package description with all documentation links pointing to
  the `atom-zero` repository.
- Removed the old `stream-finetune` documentation links from the PyPI page.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.7.md`](docs/RELEASE_ATOMZERO_0.1.7.md).

## atomzero 0.1.6 - LinkedIn and ModelScope restored (2026-11-22)

- Restored LinkedIn and ModelScope as canonical Atom Zero release surfaces.
- Added both links back to the package README, PyPI metadata, HF model card and
  project overview.
- Added a job-search-oriented LinkedIn post for Atom Zero 0.1.6.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.6.md`](docs/RELEASE_ATOMZERO_0.1.6.md).

## atomzero 0.1.5 - canonical surfaces cleanup (2026-11-22)

- Removed the remaining ModelScope link from the published package description
  and the LinkedIn project URL from package metadata.
- Documented the five canonical Atom Zero surfaces: GitHub, Hugging Face, PyPI,
  dev.to and Medium.
- Superseded by `0.1.6`: LinkedIn and ModelScope are canonical surfaces again.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.5.md`](docs/RELEASE_ATOMZERO_0.1.5.md).

## atomzero 0.1.4 - standalone repository (2026-11-22)

- Moved the Atom Zero package to its own GitHub repository:
  <https://github.com/byjameson/atom-zero>.
- PyPI project URLs now point to the standalone repository.
- Current Atom Zero links target the `atom-zero` repository, `atomzero` PyPI,
  Hugging Face and the two launch articles. ModelScope/LinkedIn were temporarily
  removed in 0.1.4 and restored as canonical surfaces in 0.1.6.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.4.md`](docs/RELEASE_ATOMZERO_0.1.4.md).

## atomzero 0.1.3 - full package description (2026-11-22)

- Expanded the standalone GitHub/PyPI README to cover the full system: model,
  training, RAG, documents/citations, evidence, tools, research, curiosity,
  learning ladder, provider, safety, privacy, benchmarks and limits.
- Removed stale streamft links from the Atom Zero model card; it now links to
  the atomzero package, source and `atomzero-v0.1.3` release.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.3.md`](docs/RELEASE_ATOMZERO_0.1.3.md).

## atomzero 0.1.2 - standalone quickstart (2026-11-22)

- Added `atomzero-quickstart`: downloads the public checkpoint/tokenizer from
  Hugging Face, seeds a demo corpus and starts the local UI.
- No streamft checkout, trainer install or `git clone` is required.
- Added `--provider`, `--safety-filter`, `--port`, `--no-download` and
  `--no-browser` options.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.2.md`](docs/RELEASE_ATOMZERO_0.1.2.md).
