# Six-criterion capability scorecard

This is an honest, content-free scorecard for six separate capabilities. It is
not a single aggregate loss number and it is not a claim that Atom Zero is
generally capable.

The scorecard reads only numeric metrics and fixed labels from result JSON. It
never writes prompt text, answer text, raw completions or free-form notes.

## Current scorecard

Generated from the checked-in result files listed under
[Input sources](#input-sources).

| # | Criterion | Status | Score | Evidence | Next measurable target |
|---:|---|---|---:|---|---|
| 1 | Discovers its own knowledge gaps | partial | 0.35 | Answerability ECE 0.064 passes, but Brier 0.233 misses, refusal recall is 0.546, and 2/2 specs FAIL. No content-free gap-closure ledger was supplied. | Close one self-discovered gap end-to-end; reach Brier <=0.15, refusal recall >=0.80 and zero failed specs. |
| 2 | Decides what, when and how to learn | partial | 0.80 | First curriculum attempt failed closed (0.37 exact / 0.55 key copy); the second passed (0.53 / 0.60) and was promoted. Selection is still human-gated. | One autonomous learn/stop decision with no human approval, a rejected alternative and a passing held-out gate. |
| 3 | Retains old knowledge while learning | partial | 0.53 | Best rehearsal kept only 2/5 stages within 0.02; mixed copy is 0.48 exact / 0.604 key copy; 3-4 arbitrary keys remain at 0.06 exact. | Keep >=4/5 stages within 0.02 after a new learning step and reach mixed exact >=0.70. |
| 4 | Transfers learning to different problems | not_met | 0.04 | Held-out arithmetic, city-YAML and city/river are all 0/30 exact; adapter content loss is worse on arithmetic and city-YAML. Copy curriculum is 0/200 exact. The 1.0 runtime copy tool is tool use, not model transfer. | >=0.50 exact on held-out city/river and arithmetic/YAML with n>=100; >=0.30 exact on 3-4 keys. |
| 5 | Improves its own learning algorithm | not_met | 0.30 | Weak-stage rehearsal and key-weighted loss were negative; QKV LoRA's first result is negative/uncontrolled. Process evidence is not counted as improvement. | One learning-algorithm change that beats its own mixed-exact baseline by >=0.05, retains >=4/5 stages and reproduces across >=2 seeds. |
| 6 | Does this at low hardware cost | met | 1.00 | 8 vCPU / 16 GB RAM / 40 GB disk; 49.2M base; 405,504 adapter parameters; 149.0 s recorded adapter training; 1,399 tests. | Keep the full loop inside the envelope while adding a held-out transfer eval; report peak RAM and wall-clock per cycle. |

Status counts: **met 1 Â· partial 3 Â· not_met 2 Â· unknown 0**. No aggregate
score is reported because the six criteria are not interchangeable.

## Held-out tasks

The current held-out evidence is in
[`research/results/heldout_task_transfer_summary.json`](../research/results/heldout_task_transfer_summary.json):
30 rows per task, base and `multi_recover` both 0% exact.

| Task | n | Base exact | Adapter exact | Model transfer? | Status |
|---|---:|---:|---:|---|---|
| city/river | 30 | 0.000 | 0.000 | yes | not_met |
| 3-4 arbitrary keys | â€” | â€” | 0.060 | yes | not_met |
| arithmetic | 30 | 0.000 | 0.000 | yes | not_met |
| city YAML | 30 | 0.000 | 0.000 | yes | not_met |
| runtime copy tool | 200 | â€” | 1.000 | **no, tool use** | met |

The runtime copy tool is a deterministic transcription tool. Its 1.0 accuracy
is a harness result, not evidence that the model learned to transfer copying.
It is excluded from the transfer score.

The held-out copy result is not rescued by a lower loss: on arithmetic the
adapter content loss is 5.142 versus 4.391 for the base, and on city-YAML it is
3.632 versus 3.484. Lower loss without an exact match is not transfer. On
city/river the adapter content loss improves (3.243 versus 4.794) but exact
match is still 0/30, so the scorecard does not count it as a win.

## Capacity probe

A larger pretrained model was measured as a capacity reference, not as a
product model. Qwen2.5-0.5B-Instruct, 20 rows per stage, tolerant parsing:

| Stage | Baseline exact | Flat-prompt exact |
|---|---:|---:|
| 2short | 65% | 85% |
| 3short | 55% | 65% |
| 4short | 55% | 85% |

This is far above the 49M adapter's 0-6% on 3-4 keys and confirms that the copy
wall is capacity-related. It is **not** counted as 49M transfer: the model is
pretrained, has a different tokenizer/data/compute, and scored 0/10 exact on
arithmetic and city-YAML zero-shot. The decision is to keep `0.8.0` as the
low-cost default and use larger models as teachers/research backends.


## Distillation evidence (0.70.0)

A verified Qwen teacher distillation cycle improved 2-3 key copy in the 49M
student: 2short **32% exact / 50% key-copy**, 3short **12% / 28%**, 4short
**0%** on held-out sets (50/50/20 rows). The 2-3 key gain is real but narrow;
the 4-key wall remains and the runtime copy tool is still the exact path. This
is counted as partial retention/transfer evidence, not as a general model win
and not as a learning-algorithm improvement.

## Mixed replay and gate evidence (0.90.0)

The 8-layer 63M candidate fine-tuned on copy + content + safety replay passes
the full checkpoint gate on two seeds: 4-key exact **57.0% / 58.0%**, key-copy
**67.0% / 63.5%**, content loss **0.722 / 0.809** of the 49M baseline (lower is
better), and safety **40/40** on held-out probes versus 12/40 for the baseline.
The trade is explicit: safety replay costs about 9 points of 4-key exact versus
the copy-only model. A 24-layer 177M model was rejected: +4 points on 4-key and
retention 1.66. This is capacity + data evidence inside the trained key grammar,
not general transfer or a product-base promotion.
## Scoring rules

The status vocabulary is fixed:

- `met`: the criterion-specific gate passed;
- `partial`: some measured evidence, but not the full gate;
- `not_met`: measured evidence exists and the criterion failed;
- `unknown`: no defensible measurement was supplied. Missing data is never
  silently treated as a pass.

Scores are bounded to `[0, 1]` and rounded to three decimals. They are
defensible summaries of the named measurements, not probabilities.

### 1. Gap discovery

`0.25` for a recorded gap action, `0.20` for an answerability measurement,
`0.15` for ECE <=0.10, `0.20` for Brier <=0.15 and `0.20` for refusal recall
>=0.80. `met` additionally requires zero failed answerability specs. A high
score is impossible without both a gap mechanism and a calibrated refusal
signal.

### 2. Learning policy

`0.25` for a recorded learning attempt, `0.20` for a fail-closed rejection,
`0.15` for a promotion, `0.20` for a recorded selection among two or more
measured strategies and `0.20` for an end-to-end decision with no human gate.
`met` additionally requires the autonomous end-to-end decision.

### 3. Retention

`0.50 * retention_fraction + 0.30 * min(mixed_exact / 0.80, 1) + 0.20 *
min(mixed_key_copy / 0.80, 1)`. A stage is retained when both its exact and
key-copy deltas are at least `-0.02`. `met` additionally requires
`retention_fraction >= 0.80` and `mixed_exact >= 0.70`.

### 4. Transfer

Weighted, renormalised over the components that were measured:

- city/river: `0.50 * min(exact / 0.50, 1)`;
- arithmetic/YAML: `0.30 * min(exact / 0.50, 1)`;
- 3-4 arbitrary keys: `0.20 * min(exact / 0.30, 1)`.

`met` requires city/river, arithmetic and city-YAML each at `>=0.50`; a single
strong arithmetic/YAML family is not enough. `partial` requires the best
measured exact rate at `>=0.20`; otherwise `not_met`. Runtime copy-tool
accuracy is not an input to this formula.

### 5. Improving its own learning algorithm

`0.10` for a recorded change attempt, `0.10` for a measured comparison, `0.10`
for a fail-closed/reverted negative result and `0.70` for a change that beats
its own aggregate baseline by `>=0.05`. `met` requires the positive result.
Attempts and negative results are evidence of process, not improvement.

### 6. Low hardware cost

Each supplied resource scores `1` when it is inside the envelope and
`limit / value` when it is outside. The envelope is **8 vCPU, 16 GB RAM,
40 GB disk**. All three resources must be supplied and inside the envelope for
`met`; a partial resource set is capped at `0.80` and cannot be `met`.

## Honest interpretation

- **Only low hardware cost is met.** That says the loop fits a small machine;
  it does not say the model is capable.
- **Gap discovery is partial, not solved.** The answerability signal is
  calibrated at ECE 0.064, but Brier 0.233 is above the 0.15 target, refusal
  recall is 0.546, and 2/2 specs FAIL. The system still misses many
  unanswerable inputs.
- **Learning policy is partial.** The loop can reject a bad candidate and
  promote a later one, and it can rank stages before choosing a mix. The
  curriculum and the promotion decision are still human-authored/human-gated,
  so this is not autonomous learning policy.
- **Retention is partial and fragile.** The best rehearsal retained 2/5 stages
  within 0.02; the first rehearsal retained 1/5. Mixed exact is 0.48 and 3-4
  key exact is 0.06. Learning a new stage still erodes old stages.
- **Transfer is not met.** Base and `multi_recover` are both 0/30 exact on
  arithmetic, city-YAML and city/river. The copy curriculum is 0/200 exact
  across its five city/river families. Lower content loss on one family does
  not override zero exact match.
- **Improving the learning algorithm is not met.** Weak-stage rehearsal,
  key-weighted loss and the first QKV run are negative. The scorecard records
  the attempts, but does not turn them into a capability claim.
- **The runtime copy tool is not model capability.** It is 1.0 on explicit
  transcription by construction. It is reported separately so it cannot inflate
  the transfer criterion.

## Caveats

- The held-out tasks use **30 rows per task**. `0/30` is a clear failure, but
  the exact confidence interval is wide; this is a smoke test, not a
  publication-grade transfer benchmark.
- The thresholds (`0.50`, `0.15`, `0.10`, `0.80`, `0.02`, the hardware
  envelope) are project conventions chosen to make failure visible. They are
  not universal standards.
- The scorecard is only as honest as the supplied files. A missing file
  produces `unknown`, not a pass. The current run supplies the answerability
  profile, the copy-tool benchmark and the held-out transfer summary.
- The `0.30` score for learning-algorithm improvement is process evidence
  (attempts, measurements, a negative result). The status remains `not_met`
  because no change improved the aggregate target.
- The results come from narrow synthetic copy and QA tasks. They do not
  establish general reasoning, coding ability or production readiness.
- The generated JSON has no timestamp. Use the commit and the listed input
  sources to identify a run. Do not hand-edit the generated JSON or Markdown.

## Regenerate

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/capability_scorecard.py `
  --result research/results/self_improve_curriculum_summary.json `
  --result research/results/multi_recover_summary.json `
  --result research/results/multi_key_0.30_summary.json `
  --result research/results/weak_stage_rehearsal.json `
  --result research/results/weak_stage_rehearsal_gentle.json `
  --result research/results/copy_curriculum_summary.json `
  --result research/results/copy_tool_benchmark_summary.json `
  --result research/results/heldout_task_transfer_summary.json `
  --answerability benchmarks/results/answerability_hybrid/answerability_profile.json `
  --answerability-specs-failed 2 --answerability-specs-total 2 `
  --cpu 8 --ram-gb 16 --disk-gb 40 `
  --model-params-m 49.2 --adapter-params 405504 --tests 1399 `
  --json-out research/results/capability_scorecard.json `
  --md-out research/results/capability_scorecard.md
```

The generated table is
[`research/results/capability_scorecard.md`](../research/results/capability_scorecard.md)
and the machine-readable record is
[`research/results/capability_scorecard.json`](../research/results/capability_scorecard.json).

## Input sources

- `research/results/self_improve_curriculum_summary.json`
- `research/results/multi_recover_summary.json`
- `research/results/multi_key_0.30_summary.json`
- `research/results/weak_stage_rehearsal.json`
- `research/results/weak_stage_rehearsal_gentle.json`
- `research/results/copy_curriculum_summary.json`
- `research/results/copy_tool_benchmark_summary.json`
- `research/results/heldout_task_transfer_summary.json`
- `benchmarks/results/answerability_hybrid/answerability_profile.json`
- `docs/BATCH_0.60.0.md` (QKV first-result caveat)

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m pytest tests/test_capability_scorecard.py -q
```

## Controlled QKV vs MLP ablation

## QKV vs MLP LoRA on multi-key copy

Status: runner added; no multi-seed result has been committed yet.

The 49M base plus MLP and `out_proj` LoRA reaches 48% exact / 60.4% key copy
on the named `multi_recover` mixed split, while three and four arbitrary keys
stay at 0-6%. A first from-base QKV run scored 6% exact / 18% key copy. That
is one run, not a comparison. The controlled A/B holds seed, data order,
steps, batch size, learning rate, loss weights, MLP geometry and `out_proj`
geometry fixed:

| arm | MLP | `out_proj` | fused QKV |
|---|---|---|---|
| `qkv4` | r8/a8 | r4/a8 | r4/a8 |
| `mlp` | r8/a8 | r4/a8 | off |

`mlp` keeps the MLP and `out_proj` adapters; "MLP-only" means the fused QKV
projection is frozen, not that attention is untouched.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/run_qkv_capacity_ablation.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/multi_short_v1/stage_3short/train.jsonl `
  --eval data/multi_short_v1/stage_3short/eval.jsonl `
  --arms qkv4,mlp --seeds 23,29,37 `
  --steps 4000 --batch-size 4 --lr 3e-4 `
  --content-weight 2 --key-weight 3 --value-weight 2 `
  --out checkpoints/qkv_capacity_ablation `
  --result research/results/qkv_capacity_ablation.json
```

The JSON report has this shape:

```text
schema, created, content_free, status
base, train, eval, eval_rows, eval_rows_used
eval_subset.{applied,selection,seed,rows,total_rows,sha256}
settings.{arms,seeds,steps,batch_size,lr,content_weight,key_weight,
          value_weight,rank,alpha,rank_attn,alpha_attn,qkv_rank,qkv_alpha}
arms.<arm>.{geometry,status,completed,failed,runs,aggregate}
arms.<arm>.aggregate.<metric>.{count,mean,std,min,max,ci95}
arms.<arm>.runs[] = {arm,seed,status,returncode,elapsed_seconds,
                     exact,key_copy,key_recall,loss,content_loss}
verdict.{claim,qkv_improves,metrics,note}
fair_comparison.{same_*,only_planned_difference,qkv_adds_trainable_parameters}
```

A sibling `.md` report renders the same numbers. `verdict.claim` is one of
`qkv_improves`, `mlp_better`, `mixed`, `inconclusive` or `not_applicable`.
Only the first two imply a directional claim, and only when both exact match
and key copy have non-overlapping intervals.

QKV rank 4 adds 73,728 trainable parameters. A QKV win can therefore be a
capacity effect, not an attention-locus effect. The matched follow-up is an
MLP widening with the same added parameter budget, run through the same
seeds and data. See [`docs/QKV_LORA.md`](QKV_LORA.md) for the adapter
details and limitations.

## Capacity evidence (0.80.0)

Depth upcycling the 49M checkpoint to 8 layers (**63,389,184 params**) and
fully fine-tuning on the copy curriculum broke the 4-key wall on held-out rows:
**65.5% exact / 74.0% key-copy** on 200 four-key rows, versus 0% for the 49M
student. A data-only control with 656 new teacher tasks did not move the 49M
model (0% / 5%). The gain is bounded: `field_`/`k_` prefixed out-of-distribution
keys score 0/30. This is counted as capacity evidence for the copy task, not as
general transfer and not as a product-base promotion.
