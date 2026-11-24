# Capability scorecard

This scorecard tracks narrow, reproducible measurements of the local 49M
model. It is not a general benchmark, an MMLU-style score or evidence of
general reasoning. A row is valid only for the named split, adapter geometry
and seed set.

## Reporting rules

- Use one fixed eval JSONL and keep keys/values disjoint from training.
- Repeat every arm across at least three seeds.
- Report exact match and key-copy rate separately. Aggregate loss alone can
  hide a broken or unchanged copy circuit.
- Report mean, sample standard deviation, min and max per arm.
- Do not claim a winner when the 95% t-intervals overlap. With three seeds
  only large effects are detectable; overlap means "not established".
- Store numeric metrics and paths only. Never copy prompts, answers or
  generated text into the scorecard.

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
