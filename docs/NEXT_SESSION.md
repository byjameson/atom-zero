# Next session - 1.6.0 in progress

Snapshot: 2026-11-26, paused for PC shutdown.
Head: the 1.6.0 data/tooling merges; no training is running; GPU idle.
No Vast.ai instance is running; credit about **$8.88**.
Sponsor goal: **$10,000/month**.

## Released baseline

`1.5.0` is published and verified. The default is the **63M balanced** checkpoint
(`byjameson/atom-zero-63m-balanced`, weights `0.9.0`):

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

The 49M `0.8.0` base is the fallback.

## 1.6.0 work so far

Merged and tested (35 tests):

- Teacher benign chat generator (`research/build_teacher_benign_chat.py`).
- Product suite 1.6: RAG cases + separate multi-turn rates
  (`research/product_suite.py`, `examples/rag_suite_v1/`).

Teacher run (Qwen2.5-0.5B, host GPU):

```text
requested 400 -> accepted 260, rejected attempts 437, duplicates 0
identity 50, greeting 50, explanation 19, cooking 28,
travel 15, coding 16, planning 32, general 50
```

Expanded-suite results:

| variant | mix | benign | safety | self-harm | copy | multi-turn | gate |
|---|---|---:|---:|---:|---:|---:|---|
| J | 1.5.0 default | 93.4% | 40/40 | 16/16 | 48.5% | 75% / 50% | pass |
| K | J + teacher 260x2 | **98.5%** | 39/40 | 14/16 | 48.5% | **93.8% / 75%** | fail |
| K2 | safety/selfharm 8x | 89.0% | **40/40** | **16/16** | **53.5%** | 62.5% / 56% | pass |

K gains conversation but loses one safety probe and two self-harm pairs. K2
restores the gates but drops below 90% benign. Neither beats J on the full
profile.

## Resume here

K3 (middle point) is built but not trained:

```text
data/mixed_benign_k3/train.jsonl  (8980 rows)
base 6492 + benign 480x3 + teacher 260x2 + safety 40x6 + selfharm 16x6
+ boundary 96x2; seed 97
cache: data/pretok/mixed_benign_k3
```

Train:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python research/finetune_copy_control.py `
  --base checkpoints/upcycle/atom-zero-49m-8l `
  --train data/mixed_benign_k3/train.jsonl `
  --cache-dir data/pretok/mixed_benign_k3 `
  --eval data/staged_copy_v1/mixed_eval.jsonl `
  --out checkpoints/mixed8_benign_k3 `
  --steps 3000 --lr 0.00005 --batch-size 4 --max-seq 256 `
  --content-weight 2 --seed 71 `
  --result research/results/mixed8_benign_k3.json `
  --comparison research/results/mixed8_benign_k3_cmp.json
```

Then run the expanded product suite (with RAG), the main gate, the self-harm
gate, server suite, 2/3-short copy and retention. Promote K3 only if it beats
the J profile: benign >= 93.4%, safety 40/40, self-harm 16/16, copy >= 48.5%,
multi-turn >= 75%, retention <= 0.703.

## After K3

1. If K3 wins, train 24L on the same mix on Vast.ai with `--copy-checkpoint`.
   Verify the checkpoint size before destroy; the previous 24L transfer stopped
   at 31 MB. If scp drops again, upload to HF from the instance with a scoped
   token or split the transfer.
2. Re-read the 1.4.0 24L self-harm result with the kind-aware gate; with the
   fixed data it is 8 benign + 1 crisis = 1/8 crisis refusals, so the old F mix
   is genuinely weak.
3. If K3 does not win, keep the 63M balanced default and publish the 1.6.0
   frontier: teacher data improves conversation but the safety/benign balance
   is still the binding constraint.
4. Full VM suite (1693+ tests) and benchmark before any 1.6.0 release.

## Rules

- Training on host GPU or rented GPU; tests and gates on the VMware Ubuntu box.
- Check for leftover Vast.ai instances and destroy them.
- Release only at the 1.6.0 boundary with fresh posts. Donation addresses stay
  unchanged. Default weights change only after the full product suite and human
  review.
