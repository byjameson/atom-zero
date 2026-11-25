# Atom Zero 0.80.0: the 4-key copy wall was capacity, not data

Atom Zero 0.80.0 answers a question the last three releases could not: why does
the 49M model copy two or three arbitrary keys but fail at four?

The answer is capacity. A 63M model built by depth upcycling the 49M checkpoint
and fully fine-tuning it now reaches **65.5% exact / 74.0% key-copy on 200
held-out 4-key rows**. The 49M model stays at 0%.

## The experiment

```text
49.2M, 6 layers, 768 dim
  -> depth upcycle to 8 layers (interpolate copied blocks)
  -> 63,389,184 params (+28.8%)
  -> full fine-tune on copy curriculum + 153 verified teacher rows
  -> 200 held-out rows per stage
```

Results, all on held-out rows with fresh keys and values:

| stage | 49M distilled | 63M upcycled |
|---|---:|---:|
| 2short | 32% exact / 50% key-copy | **89.5% / 91.5%** |
| 3short | 12% / 28% | **77.0% / 83.5%** |
| 4short | 0% / 0% | **65.5% / 74.0%** |

The 49M comparison uses `distill_student_153` from 0.70.0. The 63M model is
fully fine-tuned, not a LoRA adapter.

## The control: data alone does not help

To separate data from capacity, a Qwen2.5-0.5B teacher generated **new** copy
tasks with keys and values never used in the staged sets:

```text
requested 1000 -> accepted 656, rejected 344
json 332, kv 324
2-key 194, 3-key 216, 4-key 246
```

Training the 49M student on those 656 new rows plus rehearsal produced:

| stage | 49M + new tasks |
|---|---:|
| 2short (20) | 20% exact / 30% key-copy |
| 3short (20) | 15% / 30% |
| 4short (20) | **0% / 5%** |

More 4-key data did not move the 49M model. Capacity did.

## The boundary

A 30-row out-of-distribution test used `field_` and `k_` prefixed keys, a style
the model never saw in training. It scored **0/30**. The 63M model copies
within the trained key grammar (3-6 lowercase alphanumeric keys, short values);
it is not a general copier and does not prove a general reasoning gain.

This is why the product base stays `0.8.0`. The 63M checkpoint was fully
fine-tuned on copy data. It is a copy specialist, not a better assistant.
Promoting it would trade source-cited QA behavior for a narrow skill. It
remains a research artifact until it passes the content, exact-match, key-copy
and safety gates as a complete base.

## What is new in the repository

- `research/upcycle_model.py`: deterministic depth expansion with
  interpolation, forward-shape checks and research-only metadata.
- `research/build_distill_new_tasks.py`: teacher new-task generation with
  deterministic validation, dedupe and a content-free manifest.
- Evidence: `research/results/upcycle_8l_capability_summary.json`.

## Verification

- VM Ubuntu: **1521 tests passed in 47.52s**.
- Benchmark: 20/20.
- Base weights: `0.8.0` (unchanged).
- No adapter is promoted.

## Install

```powershell
pip install atomzero==0.80.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.80.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The wall was real. It was made of parameters, not examples.
