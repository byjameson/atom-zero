# atomzero 0.80.0 - depth upcycling breaks the 4-key copy wall at 63M

Status: **release candidate**. Product base weights remain `0.8.0`; the 63M
checkpoint is a research artifact.

## Highlights

- **Depth upcycling.** The 6-layer, 49.2M `0.8.0` checkpoint was expanded to
  **8 layers / 63,389,184 parameters** with interpolated block copies
  (`[0,0,1,2,3,3,4,5]`), then fully fine-tuned on the copy curriculum plus
  verified teacher rows.
- **The 4-key wall was capacity, not data.** On 200 held-out rows per stage:

| stage | 49M `distill_student_153` | 63M upcycled + full FT |
|---|---:|---:|
| 2short | 32% exact / 50% key-copy | **89.5% / 91.5%** |
| 3short | 12% / 28% | **77.0% / 83.5%** |
| 4short | 0% / 0% | **65.5% / 74.0%** |

- **Data-only control.** A new-task teacher generator produced **656 verified
  tasks** (246 of them 4-key) from 1000 requested. Training the 49M student on
  that fresh content moved 4short only to **0% exact / 5% key-copy** and left
  2-3 key performance flat. New data alone did not break the wall.
- **The claim is bounded.** A 30-row out-of-distribution set with `field_` and
  `k_` prefixed keys scored **0/30**. The 63M model copies within the trained
  3-6 lowercase alphanumeric key grammar; it is not a general copier.
- **Verification:** 1521 tests passed on the VM in 47.52s; benchmark 20/20.

## What ships

- `research/build_distill_new_tasks.py`: teacher-generated new copy tasks with
  deterministic validation, dedupe and content-free manifest.
- `research/upcycle_model.py` and `research/upcycle_report.py`: depth expansion
  with research-only metadata (`research_control: true`,
  `product_weights: false`).
- `checkpoints/fullft_upcycle8l_v1`: 63M copy-specialized research checkpoint.
- Full evidence in `research/results/upcycle_8l_capability_summary.json` and
  `docs/UPCYCLE.md`.

## Why the product base stays 0.8.0

The 63M checkpoint was fully fine-tuned on copy data only. It is excellent at
the copy task and is not a general assistant. Promoting it would trade the
product's source-cited QA behavior for a narrow copy skill. It stays a research
artifact until it passes the content, exact-match, key-copy and safety gates as
a complete base.

## Install

```powershell
pip install atomzero==0.80.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.80.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
