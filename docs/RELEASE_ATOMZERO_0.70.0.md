# atomzero 0.70.0 - verified teacher distillation gives partial 2-3 key copy

Status: **published on 2026-11-25**. Base weights remain `0.8.0`.

## Highlights

- **Verified teacher distillation pipeline.** Qwen2.5-0.5B-Instruct generates
  copy answers on the host GPU. A tolerant parser recovers the exact pairs, and
  only rows whose keys and values match the source are canonicalized into 49M
  training data.
- **300-prompt teacher run:** 153 accepted rows (json 66, kv 87, yaml 0;
  2short 66, 3short 55, 4short 32), 147 rejected. The pipeline records the
  acceptance counts instead of assuming all prompts produced data.
- **Distilled student `distill_student_153`:** started from `multi_recover`,
  trained on the 153 teacher rows plus 50% rehearsal. Mixed loss was
  essentially flat (`0.7769 -> 0.7683`), but held-out 2-3 key copy improved:

| stage | distilled student | multi_recover |
|---|---:|---:|
| 2short (50 rows) | **32% exact / 50% key-copy** | 0% / 0% |
| 3short (50 rows) | **12% / 28%** | 0% / 0% |
| 4short (20 rows) | 0% / 0% | 0% / 0% |

- **Honest reading:** this is a partial 2-3 key distillation win. The 4-key
  wall is unchanged. The runtime `atomzero-copy` tool remains the exact 4-key
  path; it is tool use, not model capability.
- **1493 tests** passed on the VM in 50.08s; benchmark 20/20.

## Why the mixed loss is flat

Most of the teacher rows are canonicalized to the same key/value content as the
source oracle answers. Hard-label distillation therefore tests whether clean
teacher surface variation and selection help, not whether new content is
added. The 2-3 key exact gain is real on the held-out sets, but it is narrow.
A future cycle should test new-task generation or soft-target distillation.

## Install

```powershell
pip install atomzero==0.70.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.70.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-0700-verified-teacher-distillation-gives-partial-2-3-key-copy-31d5>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-0-70-0-verified-teacher-distillation-gives-partial-2-3-key-copy-45fef0982f0d>
