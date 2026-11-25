# Atom Zero 0.70.0: verified teacher distillation gives partial 2-3 key copy

Atom Zero 0.70.0 is the 0.7 boundary. The result is small but real: a larger
teacher’s verified output improves 2-3 key copying in the 49M student. The
4-key wall is still there.

## The pipeline

```text
staged train prompts
  -> Qwen2.5-0.5B teacher (host GPU)
  -> tolerant parse + exact key/value verification
  -> canonical flat JSON / KV / YAML
  -> 49M student training (host GPU)
  -> held-out 2/3/4-key evaluation
```

Only verified rows are used. The teacher cannot see eval files.

## Teacher run

300 attempted prompts produced **153 accepted rows**:

```text
json 66, kv 87, yaml 0
2short 66, 3short 55, 4short 32
rejected 147
```

The YAML acceptance was zero, so the pipeline reports the gap instead of
pretending the formats are equal.

## Student result

`distill_student_153` started from `multi_recover` and trained on the teacher
rows plus 50% rehearsal. Mixed loss was essentially flat
(`0.7769 -> 0.7683`), but held-out copy improved:

| stage | distilled student | multi_recover |
|---|---:|---:|
| 2short (50 rows) | **32% exact / 50% key-copy** | 0% / 0% |
| 3short (50 rows) | **12% / 28%** | 0% / 0% |
| 4short (20 rows) | 0% / 0% | 0% / 0% |

This is a **partial 2-3 key win**. The 4-key wall remains; the runtime
`atomzero-copy` tool is still the exact 4-key path.

## Why the loss is flat

Most teacher rows canonicalize to the same key/value content as the source
oracle. Hard-label distillation tests clean surface variation, not new
knowledge. The next experiment should use new-task generation or soft-target
distillation, or grow the model.

## Verification

- VM: 1493 tests passed in 50.08s.
- Benchmark: 20/20.
- No model weights changed; base remains 0.8.0.

## Install

```powershell
pip install atomzero==0.70.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.70.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The wall is smaller, not gone. The claims stay narrow.
