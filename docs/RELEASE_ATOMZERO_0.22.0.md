# atomzero 0.22.0 - curriculum loop inside self-improve

Status: **intermediate patch, published 2026-11-24**. The next 0.1 release boundary is **0.30.0**. Base weights remain `0.8.0`.

## Highlights

- `atomzero-self-improve` can now run the staged copy curriculum as a bounded
  job with `--adapter-curriculum-command`.
- The loop records a content-free `curriculum` block: return code, elapsed
  time, gate status, exact match, key copy, promotion result and result path.
- It fails closed: a nonzero runner exit, missing/invalid ledger, non-passing
  gate or missed loop-side threshold sets `status: curriculum_failed`.
- New flags: `--adapter-curriculum-data-dir`,
  `--adapter-curriculum-final-eval`, `--adapter-curriculum-min-exact-match`,
  `--adapter-curriculum-min-key-copy-rate`.
- The 0.21.0 behavior-passing adapter remains in the separate curriculum
  registry, not the product default: 40.0% exact / 54.8% key copy on the
  mixed held-out set.
- 1271 tests passed on the VM in 46.34s; benchmark 20/20.

## Why it matters

The self-improvement loop is no longer only code edits and tests. It can run a
measured capability-growth job, read the behavior gate, record promotion, and
stop on failure without human approval. That is the first end-to-end
curriculum -> gate -> registry path inside the autonomous loop.

It is not general self-improvement yet. Multi-key copying is still weak
(8-12% exact), and no product adapter is active. The claim is narrow and
measured.

## Install

```powershell
pip install atomzero==0.22.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.22.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
