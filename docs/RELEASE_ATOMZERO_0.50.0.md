# atomzero 0.50.0 - copy routing, weak-stage rehearsal and QKV LoRA

Status: **release candidate**. Base weights remain `0.8.0`.

## Highlights

- **Copy tool router.** Explicit transcription requests now route to
  `atomzero-copy` in the product chat/agent paths and carry
  `source: "copy_tool"`. The router is conservative: questions, summaries and
  open-ended chat keep the previous model/RAG path. It handles 2-4 explicit
  JSON/KV/YAML pairs.
- **Weak-stage rehearsal loop.** `research/run_weak_stage_rehearsal.py` ranks
  stages, builds a weak-focused mix and retrains with before/after ledgers.
  Two real configs did **not** improve the aggregate:
  - gentle: multi 12% -> 15% exact, but echo_json 76% -> 65%;
  - aggressive: every stage dropped.
  The tool is useful, but this recipe is not a win yet.
- **Fused QKV attention LoRA.** `--qkv-rank`/`--qkv-alpha` adapt the fused
  `in_proj_weight`, not just out_proj + MLP. The first from-base multi-short run
  reached only **6% exact / 18% key copy** on the mixed set (3short 0/3,
  4short 0/0). It needs a proper control and longer training before any
  conclusion.
- **VM disk** grown from 20 GB to **40 GB** (26 GB free) with `growpart` +
  `resize2fs`.
- **1399 tests** passed on the VM in 44.36s; benchmark 20/20.

## Honest interpretation

0.50.0 is a research/tooling batch, not a capability jump. The copy tool router
makes the runtime transcription path usable in chat. The weak-stage and QKV
experiments are negative first results and are documented as such. The 3-4 key
wall remains open.

## Install

```powershell
pip install atomzero==0.50.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.50.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
