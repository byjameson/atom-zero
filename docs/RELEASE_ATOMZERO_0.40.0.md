# atomzero 0.40.0 - real autonomous promotion and a runtime copy tool

Status: **release candidate**. Base weights remain `0.8.0`.

## Highlights

- **Real self-improve curriculum promotion.** A VM run trained `echo_kv` and
  `echo_json`, evaluated a 100-row mixed held-out set, passed the 0.40 exact /
  0.50 key-copy / content gate with **53% exact / 60% key copy**, and promoted
  automatically to `data/self_improve_curriculum_active.json`.
  A first attempt at a 0.40 exact threshold failed closed on content loss and
  did **not** promote.
- **Runtime copy tool.** `atomzero-copy` parses explicit `key=value` pairs and
  emits JSON/KV/YAML deterministically. On the real 4-key eval set it scores
  **200/200 exact, key-copy 1.0**. This is tool use, not a model capability.
- **Curriculum runner `--init-adapter`.** The first stage can start from an
  existing adapter; later stages chain from the previous stage.
- 1337 tests passed on the VM in 48.82s; benchmark 20/20.

## Why this matters

The autonomous loop is no longer only a dry-run demonstration. It can train,
evaluate, gate and promote a narrow capability without human approval, and it
fails closed when the gate is not met. The runtime copy tool handles the 3-4
arbitrary key cases that the 49M model cannot learn reliably.

## Honest limits

- The promoted adapter is an echo-stage curriculum adapter, not a product
  assistant. The product default is still base `0.8.0`.
- The copy tool is deterministic tool use, not model self-improvement or ASI.
- The model's 3-4 key copying remains 0-6% exact.
- No model weights changed.

## Install

```powershell
pip install atomzero==0.40.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.40.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
