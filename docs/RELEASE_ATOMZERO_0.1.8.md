# atomzero 0.1.8 - install doctor

Status: **published to PyPI as `atomzero 0.1.8` on 2026-11-22.**

Onboarding release. No model or runtime behavior changed; the weights remain
`0.8.0`.

## What changed

- Added the `atomzero-doctor` console command.
- Checks Python, package version, tokenizers/numpy/pypdf/python-docx,
  torch/CUDA/GPU, exact model-cache sizes, document/session write access, the
  local port and the optional Ollama provider.
- Prints actionable hints; `--json` emits machine-readable diagnostics.
- Exits non-zero only on hard failures; optional provider issues are warnings.

## Check your install

```powershell
pip install atomzero==0.1.8
atomzero-doctor
atomzero-doctor --provider --json
```

## Install and quickstart

```powershell
atomzero-quickstart
```

The first run downloads the ~197 MB checkpoint/tokenizer, seeds a demo corpus
and opens the local UI. No streamft checkout is required.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.8/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.1.8>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
