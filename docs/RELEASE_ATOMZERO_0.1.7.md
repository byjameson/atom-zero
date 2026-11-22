# atomzero 0.1.7 - PyPI docs link cleanup

Status: **published to PyPI as `atomzero 0.1.7` on 2026-11-22.**

Metadata-only release. No model or runtime behavior changed; the weights remain
`0.8.0`.

## What changed

- The PyPI description now points every documentation link to the standalone
  `atom-zero` repository.
- The old `stream-finetune` links for the project map, documents, tool plans and
  benchmarks were removed from the rendered PyPI page.
- The package README, model card, quickstart and overview were bumped to 0.1.7.

## Install

```powershell
pip install atomzero==0.1.7
atomzero-quickstart
```

The first run downloads the ~197 MB checkpoint/tokenizer from Hugging Face,
seeds a demo corpus and opens the local UI. No streamft checkout is required.

## Links

- PyPI: <https://pypi.org/project/atomzero/0.1.7/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.1.7>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
