# atomzero 0.2.0 - public benchmark and demo UI

Status: **published to PyPI as `atomzero 0.2.0` on 2026-11-22.**

Product/benchmark release. No model or runtime weights changed; the weights
remain `0.8.0`.

## What changed

- Added a public SQuAD v2 retrieval benchmark:
  `benchmarks/squad_retrieval.py`.
  - 20 contexts / 100 questions: top-1 context 0.98, top-5 1.00.
  - 50 contexts / 200 questions: FTS 0.960 top-1, hybrid 0.970 top-1.
- Added one-click RAG demo chips to the local UI:
  - launch code
  - warranty
  - visitor badge
  - no-source Paris question
- Kept the 0.1.8 `atomzero-doctor`.
- Kept the 0.1.9 `--reset-demo`, `--version` flags, troubleshooting guide and
  donation addresses.

## Install

```powershell
pip install atomzero==0.2.0
atomzero-doctor
atomzero-quickstart
```

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.2.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.2.0>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
