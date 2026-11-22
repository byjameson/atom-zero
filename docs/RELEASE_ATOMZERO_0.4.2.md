# atomzero 0.4.2 - UI collection scope

Status: **published to PyPI, GitHub, Hugging Face and ModelScope as `atomzero
0.4.2` on 2026-11-22.** dev.to intentionally remains at `0.4.1` under the
batched release cadence and will sync at the next batch boundary. No model or
runtime weights changed; the weights remain `0.8.0`.

## What changed

- Added a **collection prefix** input to the document UI. A user can type
  `notes/` instead of selecting one exact source.
- Added **Summarize collection**: it calls the extractive collection summary
  endpoint and shows the selected prefix, source/chunk counts and a locator for
  each contributing source.
- Search and **Answer** now use the same scope:
  - an exact source wins when selected;
  - otherwise the collection prefix is sent as `source_prefix`;
  - choosing a source clears the prefix and typing a prefix clears the source
    selector, so the two scopes cannot conflict.
- Added a UI regression test for the collection-scope contract.
- 249 tests pass.

## Honest scope

- This is a UI/scope improvement; it does not change retrieval scoring, model
  weights or teacher behavior.
- Collection summaries remain extractive: they select existing sentences and
  preserve source, page and section; they do not generate new facts.
- The collection prefix is a string prefix, not a filesystem operation. It only
  scopes the local index.

## Verification

- 249 tests pass.
- Wheel and sdist build cleanly; `twine check` passes.
- Browser E2E: `notes/` collection summary returned 2 sources / 2 chunks and
  listed `notes/launch.txt` and `notes/warranty.txt`; prefix-scoped Answer
  returned the warranty note with confidence `high` and `prefix: notes/`.

## Install

```powershell
pip install atomzero==0.4.2
atomzero-doctor
atomzero-quickstart
```

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.4.2/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.4.2>
- Documents guide: <https://github.com/byjameson/atom-zero/blob/main/docs/DOCUMENTS.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
