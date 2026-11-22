# atomzero 0.4.1 - scoped regression and code-aware retrieval

Status: **published to PyPI as `atomzero 0.4.1` on 2026-11-22.** No model or runtime
weights changed; the weights remain `0.8.0`.

## What changed

- Added `benchmarks/scoped_regression.py`, a deterministic product regression
  gate for the document workflow. It checks:
  - source-scoped answers do not leak other documents;
  - source-prefix and kind filters stay inside their scope;
  - single-source summaries keep page/section locators;
  - collection summaries include only the selected prefix;
  - no-answer questions return nothing;
  - exact code identifiers (`L-7781` vs `W-7781`) rank the right source;
  - missing source/collection summaries raise `SourceNotFound`.
- Latest scoped benchmark result: **10/10 checks pass**.
- Added exact code/ID awareness to document search:
  - code tokens such as `L-7781`, `VB_42` and `42-A` are preserved as query
    terms;
  - matching codes get a rerank boost instead of competing only on the shared
    numeric token;
  - answer snippets prefer sentences containing the exact code.
- Added regression tests for the scoped workflow and identifier search.
- 248 tests pass.

## Honest scope

- This release changes retrieval ordering only for code-like identifiers; it
  does not add factual knowledge or change weights.
- The public SQuAD retrieval result is unchanged: hybrid top-1 context 0.975,
  top-1 answer in visible snippet 0.816.
- The scoped benchmark is deterministic and authored; it is a regression gate,
  not a public leaderboard.

## Verification

- 248 tests pass.
- Wheel and sdist build cleanly; `twine check` passes.
- `benchmarks/scoped_regression.py`: 10 passed, 0 failed.
- SQuAD 50-context run remains at the 0.4.0 level.

## Install

```powershell
pip install atomzero==0.4.1
atomzero-doctor
atomzero-quickstart
```

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.4.1/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.4.1>
- Scoped benchmark: <https://github.com/byjameson/atom-zero/blob/main/benchmarks/scoped_regression.py>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
