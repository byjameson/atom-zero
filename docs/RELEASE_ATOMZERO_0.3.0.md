# atomzero 0.3.0 - document workflow and grounded answers

Status: **published to PyPI as `atomzero 0.3.0` on 2026-11-22.** No model or
runtime weights changed; the weights remain `0.8.0`.

## What changed

- Added `atomzero-docs`, a headless CLI:
  - `add` a file or recursive directory;
  - `sync` a portable collection with relative paths, unchanged-file skips,
    duplicate detection and pruning;
  - `status` for a dry-run directory diff;
  - `doctor` for orphan-row and FTS5 integrity checks;
  - `search` with source, source-prefix and kind filters;
  - `answer` with cited evidence and a match-confidence object;
  - `summary` for extractive, citation-preserving summaries;
  - `evidence`, `export`, `import`, `list`, `stats` and `forget`.
- Added source-scoped search and answers to the HTTP API:
  - `POST /v1/documents/search`;
  - `POST /v1/documents/answer`;
  - `GET /v1/documents/summary`.
- Added a shared grounded-answer core used by the API, the CLI and chat.
  Answers now include per-hit evidence and a transparent confidence heuristic.
- Added a relative-relevance filter so a strong hit cannot drag unrelated
  documents into the answer.
- Added morphology-aware lexical reranking (`claim`/`claims`,
  `submit`/`submitted`).
- Improved answer snippets with numeric/unit cues and adjacent context.
- Added JSONL export/import preserving titles, teacher metadata, page/section
  locators and blocks.
- Added UI controls: source picker, Answer, Summary, Ask, and Copy citation.
- Added SQuAD v2 answer-in-visible-snippet and confidence metrics.
- 233 tests pass.

## Honest scope

- The confidence score is a **match-strength heuristic**, not a calibrated
  probability and not an answerability classifier. On the 50-context SQuAD
  subset it did not separate answerable from no-answer questions (0.585 vs
  0.583 mean).
- SQuAD results measure **retrieval**, not generation. Questions share
  vocabulary with their contexts; this is a public sanity check.
- Summaries are extractive. They select existing sentences and do not generate
  new facts.
- No model or runtime weights changed.

## Install

```powershell
pip install atomzero==0.3.0
atomzero-doctor
atomzero-quickstart
```

Headless document workflow:

```powershell
atomzero-docs sync .\notes --db data/docs.db
atomzero-docs answer "what is the launch code?" --db data/docs.db
atomzero-docs summary manual.pdf --sentences 5 --db data/docs.db
```

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.3.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.3.0>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
