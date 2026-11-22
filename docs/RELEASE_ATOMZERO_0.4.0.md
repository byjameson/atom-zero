# atomzero 0.4.0 - hungry mode and install diagnostics

Status: **published to PyPI as `atomzero 0.4.0` on 2026-11-22.** No model or runtime
weights changed; the weights remain `0.8.0`.

## Headline: hungry mode

`atomzero-hungry` is a bounded, opt-in curiosity loop that grows the **RAG
index**, not the weights:

1. Collect questions from pending knowledge gaps, `--topic` seeds and indexed
   document titles.
2. Check the local index first. A strong local answer marks the gap `known`
   without a teacher call.
3. Ask the configured local/OpenAI-compatible teacher only for unknowns.
4. Store the answer as an **unverified teacher note** with provenance.
5. Mark the gap `learned` and append a JSONL audit record.
6. Stop at `--max-questions` and `--max-provider-calls`.

It never fetches web pages, never changes weights and does not run as a hidden
background daemon. Continuous operation is explicit scheduling of the bounded
command.

```powershell
ollama serve
ollama pull llama3.2:3b
atomzero-quickstart --teacher
atomzero-hungry --topic Ankara --max-questions 5
atomzero-hungry --dry-run --json
```

Also added:

- `POST /v1/hungry/run` and a UI **Hungry mode** section.
- `atomzero-quickstart --teacher`, `--teacher-url`, `--teacher-model` and
  `--teacher-label`.
- Knowledge-gap states `known` and `learned`.
- `docs/HUNGRY_MODE.md` with boundaries, budgets, ledger format and scheduling
  examples.

## Document workflow

- `atomzero-docs summary-collection --prefix notes/ --sentences 8` and
  collection mode on `GET /v1/documents/summary`.
- Every selected summary sentence keeps its source, page and section.
- `atomzero-doctor` now validates an existing document index: chunk/source
  counts, orphan vector/chunk-meta/meta rows, missing metadata, empty chunks and
  FTS5 integrity.

## Honest scope

- Hungry mode grows retrieval memory, not the 49.2M model weights.
- Teacher answers are unverified notes. They can be wrong or outdated; the
  existing review flow is still required before treating them as authoritative.
- The confidence gate can classify a weak local match as unknown.
- SQuAD results remain retrieval measurements, not generation scores.
- `weights_changed: false` is reported on every hungry-mode run.

## Verification

- 246 tests pass.
- Wheel and sdist build cleanly; `twine check` passes.
- Hungry mode was exercised end to end against a local OpenAI-compatible test
  teacher: one unknown question, one unverified teacher note, gap `learned`,
  ledger written, `weights_changed: false`.
- `atomzero-hungry` is included in the wheel entry points.

## Install

```powershell
pip install atomzero==0.4.0
atomzero-doctor
atomzero-quickstart
```

Headless and hungry workflows:

```powershell
atomzero-docs sync .\notes --db data/docs.db
atomzero-docs answer "what is the launch code?" --db data/docs.db
atomzero-docs summary-collection --prefix notes/ --db data/docs.db
atomzero-hungry --docs data/docs.db --max-questions 3
```

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.4.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.4.0>
- Hungry mode: <https://github.com/byjameson/atom-zero/blob/main/docs/HUNGRY_MODE.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
