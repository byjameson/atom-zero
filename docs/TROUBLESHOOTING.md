# Troubleshooting Atom Zero

Start with the doctor:

```powershell
atomzero-doctor
atomzero-doctor --provider --json
```

`FAIL` means a required check is broken. `WARN` is optional (for example, no
Ollama provider). The JSON form is useful when opening an issue.

## Model download fails

The quickstart downloads `model.pt`, `config.json` and the tokenizer to
`~/.cache/atomzero/atom-zero-49m`.

1. Check free disk space (`atomzero-doctor` reports it).
2. Delete any `*.part` file and run `atomzero-quickstart` again.
3. Check that `huggingface.co` is reachable from your network/proxy.
4. Use an explicit cache path:

```powershell
atomzero-quickstart --cache D:\atomzero\atom-zero-49m
```

## Port is busy

The default port is `8013`.

```powershell
atomzero-quickstart --port 8014
```

The doctor checks the port before the server starts.

## Torch or CUDA is missing

CPU works, but generation is slower. For CUDA, install the wheel that matches
your driver from <https://pytorch.org/get-started/locally/>. Run
`atomzero-doctor` again; the GPU name and CUDA version should appear.

## PDF/DOCX dependencies are missing

```powershell
pip install "pypdf>=4" "python-docx>=1.1"
```

Encrypted PDFs are rejected. Scanned PDFs need OCR before ingestion; Atom Zero
does not ship an OCR engine.

## Ollama/provider is not reachable

```powershell
ollama serve
ollama pull llama3.2:3b
atomzero-doctor --provider
atomzero-quickstart --provider
```

The provider is optional. RAG, citations, tools and memory stay local without
it.

## The demo corpus has duplicates

```powershell
atomzero-quickstart --reset-demo
```

This deletes the demo SQLite DB and seeds the four demo documents again. It does
not touch your own documents if you use a different `--docs` path.

## JSON files written by PowerShell

Local JSON stores are read as UTF-8 with or without a BOM. Files written with
`Set-Content -Encoding utf8` work for sessions, feedback, knowledge gaps,
research queues, skills and chain recipes.

## Where to inspect failures

- `GET /health` on the running server.
- `GET /metrics`.
- The **Evidence** panel in the UI for retrieval/citation debugging.
- `atomzero-doctor --json` for environment details.

If you open an issue, include the doctor JSON, the command you ran and the
smallest failing question with its expected source.
