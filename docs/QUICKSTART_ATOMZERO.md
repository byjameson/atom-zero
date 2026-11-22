# Atom Zero quickstart

## Requirements

- Python 3.10+
- ~197 MB free disk for the checkpoint/tokenizer
- Optional: Ollama + `llama3.2:3b` for the teacher/curiosity flow and provider mode
- Optional: `models/all-MiniLM-L6-v2` for neural retrieval
- `pypdf` and `python-docx` are installed with the package for PDF/DOCX ingestion

## Run

```powershell
pip install atomzero==0.3.0
atomzero-quickstart
```

The first run:

1. downloads the checkpoint/tokenizer from Hugging Face;
2. seeds `data/atomzero_demo.db` with launch, warranty, travel and visitor docs;
3. starts the local server on `http://127.0.0.1:8013`;
4. opens the browser.

No streamft checkout or trainer install is needed.

## Check the install

```powershell
atomzero-doctor
atomzero-doctor --provider --json
```

The doctor checks Python, dependencies, torch/CUDA, the exact model-cache sizes,
document/session write access, the local port and the optional Ollama provider.

## Try these

```text
What is the launch code?                          -> cited to Launch policy
How long do I have to submit a warranty claim?   -> cited to Warranty
What class should I book for a four-hour flight? -> cited to Travel
What is the visitor badge code?                  -> cited to Visitor security
What is the population of Paris?                 -> no local source; gap recorded
Propose curiosity topics                         -> pending research; approval required
Add task buy milk
My name is Alex
What is my name?
```

The first four exercise RAG. The Paris question exercises the honest
missing-source path. **Propose curiosity topics** creates pending research
items; approve one in the research queue before the worker fetches anything.

## Add your own documents

Under **Add a document**, choose a file and click **Add document**. Supported
formats are `.txt`, `.md`, `.markdown`, `.html`, `.htm`, `.pdf` and `.docx`
(max 10 MB). PDF citations keep the page number; DOCX/HTML/Markdown citations
keep the section heading. Scanned PDFs need OCR before ingestion.

The full format and provenance rules are in [`DOCUMENTS.md`](DOCUMENTS.md).

## Headless document workflow

The same index works without the browser:

```powershell
atomzero-docs sync .\notes --db data/docs.db
atomzero-docs answer "what is the launch code?" --db data/docs.db
atomzero-docs summary manual.pdf --sentences 5 --db data/docs.db
atomzero-docs summary-collection --prefix notes/ --sentences 8 `
  --db data/docs.db
atomzero-docs doctor --db data/docs.db
```

`sync` keeps a portable relative-path collection, skips unchanged/duplicate
files and prunes deleted ones. `status` is a dry run; `doctor` checks orphan
rows and FTS5 integrity; `export`/`import` move the index as JSONL. See
[`DOCUMENTS.md`](DOCUMENTS.md) for every command.

## Hungry mode (optional)

Start the local teacher and the UI together:

```powershell
atomzero-quickstart --teacher
```

Then run a bounded curiosity pass:

```powershell
atomzero-hungry --docs data/atomzero_demo.db --topic Ankara --max-questions 5
atomzero-hungry --docs data/atomzero_demo.db --dry-run --json
```

It checks local notes first, asks Ollama only for unknown questions, stores
answers as **unverified** teacher notes and writes an audit ledger. It never
fetches web pages and never changes weights. See
[`HUNGRY_MODE.md`](HUNGRY_MODE.md) for budgets and the exact boundaries.

## Provider mode

```powershell
atomzero-quickstart --provider
```

This routes free-form chat to local Ollama `llama3.2:3b` while RAG/tools stay
local. Start Ollama first. Use a different model with `--provider-model`.

## Reset the demo

```powershell
atomzero-quickstart --reset-demo
```

## Troubleshooting

- **Download failed** — retry `atomzero-quickstart`; cached files live under
  `~/.cache/atomzero/atom-zero-49m`.
- **Port busy** — pass `--port 8014`.
- **No neural retrieval** — the server falls back to hashed FTS-anchored
  retrieval; download `models/all-MiniLM-L6-v2` for semantic recall.
- **No Ollama** — chat still works; curiosity/teacher calls will report the
  provider error. Start Ollama and pull `llama3.2:3b`.
- **Safety** – off by default. Add `--safety-filter` only if you want the
  optional rule+ML filter.

## Support

- EVM-compatible networks:
  `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- Solana: `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- Bitcoin: `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>
