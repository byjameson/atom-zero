# Atom Zero quickstart

## Requirements

- Python 3.10+
- ~197 MB free disk for the checkpoint/tokenizer
- Optional: Ollama + `llama3.2:3b` for the teacher/curiosity flow and provider mode
- Optional: `models/all-MiniLM-L6-v2` for neural retrieval
- `pypdf` and `python-docx` are installed with the package for PDF/DOCX ingestion

## Run

```powershell
pip install atomzero==0.2.0
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
