# Documents and citations

Atom Zero keeps facts in a local index, not in the 49.2M model. A retrieved
chunk carries its source, title, SHA-256, and — when the format provides it —
a page number or section heading.

## Supported formats

| format | extraction | citation locator |
|---|---|---|
| `.txt` | UTF-8 text | source |
| `.md` / `.markdown` | heading-aware blocks | source + section |
| `.html` / `.htm` | block-level text and headings | source + section |
| `.pdf` | one block per page via `pypdf` | source + page |
| `.docx` | paragraphs and headings via `python-docx` | source + section |

PDF pages are 1-based. DOCX/HTML/Markdown sections come from heading styles
(`Heading 1`, `Heading 2`, ...) or `<h1>`-`<h6>` tags. Plain text has no page
or section.

## What is stored

For every chunk the index keeps:

- the chunk text and source path/URL;
- the document title and SHA-256;
- `page` (PDF) or `section` (DOCX/HTML/Markdown);
- `char_start` / `char_end` inside that block.

The API and UI show the locator next to the citation:

```text
The launch code is L-7781. (source: manual.pdf (p. 4))
The warranty period is 24 months. (source: policy.docx [Warranty])
```

## Hashes

Two different hashes are now explicit:

| field | what it hashes |
|---|---|
| `file_sha256` | the exact uploaded bytes (PDF/DOCX/HTML file) |
| `content_sha256` | the extracted, normalized text that is chunked and stored |
| `sha256` | legacy alias: file hash for uploads, text hash for JSON text |

`doc_meta`/the FTS `sha` column stores the content hash; `sources()` exposes it
as `content_sha256`. For a plain text upload with no BOM or trailing whitespace
the two hashes can be equal; for a PDF they normally differ.

## Evidence panel

The web UI has an **Evidence** button on each indexed document and each search
hit. It calls the local endpoint:

```text
GET /v1/documents/evidence?source=manual.pdf&page=4&limit=20
GET /v1/documents/evidence?source=policy.docx&section=Warranty
```

The response contains the stored chunk text, page, section, character offsets
and title. `page` and `section` are optional filters; `source` is required.
This is a read-only view of the local index, not a second model call.

## Add a document

### Web UI

1. Open the local Atom Zero page.
2. Under **Add a document**, choose a file.
3. Click **Add document**.

Text/Markdown/HTML files are also editable in the textarea before indexing.
PDF/DOCX files are uploaded as bytes; they are never converted to text in the
browser.

### CLI

```powershell
atomzero-docs add manual.pdf --db data/docs.db
atomzero-docs add .\notes --db data/docs.db --skip-unchanged
atomzero-docs search "warranty code" --db data/docs.db
atomzero-docs search "warranty" --db data/docs.db --source policy.docx
atomzero-docs evidence policy.docx --section Warranty --db data/docs.db
atomzero-docs list --db data/docs.db
atomzero-docs forget policy.docx --db data/docs.db
```

`add` accepts a file or a directory. Directory mode is recursive by default,
uses the supported suffix list above, and reports per-file chunks, hashes,
skips and failures. With `--skip-unchanged`, a file whose extracted content
hash matches the stored hash is not deleted or re-embedded. With `--dedupe`,
a file whose extracted content hash already exists under another source is
skipped and reported as `duplicate_of` instead of being indexed twice.

`search` accepts `--source` (exact), `--source-prefix`, `--kind` (`document`
or `teacher`) and `--no-rerank`. It prints JSON, so it can be piped into other
local tools. `python -m atomzero.doc_memory ...` is the same CLI.

### HTTP

```text
POST /v1/documents/upload?source=manual.pdf&title=Manual
Content-Type: application/octet-stream

<raw PDF bytes>
```

Or paste text as JSON:

```text
POST /v1/documents
{"source": "notes.md", "title": "Notes", "text": "# Warranty\n..."}
```

`POST /v1/documents/search` also accepts an optional exact `source` field:

```text
{"query": "warranty code", "k": 5, "source": "policy.docx"}
```

### Grounded answer API

```text
POST /v1/documents/answer
{"query": "what is the warranty period?", "source": "policy.docx"}
```

The response contains the cited note text, the source map, per-hit evidence and
a match-confidence object:

```json
{
  "text": "I found these notes: ...",
  "sources": [{"source": "policy.docx", "title": "Warranty"}],
  "evidence": [{"source": "policy.docx", "snippet": "...", "score": 0.71}],
  "confidence": {"score": 0.62, "label": "medium",
                 "coverage": 0.75, "margin": 0.12}
}
```

The confidence score is a transparent heuristic over the rerank score, query
term coverage and the runner-up margin. It is **not** a calibrated probability
and it does not replace reading the cited source. When nothing matches, the
endpoint returns the no-source answer with an empty source list rather than
inventing a fact.

Re-adding the same `source` replaces its chunks, page/section metadata and
cached vectors. Directory sync with `--skip-unchanged` avoids that work when
the extracted content hash is identical.

## Limits

- Maximum raw upload: **10 MB** per document. Raw uploads use their own limit;
  JSON text requests still use the smaller 2 MB request-body cap.
- No OCR. A scanned/image-only PDF returns a clear “no extractable text” error;
  run OCR separately and add the resulting text if needed.
- Encrypted PDFs are rejected unless they can be opened with an empty password.
- DOCX page numbers are not available because pagination is a renderer
  decision; the section heading is the locator.
- Tables are extracted as row text; complex layouts may lose structure.
- The extractor never fetches URLs. URL ingestion remains the separate,
  opt-in research path with the existing SSRF checks.

## Regression benchmark

`research/benchmark_documents.py` generates a 10-document corpus across these
formats and runs 80 answerable plus 20 no-answer questions:

```powershell
python research/benchmark_documents.py `
  --out research/results/document_benchmark.json --check
```

It checks top-1 source, page/section locator, answer hit, no-answer refusal and
latency. The corpus is authored; see [`BENCHMARKS.md`](BENCHMARKS.md) for the
current result and caveats.

## Privacy

Extraction happens locally. The browser uploads bytes to the localhost server;
the server writes the index to the SQLite database you configured. There is no
telemetry and no third-party conversion service. **Forget source** removes the
chunks, their page/section metadata, metadata row and cached vectors, then
compacts the database.
