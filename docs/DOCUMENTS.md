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

### Collection scope in the UI

The document bar accepts either one exact source or a collection prefix such as
`notes/`. Search, **Answer** and **Summarize collection** use the same scope.
Selecting an exact source clears the prefix; typing a prefix clears the source
selector, so the two scopes cannot conflict. A collection summary lists the
selected prefix, source/chunk counts and one locator per contributing source.

### Compare sources

The local index can compare one source against another source, or against all
other sources:

```powershell
atomzero-docs compare policy_a.txt --against policy_b.txt --db data/docs.db
atomzero-docs compare policy_a.txt --db data/docs.db
```

```text
GET /v1/documents/compare?source=policy_a.txt&against=policy_b.txt
```

The UI has **Compare sources** selects in the documents section. Chat also
understands the same workflow:

```text
compare sources policy_a.txt and policy_b.txt
policy_a.txt ile policy_b.txt kaynaklarını karşılaştır
```

Flags are deterministic hints, not a truth verdict:

- `numeric_mismatch` — overlapping claims have different numeric values;
  formatting differences (`1,500` / `1500` / `1.5k`, `1.5 million` /
  `1,500,000`, Turkish `1,5` / English `1.5`) and compatible units
  (`24 months` / `2 years`, `72 hours` / `3 days`) are normalized first;
- `date_mismatch` — overlapping claims give different full dates; equivalent
  formats (`5 January 2020`, `January 5, 2020`, `5 Ocak 2020`) are normalized;
- `identifier_mismatch` — overlapping claims use different code identifiers;
- `negation_mismatch` — one claim is negated and the other is not.

Claims are aligned lexically (shared terms/entities) or by topic: a shared
title/section plus a comparable measure, date or code. Each conflict includes
`alignment` (`lexical` or `topic`) and the shared topic terms.

Every conflict includes both snippets and page/section locators for review.

### Timeline

Extract dated events with provenance:

```powershell
atomzero-docs timeline --db data/docs.db
atomzero-docs timeline --source policy_a.txt --limit 20
atomzero-docs timeline --entity Ankara --db data/docs.db
```

```text
GET /v1/documents/timeline?source=policy_a.txt&limit=20
GET /v1/documents/timeline?entity=Ankara
```

Full dates (`5 January 2020`, `January 5, 2020`, `5 Ocak 2020`,
`2020-01-05`) become ISO dates. Year-only mentions keep `precision: year`.
Every event keeps source, title, page/section, kind, verified flag and extracted
entities; results are sorted by date. An entity filter is a case-insensitive
match against the event text and extracted entities. This is a textual
timeline, not inference.

Chat also understands `timeline` and `zaman çizelgesi`; the documents UI has a
**Show timeline** action with evidence and citation buttons.

### Typed relations

Extract explicit relation triples with provenance:

```powershell
atomzero-docs relations --entity Ankara --relation capital_of `
  --db data/docs.db
```

```text
GET /v1/graph/relations?entity=Ankara&relation=capital_of&limit=100
```

Supported relations are `located_in`, `capital_of`, `founded_by`, `ceo_of` and
`born_in`, with English and Turkish sentence patterns. Filters accept a
case-insensitive entity substring and an exact relation name. Co-occurrence
alone is not a relation; every triple keeps the sentence and source locators.

Find a multi-hop path through explicit relation edges:

```powershell
atomzero-docs relation-path "Jane Doe" Boston --max-depth 2 `
  --db data/docs.db
```

```text
GET /v1/graph/relations/path?source=Jane%20Doe&target=Boston&max_depth=2
```

Chat understands `relation path from Jane Doe to Boston` and
`Jane Doe ile Boston arasındaki ilişki yolu`. Each path edge records its
direction and source locator; the path is explicit extraction, not inference.

### CLI

```powershell
atomzero-docs add manual.pdf --db data/docs.db
atomzero-docs add .\notes --db data/docs.db --skip-unchanged
atomzero-docs sync .\notes --db data/docs.db
atomzero-docs status .\notes --db data/docs.db
atomzero-docs doctor --db data/docs.db
atomzero-docs search "warranty code" --db data/docs.db
atomzero-docs search "warranty" --db data/docs.db --source policy.docx
atomzero-docs answer "what is the warranty period?" --source policy.docx `
  --db data/docs.db
atomzero-docs summary manual.pdf --sentences 5 --db data/docs.db
atomzero-docs summary-collection --prefix notes/ --sentences 8 `
  --db data/docs.db
atomzero-docs graph --db data/docs.db
atomzero-docs entity Ankara --db data/docs.db
atomzero-docs evidence policy.docx --section Warranty --db data/docs.db
atomzero-docs export --out backup.jsonl --db data/docs.db
atomzero-docs import backup.jsonl --db data/docs.db
atomzero-docs list --db data/docs.db
atomzero-docs forget policy.docx --db data/docs.db
```

`add` accepts a file or a directory. Directory mode is recursive by default,
uses the supported suffix list above, and reports per-file chunks, hashes,
skips and failures. With `--skip-unchanged`, a file whose extracted content
hash matches the stored hash is not deleted or re-embedded. With `--dedupe`,
a file whose extracted content hash already exists under another source is
skipped and reported as `duplicate_of` instead of being indexed twice.

`sync` is the repository-style command for a maintained collection. It stores
sources as `<prefix>/<relative-path>` (prefix defaults to the directory name),
skips unchanged files and duplicate content, and prunes sources whose files
disappeared. Use `--no-prune`, `--no-dedupe` or `--prefix NAME` to change that
behavior. This keeps the index portable when the folder moves to another
machine.

`status` performs the same comparison without writing: it reports
`added`/`changed`/`unchanged` files and `missing` indexed sources. `doctor`
checks for orphan vectors, orphan chunk metadata, orphan source metadata,
sources without metadata, empty chunks and FTS5 corruption; it exits non-zero
when the index is not clean.

`search` and `answer` accept `--source` (exact), `--source-prefix` and
`--kind` (`document` or `teacher`); `search` also accepts `--no-rerank`.
`answer` runs the same grounded-answer path as the HTTP endpoint and returns
the note text, evidence and confidence. `summary` returns an **extractive**
summary of one source: it selects and orders existing sentences and keeps the
page/section locators; it does not generate new prose.
`summary-collection` does the same across a source prefix and/or kind, with a
locator on every selected sentence. `graph` builds the local co-occurrence
knowledge graph; `entity NAME` returns its evidence chunks and connections. See
[`KNOWLEDGE_GRAPH.md`](KNOWLEDGE_GRAPH.md). All commands print JSON, so they
can be piped into other local tools.
`python -m atomzero.doc_memory ...` is the same CLI.

`export` writes one JSONL record per source with its title, teacher/provenance
metadata and blocks (page/section preserved). `import` restores those records,
replacing the same sources by default; `--no-replace` appends instead. The
backup contains your documents and notes, so keep it private.

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

The same endpoint accepts `source_prefix` for a synced collection
(`"notes/"`) and `kind` (`"document"` or `"teacher"`). `POST
/v1/documents/search` accepts the same filters.

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

Each answer also passes a calibrated CitationGate before release. The default
threshold is `0.5` with the `squad_v2_stress` profile; set
`ATOMZERO_CITATION_THRESHOLD`, `ATOMZERO_CITATION_PROFILE`,
`ATOMZERO_CITATION_CALIBRATION` or `ATOMZERO_CITATION_GATE` to change it. When
no citation passes, the response has `refused: true`, `reason:
no_citation_above_threshold` and an empty source list. See
[`CITATION_GATE.md`](CITATION_GATE.md) for calibration and measurement details.

### Extractive summary API

```text
GET /v1/documents/summary?source=manual.pdf&sentences=5
GET /v1/documents/summary?source_prefix=notes/&kind=document&sentences=8
```

The response contains the selected sentences in source order, the combined
`text`, `method: "extractive"`, candidate/chunk counts and page/section
citations. Collection mode also returns each sentence's source and locator. It
scores existing sentences for content-word frequency, position and length; it
does not call a model or add facts.

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
