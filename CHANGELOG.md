# Changelog

## Unreleased - next batch

Batched release train: accumulate about 10 development increments before
publishing to PyPI/GitHub releases/Hugging Face/ModelScope/dev.to. Intermediate
work stays on `main`; see `docs/RELEASING.md`.

- Added a local **second brain graph**: capitalized-phrase/code/year entity
  extraction, chunk co-occurrence edges, source counts and evidence lookup.
- Added `atomzero-docs graph` / `entity`, `GET /v1/graph` and
  `GET /v1/graph/entity`, plus a UI graph panel with entity chips, edges and
  source/snippet cards.
- Chat now answers relation queries such as “what is related to Ankara?” with
  graph evidence and connections.
- Added graph regression tests and `docs/KNOWLEDGE_GRAPH.md`.
- Added declarative **custom skills**: JSON trigger phrases, local-context
  prompts, provider workflows and deterministic templates.
- Added `atomzero-skills list/validate/show`, `GET /v1/skills`, a UI skills
  panel, three example skills and `docs/SKILLS.md`.
- Added skill regression tests for loading, matching, context rendering,
  provider/template modes and the CLI.

## atomzero 0.4.2 - UI collection scope (2026-11-22)

- Added a collection-prefix input and **Summarize collection** button to the
  document UI. Search and Answer accept the same prefix scope.
- Exact source and prefix scopes are mutually exclusive in the UI.
- Added a UI regression test for the collection-scope contract; 249 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.4.2.md`](docs/RELEASE_ATOMZERO_0.4.2.md).

## atomzero 0.4.1 - scoped regression and code-aware retrieval (2026-11-22)

- Added `benchmarks/scoped_regression.py`, a deterministic regression gate for
  source/prefix/kind-scoped answers, summaries, locators, no-answer behavior
  and exact code identifiers. Latest result: 10/10 checks pass.
- Added exact code/ID awareness to FTS candidates, reranking and snippets:
  `L-7781` and `W-7781` now rank their matching source instead of competing on
  the shared numeric token.
- Added tests for the benchmark and identifier-aware search; 248 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.4.1.md`](docs/RELEASE_ATOMZERO_0.4.1.md).

## atomzero 0.4.0 - hungry mode and install diagnostics (2026-11-22)

- Added **hungry mode**: `atomzero-hungry`, `POST /v1/hungry/run` and a UI
  section. It generates bounded questions, checks local knowledge, asks the
  configured teacher only for unknowns, stores unverified notes and writes an
  audit ledger. It never fetches web pages and never changes weights.
- Added `atomzero-quickstart --teacher`, gap states `known`/`learned` and a
  dry-run/provider-call budget for hungry mode.
- Added `atomzero-docs summary-collection` and collection mode on
  `GET /v1/documents/summary`; every selected sentence keeps its source,
  page and section.
- `atomzero-doctor` now validates an existing document index (chunk/source
  counts, orphan rows and FTS5 integrity) in addition to file write access.
- Added doctor tests for absent, clean and orphaned indexes.
- 246 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.4.0.md`](docs/RELEASE_ATOMZERO_0.4.0.md).

## atomzero 0.3.0 - document workflow and grounded answers (2026-11-22)

- Added `atomzero-docs`, a headless CLI for add/sync/status/doctor/search/
  answer/summary/evidence/export/import/list/forget.
- Added recursive directory ingestion with per-file reports,
  `--skip-unchanged` and `--dedupe` content-hash deduplication.
- Added `atomzero-docs sync`: portable relative-path collections, unchanged
  file skips, duplicate detection and pruning of deleted files.
- Added `atomzero-docs status` (dry-run directory diff) and
  `atomzero-docs doctor` (orphan-row and FTS integrity check).
- Added extractive document summaries (`atomzero-docs summary`,
  `GET /v1/documents/summary` and a UI Summary action) with page/section
  citations; no model generation and no new facts.
- Chat replies now use the shared grounded-answer core and expose
  `evidence`/`confidence`; the UI shows the match label.
- Added JSONL index export/import with titles, teacher/provenance metadata and
  page/section locators preserved.
- Added source, source-prefix and kind filters to document search.
- Added `POST /v1/documents/answer` and `atomzero-docs answer`: cited answer,
  evidence list and a transparent heuristic match-confidence score from one
  shared core.
- Improved lexical reranking with a small morphology allowance
  (`claim`/`claims`, `submit`/`submitted`).
- Answer snippets now prefer answer-bearing sentences with numeric/unit cues,
  add only adjacent context, and drop notes far below the winning score.
- Added a "Grounded answer" control, an indexed-source picker, per-document
  "Ask", and "Copy citation" actions to the local document UI.
- Added answer-in-visible-snippet and confidence metrics to the public SQuAD
  retrieval benchmark.
- Added tests for all of the above; 233 pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.3.0.md`](docs/RELEASE_ATOMZERO_0.3.0.md).

## atomzero 0.2.0 - public benchmark and demo UI (2026-11-22)

- Added a public SQuAD v2 retrieval benchmark (`benchmarks/squad_retrieval.py`)
  with 20x100 and 50x200 results.
- Added one-click RAG demo chips to the UI: launch code, warranty, visitor
  badge and a no-source Paris question.
- Kept the 0.1.8 doctor, 0.1.9 reset-demo/version flags/troubleshooting and
  donation addresses.
- No model/runtime weights changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.2.0.md`](docs/RELEASE_ATOMZERO_0.2.0.md).

## atomzero 0.1.9 - support and onboarding polish (2026-11-22)

- Added EVM, Solana and Bitcoin donation addresses to the README, PyPI
  description, HF/ModelScope model card, project overview and articles.
- Added `atomzero-quickstart --reset-demo`.
- Added `atomzero --version` and `atomzero-quickstart --version`.
- Added `docs/TROUBLESHOOTING.md`.
- No model/runtime behavior changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.9.md`](docs/RELEASE_ATOMZERO_0.1.9.md).

## atomzero 0.1.8 - install doctor (2026-11-22)

- Added `atomzero-doctor`: Python/package/dependency/torch-CUDA/cache/port/
  write-access/Ollama preflight checks with actionable hints and `--json`.
- Documented the doctor in the README.
- No model/runtime behavior changed; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.8.md`](docs/RELEASE_ATOMZERO_0.1.8.md).

## atomzero 0.1.7 - PyPI docs link cleanup (2026-11-22)

- Republished the package description with all documentation links pointing to
  the `atom-zero` repository.
- Removed the old `stream-finetune` documentation links from the PyPI page.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.7.md`](docs/RELEASE_ATOMZERO_0.1.7.md).

## atomzero 0.1.6 - LinkedIn and ModelScope restored (2026-11-22)

- Restored LinkedIn and ModelScope as canonical Atom Zero release surfaces.
- Added both links back to the package README, PyPI metadata, HF model card and
  project overview.
- Added a job-search-oriented LinkedIn post for Atom Zero 0.1.6.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.6.md`](docs/RELEASE_ATOMZERO_0.1.6.md).

## atomzero 0.1.5 - canonical surfaces cleanup (2026-11-22)

- Removed the remaining ModelScope link from the published package description
  and the LinkedIn project URL from package metadata.
- Documented the five canonical Atom Zero surfaces: GitHub, Hugging Face, PyPI,
  dev.to and Medium.
- Superseded by `0.1.6`: LinkedIn and ModelScope are canonical surfaces again.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.5.md`](docs/RELEASE_ATOMZERO_0.1.5.md).

## atomzero 0.1.4 - standalone repository (2026-11-22)

- Moved the Atom Zero package to its own GitHub repository:
  <https://github.com/byjameson/atom-zero>.
- PyPI project URLs now point to the standalone repository.
- Current Atom Zero links target the `atom-zero` repository, `atomzero` PyPI,
  Hugging Face and the two launch articles. ModelScope/LinkedIn were temporarily
  removed in 0.1.4 and restored as canonical surfaces in 0.1.6.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.4.md`](docs/RELEASE_ATOMZERO_0.1.4.md).

## atomzero 0.1.3 - full package description (2026-11-22)

- Expanded the standalone GitHub/PyPI README to cover the full system: model,
  training, RAG, documents/citations, evidence, tools, research, curiosity,
  learning ladder, provider, safety, privacy, benchmarks and limits.
- Removed stale streamft links from the Atom Zero model card; it now links to
  the atomzero package, source and `atomzero-v0.1.3` release.
- No model/runtime changes; weights remain `0.8.0`.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.3.md`](docs/RELEASE_ATOMZERO_0.1.3.md).

## atomzero 0.1.2 - standalone quickstart (2026-11-22)

- Added `atomzero-quickstart`: downloads the public checkpoint/tokenizer from
  Hugging Face, seeds a demo corpus and starts the local UI.
- No streamft checkout, trainer install or `git clone` is required.
- Added `--provider`, `--safety-filter`, `--port`, `--no-download` and
  `--no-browser` options.
- Release notes: [`docs/RELEASE_ATOMZERO_0.1.2.md`](docs/RELEASE_ATOMZERO_0.1.2.md).
