# Changelog

## Unreleased - next batch

- Added multi-hop **graph paths**: `atomzero-docs path`, `GET /v1/graph/path`,
  chat “how is X connected to Y?” and UI Find path.
- Added `benchmarks/graph_path_regression.py` (6/6 checks).
- Added a **weekly review**: open/recently completed tasks, pending gaps,
  sources indexed within the window and graph highlights.
- Added `atomzero-brief --weekly --days N`, `POST /v1/weekly-review` and a UI
  Weekly review button.
- Added task `created`/`completed` timestamps with an undated-completion
  fallback for older sessions.
- Added `benchmarks/weekly_review_regression.py` (6/6 checks).
- Added chat intents for the weekly review (`weekly review`,
  `haftalık özet`, optional day window) with the full review payload.
- Weekly review text now says “completed in the last N days” for non-default
  windows instead of always “this week”.
- Added `benchmarks/weekly_chat_regression.py` (6/6 checks).
- Added weekly-review **Markdown export**:
  `atomzero-brief --weekly --markdown`, `GET /v1/weekly-review/export` and a UI
  Download .md button, with task checkboxes, gaps, sources, graph and next
  actions.
- Added `atomzero-brief --weekly --markdown --out PATH` to write the report
  directly to a file and create parent directories.
- `--out` now writes via a sibling temp file and atomic replace, preserving an
  existing report when the write or replace fails.
- Added a local **weekly report archive**: each successful `--out` export is
  indexed with its period, task/source/graph counts, size and SHA-256.
- Added `atomzero-brief --reports [--json]`, `GET /v1/weekly-review/reports`,
  indexed report download and a UI **Recent reports** panel.
- Added `atomzero-brief --reports --prune`,
  `POST /v1/weekly-review/reports/prune` and a UI **Prune missing** action to
  remove index entries whose report files no longer exist.
- Added `atomzero-brief --reports --keep N`,
  `POST /v1/weekly-review/reports/keep` and a UI **Keep newest** control for
  atomic retention of the newest N reports.
- Added `--keep N --dry-run`, `dry_run: true` for the keep endpoint and a UI
  **Preview only** option that lists would-be-removed reports without changing
  the archive.
- Added `atomzero-brief --reports --delete ID`,
  `POST /v1/weekly-review/reports/delete` and a UI **Delete** action for one
  archive entry; the report file is left on disk.
- Added `atomzero-brief --reports --query TEXT`,
  `GET /v1/weekly-review/reports?query=...` and a UI report search box.
- Added chat search for archived reports (`find report weekly`,
  `rapor ara weekly`) with a `search_weekly_reports` tool call, report payload
  and a UI quick prompt.
- Added chat maintenance previews for `prune report archive`,
  `keep reports N`, `keep reports N dry-run` and `delete report ID`; destructive
  actions require `confirm reports`, cancellation uses `cancel reports`.
- Added queryless chat archive listing (`list reports`,
  `show report archive 5`) with an optional recent-entry limit.
- Added archive ordering: `--reports --sort newest|oldest|largest`,
  `GET /v1/weekly-review/reports?sort=...` and a UI sort selector.
- Added chat sort suffixes for archive list/search commands, such as
  `list reports oldest` and `find report weekly largest`.
- Added chat report viewing by id (`show report w0001`, `rapor göster w0001`)
  with metadata, Markdown content and a `show_weekly_report` tool call.
- Added archived-report integrity checks: `--reports --verify`,
  `GET /v1/weekly-review/reports/verify` and a UI **Verify files** action
  comparing stored size and SHA-256 with each file.
- Added chat integrity commands (`verify report archive`,
  `verify report w0001`) and a UI quick prompt.
- Added archived-report metadata repair: `--reports --repair`,
  `POST /v1/weekly-review/reports/repair` and a UI **Repair metadata** action
  with atomic index updates for changed files.
- Added confirmed chat repair (`repair report archive`, `rapor arşivini onar`)
  with a preview and `confirm reports` gate.
- Added single-entry chat repair (`repair report w0001`, `rapor w0001 onar`)
  with the same confirmation flow.
- Added report comparison: `--reports --compare LEFT RIGHT`,
  `GET /v1/weekly-review/reports/compare` and
  `compare reports w0001 w0002` showing task, gap, source, size and SHA-256
  deltas.
- Added comparison Markdown export: `--compare ... --markdown`,
  `GET /v1/weekly-review/reports/compare/export` and UI Download/Copy actions.
- Added task due dates (`due YYYY-MM-DD`, `today`, `tomorrow`, `in N days`)
  with overdue markers in `list tasks`, the daily brief, weekly review and
  Markdown export.
- Added an overdue-task view: `atomzero-brief --overdue`,
  `GET /v1/tasks/overdue` and a UI **Overdue tasks** button.
- Added a due-soon view: `atomzero-brief --due-soon --due-days N`,
  `GET /v1/tasks/due-soon?days=N` and a UI **Due soon** button.
- Integrated due-soon tasks into the daily brief, weekly review, weekly
  Markdown report and chat (`due soon`, `due soon in 14 days`).
- Added task rescheduling (`reschedule task ID due DATE`),
  `POST /v1/tasks/reschedule` and a UI reschedule form.
- Added confirmed task deletion (`delete task ID` → `confirm task delete`),
  `POST /v1/tasks/delete` with `"confirm": true`, and a UI delete form.
- Added task text search (`search tasks TEXT`, `find task TEXT` or
  `task search TEXT`), `GET /v1/tasks/search?q=TEXT`,
  `atomzero-brief --search-tasks TEXT` and a UI search field/button.
- Added task-search status/due filters: chat suffixes (`open`, `done`,
  `overdue`, `today`, `week`, `none`), HTTP `status`/`due` query parameters,
  `--search-status`/`--search-due` CLI flags and UI selectors.
- Added task priorities (`low`, `normal`, `high`), `priority high` on
  `add task`, `set task ID priority LEVEL`, priority-aware list/brief/review
  labels, `priority` search filters in chat/CLI/HTTP and a UI selector.
- Added `benchmarks/weekly_export_regression.py` (29/29 checks: CLI, HTTP, UI,
  the report archive, pruning, retention, dry-run preview, deletion, query
  filtering, chat search/list/maintenance, chat/HTTP sorting, report viewing,
  integrity verification, all/single-entry metadata repair, comparison and
  comparison Markdown export).
- Fixed BOM-sensitive session loading: `atomzero-brief` and `ChatHuman` now
  accept PowerShell-written UTF-8 session files with a BOM.
- Extended BOM-tolerant reads to feedback, knowledge-gap, research-queue,
  skill and recipe JSON files.
- Added deterministic **source comparison**: `atomzero-docs compare`,
  `GET /v1/documents/compare` and UI Compare sources, flagging numeric,
  identifier and negation conflicts with both snippets/locators.
- Added `benchmarks/contradiction_regression.py` (5/5 checks).
- Added a chat intent for source comparison
  (`compare sources A and B`, Turkish equivalent) with the same reasons,
  snippets and locators, plus `benchmarks/compare_chat_regression.py`
  (6/6 checks).

## atomzero 0.5.0 - second brain and skill workflows (2026-11-22)

First batched release under the ~10-increment policy. It includes the work
accumulated after 0.4.2.

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
- Added a deterministic **daily brief** (`atomzero-brief`, `POST /v1/briefing`)
  combining open tasks, pending gaps, index stats and graph entities.
- Added a cited **goal planner** (`POST /v1/goals/plan`) with suggested tasks;
  task creation requires explicit `confirm: true`.
- Added the Daily brief UI panel and `docs/BRIEFING.md`.
- Added chat goal planning: `plan goal ...`, `confirm plan` and `cancel plan`.
  Task creation requires the explicit confirmation message.
- Added chat `brief me` / `daily brief`, returning open tasks, pending gaps,
  index status and graph entities with source items.
- Added `benchmarks/briefing_regression.py` (9/9 checks) and its test.
- Added `benchmarks/goal_flow_regression.py` (6/6 checks), an end-to-end HTTP
  test for plan/no-task, exactly-once confirmation and cancel/no-mutation.
- Added `benchmarks/brief_me_flow_regression.py` (7/7 checks), an end-to-end
  HTTP chat test for tasks, gaps, index, graph and briefing sources.
- Added `benchmarks/skills_flow_regression.py` (9/9 checks), an end-to-end
  HTTP chat test for provider, template and no-provider skill modes.
- Added explicit skill execution by name/query: `atomzero-skills run`,
  `POST /v1/skills/run` and UI Run skill controls, sharing `execute_skill`.
- Added `benchmarks/skills_run_regression.py` (9/9 checks) for explicit
  provider, template, fallback and unknown-skill paths.
- Added multi-skill chains: `atomzero-skills chain`, `POST /v1/skills/chain`
  and a UI Run chain control, with explicit `{previous}` context passing,
  no-carry mode and merged sources.
- Added `benchmarks/skills_chain_regression.py` (6/6 checks).
- Added `benchmarks/skills_chain_ui_regression.py` (6/6 checks): UI contract +
  the exact chain request through the real handler.
- Added named chain recipes under `data/skills/chains`: `atomzero-skills
  recipes` / `recipe`, `GET /v1/skills/chains`,
  `POST /v1/skills/chains/run` and a UI recipe select/run control.
- Added `benchmarks/skills_recipe_regression.py` (6/6 checks).
- Added `benchmarks/skills_recipe_ui_regression.py` (7/7 checks): recipe UI
  contract + exact request, carry override and unknown-recipe path.
- 286 tests pass.
- Release notes:
  [`docs/RELEASE_ATOMZERO_0.5.0.md`](docs/RELEASE_ATOMZERO_0.5.0.md).

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
