# Atom Zero 49.2M — a local assistant that cites its sources

Atom Zero is a local, source-cited assistant built around a **49.2M parameter
model trained from scratch**. Facts live in a local RAG index, not in the model
weights; the model handles wording and behavior while deterministic tools,
documents, memory and citations do the rest.

This is the standalone **`atomzero`** package. The low-VRAM trainer is a separate
package and is **not required** to run Atom Zero. Atom Zero is published on seven
canonical surfaces: GitHub, Hugging Face, PyPI, dev.to, Medium, LinkedIn and
ModelScope.

## Install

```powershell
pip install atomzero==0.6.1
```

Optional local MiniLM reranker:

```powershell
pip install "atomzero[neural]"
```

## Quickstart

```powershell
atomzero-quickstart
```

The first run:

1. downloads the ~197 MB checkpoint/tokenizer from
   [Hugging Face](https://huggingface.co/byjameson/atom-zero-49m);
2. seeds a small demo corpus (launch code, warranty, travel, visitor badge);
3. starts the local UI at `http://127.0.0.1:8013`;
4. opens the browser.

Useful options:

```powershell
atomzero-quickstart --provider          # local Ollama free-form chat
atomzero-quickstart --port 8014
atomzero-quickstart --safety-filter
atomzero-quickstart --no-download       # use cached files
atomzero-quickstart --no-browser
atomzero-quickstart --reset-demo        # rebuild the demo document DB
```

The cache lives at `~/.cache/atomzero/atom-zero-49m`.

## Check your install

```powershell
atomzero-doctor
atomzero-doctor --provider --json
```

The doctor checks Python, the package version, torch/CUDA, the model cache
(exact file sizes), document/session write access, the document index
(orphan rows and FTS5 integrity), the local port and the optional Ollama
provider. It exits non-zero only on hard failures.

More fixes are in [`docs/TROUBLESHOOTING.md`](docs/TROUBLESHOOTING.md).

## What it is

- **A local, single-user daily assistant:** short chat, tasks with optional due
  dates, priorities and overdue/due-soon focus in brief/review/chat, task
  priority updates via chat/CLI/HTTP/UI, rescheduling, confirmed deletion,
  filtered task search (status/due/priority) and priority-first sorting in
  search/overdue/due-soon, daily-brief, weekly-review and chat commands
  (`brief me priority`, `weekly review priority`, `list tasks priority`,
  filtered `list tasks` status/due/priority/tag views with structured tool
  calls, task tags, `list tags` counts, confirmed `rename tag` /
  `delete tag` with undo and a filterable tag audit with CLI/HTTP/UI
  CSV/Markdown export, matching
  `/v1/tasks/list` + UI panels and CSV/Markdown export via `/v1/tasks/export`,
  tag-aware `search tasks report sort priority`, `overdue tasks tag work`,
  `due soon tag work`), notes, arithmetic, time, optional AISC architecture
  audit and source-cited document lookup.
- **A source-cited RAG system** over your own text, Markdown, HTML, PDF and DOCX
  files.
- **A deterministic tool runtime:** arithmetic, time, tasks, name/likes memory and
  a constrained plan API.
- **A reviewed research loop:** approved-only Wikipedia fetches, local teacher
  notes and a human approval gate.
- **A small-model + harness reference:** the harness is the reusable asset; the
  49.2M core is deliberately narrow.
- **Optionally, a front-end for a stronger local provider** while RAG, tools and
  memory stay local.

## What it is not

- Not a general reasoning or coding model.
- Not a ChatGPT competitor.
- Not a hosted, multi-tenant service.
- Not a model that stores all facts in its weights.
- Not an autonomous crawler or self-modifying agent.
- Not a claim that 49.2M parameters are enough for general intelligence.

## System features

| area | implemented |
|---|---|
| core model | 49.2M decoder-only transformer, BPE-8k, 6 layers, 768 dim, 128 context |
| training | 175.37M clean + 214.8M replay; 40k product checkpoint |
| RAG | SQLite FTS5, hashed/MiniLM embeddings, hybrid candidates, lexical+cosine rerank |
| vector index | cached vectors, prewarmed numpy matrix, IVF ANN above 256 chunks, exact fallback |
| documents | text/Markdown/HTML/PDF/DOCX ingestion, page/section citations, evidence panel/API |
| tools | time, safe arithmetic, tasks, name/likes memory, allowlisted 3-step plan API |
| sessions | transcript restore, JSON export/import, exact Forget, session Forget |
| research | approved-only queue, Wikipedia fetch, worker lock/log, teacher synthesis |
| teacher | local Ollama/OpenAI-compatible, unverified notes, reviewer verification |
| curiosity | gap → pending topic → approval → Wikipedia + teacher → RAG note |
| self-learning | RAG memory (no weights), reviewed LoRA (weights), manual full training |
| provider | optional local free-form chat; RAG/tools stay local |
| safety | optional rules + local linear ML head; off by default; exact fallback |
| privacy | localhost, no telemetry, SSRF guard, upload caps, deletion |
| UI | chat, documents, evidence, gaps, research queue, teacher notes, adapters, health |

## Model

| item | value |
|---|---|
| name | Atom Zero 49.2M (`transformer_behavior_50m_scaled`) |
| architecture | decoder-only transformer |
| parameters | 49,213,440 |
| layers / dim / heads / FF | 6 / 768 / 12 / 3072 |
| context | 128 tokens |
| tokenizer | 8,192-token BPE (`chat_en_bpe8k.json`) |
| checkpoint step | 40,000 |
| training tokens | 175.37M clean + replay mix to 214.8M |
| runtime | PyTorch, CUDA or CPU; RTX 3050 6 GB tested |

The model is trained from scratch. It is a **narrow chat/behavior model**, not a
foundation model. Facts are expected to live in RAG; skills can live in reviewed
LoRA adapters.

## Honest benchmark

| check | result |
|---|---:|
| raw 49.2M, no RAG/tools | **2/10** |
| product with RAG/tools | **21/21** |
| behavior checks | **16/16** |
| source-cited retrieval top-1 | **1.000** (FTS 0.850) |
| document benchmark, 100 authored questions | **80/80 answerable, 20/20 no-answer** |
| public SQuAD v2 retrieval (20 contexts) | **0.98 top-1 context (FTS), 1.00 top-5** |
| public SQuAD v2 retrieval (50 contexts) | **hybrid 0.975 top-1; 0.884 answer in visible snippet** |
| Turkish morphology retrieval (20 authored queries) | **hybrid 1.000 top-1; FTS 0.950** (before vowel drop 0.800 / 0.700) |
| document timeline regression | **10/10** |
| typed relations regression | **12/12** |
| citation gate, SQuAD v2 (default 0.5) | **92/103 answerable retained; 18/97 no-answer refused; precision 0.479** |
| test suite | **432 passed** |

The raw-model probe asks ten simple general questions with no documents, no
calculator and no agent. The 49.2M model gets **2/10**. The product gets
**21/21** because the runtime does the work.

The raw `pytest -v` run for the tested commit, together with the AISC suite
run, is recorded in [`TEST_RESULTS.md`](TEST_RESULTS.md).

The product benchmark is narrow and authored: tools, memory, tasks,
source-cited retrieval, teacher-note review, missing-source detection and one
short generation. It is **not MMLU, GSM8K or HumanEval**.

## Documents and citations

- **Text, Markdown, HTML, PDF and DOCX** ingestion.
- PDF citations keep the **page number**; DOCX/HTML/Markdown keep the
  **section heading**.
- **Evidence panel/API:** `GET /v1/documents/evidence` opens the exact local
  chunks behind a citation.
- **Grounded answers:** `POST /v1/documents/answer` returns cited notes,
  evidence and a transparent match-confidence score; chat replies use the same
  core and expose the match label.
- **Calibrated citation gate:** rerank scores are mapped to citation-validity
  probabilities; citations below the configurable threshold (default `0.5`)
  are dropped and an empty gate returns an explicit refusal. Threshold,
  profile and custom calibration are set with `ATOMZERO_CITATION_*`
  environment variables; see [`docs/CITATION_GATE.md`](docs/CITATION_GATE.md).
  `python -m atomzero.calibrate` fits a per-deployment isotonic map and
  recommends a threshold from a score/label CSV.
- **Source-scoped retrieval:** search and answers can be limited to one exact
  source, a source prefix or teacher notes.
- **Exact code/ID retrieval:** `L-7781` and `W-7781` are treated as distinct
  identifiers in FTS candidates, reranking and snippets.
- **Turkish morphology:** suffix, final-consonant and vowel-drop matching
  (`kitap` / `kitabın`, `ağaç` / `ağacın`, `ev` / `evler`, `şehir` / `şehrin`,
  `burun` / `burnu`) plus FTS prefix variants for Turkish queries; English
  matching is unchanged.
- **Collection scope in the UI:** choose an exact source or type a prefix such
  as `notes/`; search, Answer and Summarize collection share the same scope.
- **Second brain graph:** local co-occurrence graph over names, codes and years
  with source counts and one-click evidence; CLI `graph`/`entity`/`path`, HTTP
  and UI. Chat answers “what is related to X?” and multi-hop “how is X
  connected to Y?” with graph evidence.
- **Source comparison:** flag overlapping claims with different numbers, units,
  dates, code identifiers or negation, with lexical and topic-paraphrase
  alignment; CLI/API/UI and chat (`compare sources A and B`) keep both snippets
  and locators for review. It is a hint, not a truth verdict.
- **Document timeline:** `atomzero-docs timeline`,
  `GET /v1/documents/timeline`, chat `timeline` / `zaman çizelgesi` and the UI
  Show timeline action extract full dates and year-only mentions with
  source/title/page/section provenance; `--entity` / `entity=` / the UI entity
  filter scopes events to an entity. English and Turkish dates normalize to
  ISO, and year-only events keep `precision=year`.
- **Typed relations:** `atomzero-docs relations`,
  `atomzero-docs relation-path`, `GET /v1/graph/relations`,
  `GET /v1/graph/relations/path`, chat (`relations for X`, `X ilişkileri`,
  `relation path from X to Y`) and the UI Show relations action extract
  explicit `located_in`, `capital_of`, `founded_by`, `ceo_of` and `born_in`
  triples with source/title/page/section provenance; multi-hop paths use those
  edges. English and Turkish patterns are supported, and co-occurrence alone
  is not treated as a relation.
- **Custom skills:** drop JSON skills into `data/skills` to add trigger phrases,
  local-context prompts, provider workflows or deterministic templates;
  `atomzero-skills list/validate/show`. Skills are declarative, not arbitrary
  code. Run by name/query through `atomzero-skills run`, `POST /v1/skills/run`
  or the UI; chain skills with explicit `{previous}` context through
  `atomzero-skills chain` / `POST /v1/skills/chain`; persist named recipes under
  `data/skills/chains` and run them with `atomzero-skills recipe` /
  `POST /v1/skills/chains/run`; end-to-end benchmarks cover provider, template,
  fallback, chain and recipe modes.
- **Daily brief + weekly review + goal planner:** local tasks, pending gaps,
  index stats and graph entities in one brief (with `--brief-tag`, `brief me
  tag work` and HTTP/UI tag filters, plus recent tag audit events);
  `atomzero-brief` (including
  `--weekly --days 7 --markdown --out reports\weekly.md`) and
  `/v1/weekly-review` aggregate recent work with active task-tag counts,
  optional `--weekly-tag` filtering, `--weekly-audit-action` filtering and a
  Markdown Tags/Tag audit table.
  `/v1/weekly-review/export`
  produces a portable Markdown report; `--reports` and
  `/v1/weekly-review/reports` list the local report archive with UI download;
  `--reports --prune` removes entries whose files were deleted,
  `--reports --keep N` retains only the newest N reports,
  `--reports --keep N --dry-run` previews the removals without writing,
  `--reports --delete ID` removes one index entry, `--reports --query TEXT`
  filters the archive, `--reports --tag TAG` filters by exact report tag,
  `--reports --sort newest|oldest|largest` orders it,
  `--reports --verify` checks stored size/SHA-256 values and
  `--reports --repair` recomputes changed metadata.
  `/v1/goals/plan` produces a cited plan skeleton. Chat supports `brief me`,
  `find report weekly` / `rapor ara weekly`, `find report tag work`,
  `list reports tag work`, `list reports` (optional limit and
  newest/oldest/largest sort), `show report w0001`,
  `verify report archive` / `verify report w0001`,
  `repair report archive` / `repair report w0001` with `confirm reports`,
  `compare reports w0001 w0002` with Markdown export, archive
  `prune/keep/delete` previews,
  `weekly review` (optionally `for 14 days` / `haftalık özet 14 gün`) and
  `plan goal ...` → `confirm plan` with explicit task creation only after
  confirmation.
- **Goal-flow regression:** an end-to-end HTTP benchmark checks plan creates no
  tasks, confirm creates exactly three once, and cancel leaves memory unchanged.
- **Brief-me regression:** an end-to-end HTTP chat benchmark checks tasks,
  pending gaps, index counts, graph entities and briefing sources.
- **Skill-chain UI regression:** checks the served chain controls and executes
  the same request through the real handler, including `{previous}` hand-off.
- **Named-recipe UI regression:** checks recipe selection/execution, carry
  override and unknown-recipe behavior through the served UI contract and HTTP.
- **Headless CLI:** `atomzero-docs add/sync/status/doctor/search/answer/
  summary/summary-collection/list/evidence/export/import/forget` works without
  the web UI; `sync` keeps a portable relative-path collection, skips unchanged
  or duplicate content, prunes deleted files, summaries are extractive, and
  `doctor` checks index integrity.
- **Per-source delete:** Forget removes chunks, page/section metadata, source
  metadata and cached vectors.

See [DOCUMENTS.md](https://github.com/byjameson/atom-zero/blob/main/docs/DOCUMENTS.md).

## Tools and constrained plans

- time, safe arithmetic, task add/list/complete, name/likes memory.
- `POST /v1/agent/plan` executes a small JSON plan through an allowlist:
  - at most **3 steps**;
  - no shell, arbitrary URL or code execution;
  - **explicit confirmation** before task/research side effects;
  - text arguments pass the optional safety filter when enabled.

The 49.2M model does not choose tools freely. This is a constrained executor,
not model-driven planning. See
[TOOL_PLANNING.md](https://github.com/byjameson/atom-zero/blob/main/docs/TOOL_PLANNING.md).

## Research, curiosity and learning

When no local source answers a question, it becomes a knowledge gap. The
assistant can propose pending research topics, but **nothing fetches until a
human approves**:

```text
question → no local source → knowledge gap
gap → pending curiosity topic
human approves
worker fetches Wikipedia context
local Ollama teacher synthesizes
answer stored as an unverified teacher note in RAG
next answer can cite it; weights_changed: false
```

The self-improvement ladder is deliberately split:

- **RAG memory:** a teacher note or your document enters the index; no weight
  change.
- **Reviewed adapter:** a LoRA adapter for style/skills; human review + gates;
  weights change.
- **Full training:** a new base checkpoint; offline project decision; weights
  change offline.

Adapters are trained by the separate `streamft` / `continual-lora` projects and
validated locally with `research/adapter_check.py`; the product runtime does not
import either trainer. See [`docs/ADAPTERS.md`](docs/ADAPTERS.md).

The model never crawls, mutates its own weights or studies topics on its own
during chat. **Hungry mode** is an explicit, bounded command for growing the
retrieval index:

```powershell
atomzero-hungry --topic Ankara --max-questions 5
atomzero-hungry --dry-run --json
```

It generates questions from pending gaps, topic seeds and document titles,
checks the local index, asks the configured local teacher only for unknown
questions, stores answers as unverified notes and writes an audit ledger. It
never fetches web pages, never changes weights and stops at the question and
provider-call budgets. See
[HUNGRY_MODE.md](https://github.com/byjameson/atom-zero/blob/main/docs/HUNGRY_MODE.md).

## Optional local provider

```powershell
atomzero-quickstart --provider
```

This routes free-form chat to local Ollama `llama3.2:3b` while RAG, tools and
memory remain local. It is not the default backend. Use a different model with
`--provider-model`.

## Optional safety

Safety is **off by default**. With `--safety-filter`, exact rules run first and
a small local linear ML head is additive; a missing/broken head falls back to
the exact rules. It is an operator tool, not a hosted content policy.

## Limits

- 2/10 raw knowledge; weak general reasoning and coding.
- English-first tokenizer/data.
- 128-token core context; longer context is a model/training change.
- No OCR; scanned PDFs need OCR before ingestion.
- Encrypted PDFs are rejected; DOCX page numbers are unavailable.
- Product benchmark is narrow and authored; it is not MMLU.
- The plan API is not free-form model planning.
- The document benchmark is an authored regression gate, not a public
  leaderboard.

## Support

If Atom Zero is useful, donations help cover compute, evaluation and
maintenance. There is no paid support or investment offer, and donations do not
buy influence over results or roadmap decisions.

- **EVM-compatible networks** (Ethereum, BNB Smart Chain, Polygon, Optimism,
  Arbitrum One, Base, Avalanche C-Chain):
  `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.6.1/>
- GitHub source: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.6.1>
- Quickstart: <https://github.com/byjameson/atom-zero/blob/main/docs/QUICKSTART_ATOMZERO.md>
- Model card: <https://github.com/byjameson/atom-zero/blob/main/docs/MODEL_CARD.md>
- Full project map: <https://github.com/byjameson/atom-zero/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents: <https://github.com/byjameson/atom-zero/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/atom-zero/blob/main/docs/TOOL_PLANNING.md>
- Adapters: <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTERS.md>
- Answerability experiment: <https://github.com/byjameson/atom-zero/blob/main/docs/ANSWERABILITY.md>
- Context budget: <https://github.com/byjameson/atom-zero/blob/main/docs/CONTEXT_BUDGET.md>
- Proactive conversation: <https://github.com/byjameson/atom-zero/blob/main/docs/PROACTIVE.md>
- Memory lifecycle: <https://github.com/byjameson/atom-zero/blob/main/docs/MEMORY_LIFECYCLE.md>
- Claim / evidence algebra: <https://github.com/byjameson/atom-zero/blob/main/docs/CLAIMS.md>
- Cross-source consensus: <https://github.com/byjameson/atom-zero/blob/main/docs/CONSENSUS.md>
- Source freshness: <https://github.com/byjameson/atom-zero/blob/main/docs/FRESHNESS.md>
- Document version diff: <https://github.com/byjameson/atom-zero/blob/main/docs/DOCUMENT_DIFF.md>
- AISC spec generator: <https://github.com/byjameson/atom-zero/blob/main/docs/AISC_SPEC_GENERATOR.md>
- Multi-agent orchestration: <https://github.com/byjameson/atom-zero/blob/main/docs/ORCHESTRATION.md>
- Branch research: <https://github.com/byjameson/atom-zero/blob/main/docs/BRANCH_RESEARCH.md>
- Ten-increment batch plan: <https://github.com/byjameson/atom-zero/blob/main/docs/BATCH_0.7.0.md>
- Benchmarks: <https://github.com/byjameson/atom-zero/blob/main/docs/BENCHMARKS.md>
- AISC architecture audit: <https://github.com/byjameson/atom-zero/blob/main/docs/AISC_AUDIT.md>
- Public SQuAD retrieval benchmark: <https://github.com/byjameson/atom-zero/blob/main/benchmarks/README.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-a-49m-local-assistant-that-says-i-dont-know-1p7m>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-a-49m-local-assistant-that-says-i-dont-know-df1ebc0e2e78>
- LinkedIn: <https://www.linkedin.com/in/g%C3%B6khan-cava-874a32157/>
- Contact: **gokhancava@gmail.com**

Apache-2.0 for code and weights; data attribution in `NOTICE`.
