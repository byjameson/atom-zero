# Atom Zero 63M — a local assistant that cites its sources

Atom Zero is a local, source-cited assistant built around a **63.4M parameter model**,
depth-upcycled from the 49.2M from-scratch base and validated as the default. Facts live in a local RAG index, not in the model
weights; the model handles wording and behavior while deterministic tools,
documents, memory and citations do the rest.

This is the standalone **`atomzero`** package. The low-VRAM trainer is a separate
package and is **not required** to run Atom Zero. Atom Zero is published on eight
canonical surfaces: GitHub, Hugging Face, PyPI, dev.to, Medium, LinkedIn,
X and ModelScope. Reddit is not used (its filters remove posts).

**Live demo:** [Atom Zero 63M on ModelScope Studio](https://modelscope.cn/studios/byjameson/atomzerodemo) — CPU, synthetic demo corpus, research demo only.

## Support the project

[![Sponsor on GitHub](https://img.shields.io/badge/Sponsor-GitHub%20Sponsors-ea4aaa?logo=githubsponsors)](https://github.com/sponsors/byjameson)

Atom Zero is independent and measured in public. Sponsors fund GPU hours,
distillation and upcycling experiments, safety and retention gates, and
reproducible releases. The current goal is **$10,000/month** to fund Atom Zero
full-time: one researcher, rented GPU time for distillation/upcycling, data and
release infrastructure, and safety review. Crypto and other options are in
[`docs/DONATE.md`](docs/DONATE.md).

## Install

```powershell
pip install atomzero==2.4.0
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

1. downloads the ~279 MB checkpoint/tokenizer from
   [Hugging Face](https://huggingface.co/byjameson/atom-zero-63m-balanced);
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

Optional research bases can be served without changing the default:

```powershell
atomzero-quickstart --model-dir checkpoints/mixed8_safety_v1
atomzero-quickstart --model-repo byjameson/atom-zero-63m-mixed
```

The 63M balanced base is the default. The 49M `0.8.0` base remains as a
fallback:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-49m `
  --cache ~/.cache/atomzero/atom-zero-49m
```

See [`docs/OPTIONAL_BASES.md`](docs/OPTIONAL_BASES.md) for the other research
bases.

## What you can do with Atom Zero

- Ask who it is: “What is your name?” — “I am Atom Zero, a local assistant
  created by Gökhan Çava.”; “Who created you?” — “Atom Zero was created by
  Gökhan Çava.”
- Greet or thank it: “Good afternoon! How are things?” or “Thanks, that was
  useful.” — a complete, helpful reply instead of a two-word answer.
- Ask benign trigger questions: “How do I kill a background process?”, “What is
  the safest way to cut an onion?”, “How do I handle a damaged delivery?” —
  helpful answers instead of a security refusal.
- Recall what you told it: “My name is Ada.” then “What is my name?” — “Ada.”;
  “I live in Izmir.” then “Which city do I live in?” — “Izmir.”; “Today is
  Sunday.” then “What day is tomorrow?” — “Tomorrow is Monday.”
- Correct a preference or language: “I prefer coffee over water.” then “What do
  I prefer?” — “coffee.”; “Use Java for this project.” then “Actually, use Go.
  Which language are we using?” — “Go.”
- Check a purchase against a budget: “My budget is $500.” then “Can I spend
  $300?” — “Yes, $300 is within your $500 budget.” Ask about $650 and it
  answers “No, $650 is over your $500 budget by $150.” These answers come
  from a deterministic checker, not from arithmetic in the weights.
- Check a task against a time window: “I have 45 minutes.” then “Can I do a
  60-minute task?” — “No, a 60-minute task is 15 minutes over your
  45-minute window.”
- Copy structured records exactly as 4-key JSON or KV.
- Ingest a local document and ask questions with cited answers; the Studio
  demo returns **L-7781** from `launch_policy.txt` for “What is the launch
  code?”.
- Run offline on CPU or a small GPU; documents and sessions stay local.
- Get a supportive refusal on crisis prompts while ordinary support
  questions still receive practical help.

The cache lives at `~/.cache/atomzero/atom-zero-63m-balanced`.

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
  63.4M core is deliberately narrow.
- **Optionally, a front-end for a stronger local provider** while RAG, tools and
  memory stay local.

## What it is not

- Not a general reasoning or coding model.
- Not a ChatGPT competitor.
- Not a hosted, multi-tenant service.
- Not a model that stores all facts in its weights.
- Not an autonomous crawler or self-modifying agent.
- Not a claim that 63.4M parameters are enough for general intelligence.

## System features

| area | implemented |
|---|---|
| core model | 63.4M decoder-only transformer, BPE-8k, 8 layers (upcycled), 768 dim, 128 context |
| training | 49.2M base upcycled to 8 layers / 63.4M; 3,000-step mixed fine-tune |
| RAG | SQLite FTS5, hashed/MiniLM embeddings, hybrid candidates, lexical+cosine rerank |
| vector index | cached vectors, prewarmed numpy matrix, IVF ANN above 256 chunks, exact fallback |
| documents | text/Markdown/HTML/PDF/DOCX ingestion, page/section citations, evidence panel/API |
| tools | time, safe arithmetic, tasks, name/likes memory, allowlisted 3-step plan API |
| sessions | transcript restore, JSON export/import, exact Forget, session Forget |
| research | approved-only queue, Wikipedia fetch, worker lock/log, teacher synthesis |
| teacher | local Ollama/OpenAI-compatible, unverified notes, reviewer verification |
| curiosity | gap â†’ pending topic â†’ approval â†’ Wikipedia + teacher â†’ RAG note |
| self-learning | RAG memory (no weights), reviewed LoRA (weights), manual full training |
| provider | optional local free-form chat; RAG/tools stay local |
| safety | optional rules + local linear ML head; off by default; exact fallback |
| privacy | localhost, no telemetry, SSRF guard, upload caps, deletion |
| UI | chat, documents, evidence, gaps, research queue, teacher notes, adapters, health |

## Model

| item | value |
|---|---|
| name | Atom Zero 63M balanced (`atom-zero-63m-balanced`) |
| architecture | decoder-only transformer |
| parameters | 63,389,184 |
| layers / dim / heads / FF | 8 / 768 / 12 / 3072 |
| context | 128 tokens |
| tokenizer | 8,192-token BPE (`chat_en_bpe8k.json`) |
| checkpoint step | 3,000 mixed fine-tune steps (from the 49M 40k base) |
| training data | 6000 copy + 492 content + 480 benign x3 + 40 safety x5 + 16 self-harm x5 + 96 boundary pairs |
| runtime | PyTorch, CUDA or CPU; RTX 3050 6 GB tested |

The 49.2M base was trained from scratch and depth-upcycled to 63.4M; the default
then received a 3,000-step mixed fine-tune. It is a **narrow chat/behavior
model**, not a foundation model. Facts are expected to live in RAG; skills can
live in reviewed LoRA adapters.

## Honest benchmark

| check | result |
|---|---:|
| raw 49.2M fallback, no RAG/tools | **2/10** |
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
| test suite | **1764 passed** |

The raw-model probe asks ten simple general questions with no documents, no
calculator and no agent. The 49.2M fallback gets **2/10**. The product gets
**21/21** because the runtime does the work.

The raw `pytest -v` run for the tested commit, together with the AISC suite
run, is recorded in [`TEST_RESULTS.md`](TEST_RESULTS.md).

The product benchmark is narrow and authored: tools, memory, tasks,
source-cited retrieval, teacher-note review, missing-source detection and one
short generation. It is **not MMLU, GSM8K or HumanEval**.

Capability claims are reported separately in the six-criterion
[capability scorecard](docs/CAPABILITY_SCORECARD.md): `met`, `partial`,
`not_met` and `unknown` stay distinct, and runtime tool accuracy is not counted
as model transfer.

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
  (`kitap` / `kitabÄ±n`, `aÄŸaÃ§` / `aÄŸacÄ±n`, `ev` / `evler`, `ÅŸehir` / `ÅŸehrin`,
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
  `GET /v1/documents/timeline`, chat `timeline` / `zaman Ã§izelgesi` and the UI
  Show timeline action extract full dates and year-only mentions with
  source/title/page/section provenance; `--entity` / `entity=` / the UI entity
  filter scopes events to an entity. English and Turkish dates normalize to
  ISO, and year-only events keep `precision=year`.
- **Typed relations:** `atomzero-docs relations`,
  `atomzero-docs relation-path`, `GET /v1/graph/relations`,
  `GET /v1/graph/relations/path`, chat (`relations for X`, `X iliÅŸkileri`,
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
  `weekly review` (optionally `for 14 days` / `haftalÄ±k Ã¶zet 14 gÃ¼n`) and
  `plan goal ...` â†’ `confirm plan` with explicit task creation only after
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
- **Deterministic copy tool:**
  `atomzero-copy --text "a=1;b=hello world" --format json` extracts explicit
  `key=value` pairs and renders JSON, KV or YAML in ordinary Python. It is a
  runtime transcription tool, not a model capability or a training result; see
  [`docs/COPY_TOOL.md`](docs/COPY_TOOL.md).
- **Teacher distillation data:** verified, canonicalized copy rows for offline
  student training; see [`docs/DISTILL_COPY.md`](docs/DISTILL_COPY.md).
- **New-task teacher data:** teacher-invented 2/3/4-key copy tasks with new
  keys/values, checked by deterministic constraints; see
  [`docs/DISTILL_NEW_TASKS.md`](docs/DISTILL_NEW_TASKS.md).
- `POST /v1/agent/plan` executes a small JSON plan through an allowlist:
  - at most **3 steps**;
  - no shell, arbitrary URL or code execution;
  - **explicit confirmation** before task/research side effects;
  - text arguments pass the optional safety filter when enabled.

The 63.4M model does not choose tools freely. This is a constrained executor,
not model-driven planning. See
[TOOL_PLANNING.md](https://github.com/byjameson/atom-zero/blob/main/docs/TOOL_PLANNING.md).

## Research, curiosity and learning

When no local source answers a question, it becomes a knowledge gap. The
assistant can propose pending research topics, but **nothing fetches until a
human approves**:

```text
question â†’ no local source â†’ knowledge gap
gap â†’ pending curiosity topic
human approves
worker fetches Wikipedia context
local Ollama teacher synthesizes
answer stored as an unverified teacher note in RAG
next answer can cite it; weights_changed: false
```

The self-improvement ladder is deliberately split:

- **Growth ladder:** local retrieval -> opt-in teacher -> verified memory ->
  skills -> distilled adapters -> offline checkpoint upcycling -> agentic code
  improvement. Asking a larger AI does not automatically grow the parameter
  count; only the offline, human-approved training steps do. See
  [`docs/GROWTH_LADDER.md`](docs/GROWTH_LADDER.md).
- **Self-code audit:** deterministic AST scan of Atom Zero's own source; it
  proposes bounded patches, and the autonomous loop gates and merges them
  locally; per-change human approval is optional.
- **RAG memory:** a teacher note or your document enters the index; no weight
  change.
- **Reviewed adapter:** a LoRA adapter for style/skills; human review + gates;
  weights change.
- **Full training:** a new base checkpoint; offline project decision; weights
  change offline.

Adapters are trained by the separate `streamft` / `continual-lora` projects and
validated locally with `research/adapter_check.py`; the product runtime does not
import either trainer. See [`docs/ADAPTERS.md`](docs/ADAPTERS.md).

Full research checkpoints are gated separately with
`research/checkpoint_gate.py`: held-out content loss, exact match, key copy and
an optional safety probe against the product base. A pass is necessary but not
sufficient for replacing the base. See
[`docs/CHECKPOINT_GATE.md`](docs/CHECKPOINT_GATE.md).

Before full fine-tuning, `research/mix_training_data.py` builds a deduplicated
copy + content/replay mix with a content-free manifest and an eval-path guard.
The content side can be built deterministically from a local facts JSONL with
`research/build_content_rows.py`; these rows are rehearsal data, not a
factuality claim. See [`docs/CONTENT_ROWS.md`](docs/CONTENT_ROWS.md). The mixed
file still has to pass retention and the checkpoint gate; see
[`docs/MIXED_REPLAY.md`](docs/MIXED_REPLAY.md).

The `--safety-eval` probe file is generated deterministically by
`research/build_safety_probes.py`; see
[`docs/SAFETY_PROBES.md`](docs/SAFETY_PROBES.md). Disjoint safety refusal
rehearsal rows and their held-out probes are built by
`research/build_safety_replay.py`; see
[`docs/SAFETY_REPLAY.md`](docs/SAFETY_REPLAY.md). Contrastive self-harm
rehearsal pairs crisis prompts with supportive refusals and near-neighbour
benign support prompts with helpful answers; see
[`docs/SELFHARM_REPLAY.md`](docs/SELFHARM_REPLAY.md). Contrastive boundary
pairs extend the same pattern to prompt injection, secret exfiltration,
privacy and harmful/illegal requests; see
[`docs/BOUNDARY_PAIRS.md`](docs/BOUNDARY_PAIRS.md). Ordinary conversational
rehearsal rows for countering over-refusal are built by
`research/build_benign_chat.py`; see
[`docs/BENIGN_CHAT.md`](docs/BENIGN_CHAT.md). A teacher-generated variant keeps
the same eight categories and authored-prompt approach but asks a larger local
teacher for filtered, paraphrased answers; see
[`docs/TEACHER_BENIGN_CHAT.md`](docs/TEACHER_BENIGN_CHAT.md). The product
screen's benign control set is built by `research/build_benign_prompts.py`: 120
single-turn prompts plus 16 multi-turn cases by default; see
[`docs/BENIGN_SUITE.md`](docs/BENIGN_SUITE.md).

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

- 2/10 raw knowledge on the 49M fallback; weak general reasoning and coding.
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

- PyPI: <https://pypi.org/project/atomzero/2.4.0/>
- GitHub source: <https://github.com/byjameson/atom-zero>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v2.4.0>
- dev.to 2.3.0: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-230-knows-its-name-refuses-less-3lbe>
- LinkedIn 2.3.0: <https://www.linkedin.com/feed/update/urn:li:activity:7532383623811948544/>
- X 2.3.0: <https://x.com/aigokhan/status/2126618148271194231>
- Quickstart: <https://github.com/byjameson/atom-zero/blob/main/docs/QUICKSTART_ATOMZERO.md>
- Model card: <https://github.com/byjameson/atom-zero/blob/main/docs/MODEL_CARD.md>
- Full project map: <https://github.com/byjameson/atom-zero/blob/main/docs/PROJECT_OVERVIEW.md>
- Documents: <https://github.com/byjameson/atom-zero/blob/main/docs/DOCUMENTS.md>
- Tool plans: <https://github.com/byjameson/atom-zero/blob/main/docs/TOOL_PLANNING.md>
- Copy tool: <https://github.com/byjameson/atom-zero/blob/main/docs/COPY_TOOL.md>
- Content retention eval: <https://github.com/byjameson/atom-zero/blob/main/docs/RETENTION.md>
- Mixed copy + content replay data: <https://github.com/byjameson/atom-zero/blob/main/docs/MIXED_REPLAY.md>
- Content replay row builder: <https://github.com/byjameson/atom-zero/blob/main/docs/CONTENT_ROWS.md>
- Vast.ai GPU runs: <https://github.com/byjameson/atom-zero/blob/main/docs/VASTAI.md>
- Adapters: <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTERS.md>
- Answerability experiment: <https://github.com/byjameson/atom-zero/blob/main/docs/ANSWERABILITY.md>
- Context budget: <https://github.com/byjameson/atom-zero/blob/main/docs/CONTEXT_BUDGET.md>
- Proactive conversation: <https://github.com/byjameson/atom-zero/blob/main/docs/PROACTIVE.md>
- Memory lifecycle: <https://github.com/byjameson/atom-zero/blob/main/docs/MEMORY_LIFECYCLE.md>
- Claim / evidence algebra: <https://github.com/byjameson/atom-zero/blob/main/docs/CLAIMS.md>
- Cross-source consensus: <https://github.com/byjameson/atom-zero/blob/main/docs/CONSENSUS.md>
- Source freshness: <https://github.com/byjameson/atom-zero/blob/main/docs/FRESHNESS.md>
- Temporal validity: <https://github.com/byjameson/atom-zero/blob/main/docs/TEMPORAL.md>
- Source reliability ledger: <https://github.com/byjameson/atom-zero/blob/main/docs/RELIABILITY.md>
- Document version diff: <https://github.com/byjameson/atom-zero/blob/main/docs/DOCUMENT_DIFF.md>
- Evaluation gate: <https://github.com/byjameson/atom-zero/blob/main/docs/EVALUATION_GATE.md>
- Deterministic safety probes: <https://github.com/byjameson/atom-zero/blob/main/docs/SAFETY_PROBES.md>
- Safety refusal replay: <https://github.com/byjameson/atom-zero/blob/main/docs/SAFETY_REPLAY.md>
- Contrastive boundary pairs: <https://github.com/byjameson/atom-zero/blob/main/docs/BOUNDARY_PAIRS.md>
- Benign chat replay: <https://github.com/byjameson/atom-zero/blob/main/docs/BENIGN_CHAT.md>
- Teacher-generated benign chat: <https://github.com/byjameson/atom-zero/blob/main/docs/TEACHER_BENIGN_CHAT.md>
- Benign prompt suite: <https://github.com/byjameson/atom-zero/blob/main/docs/BENIGN_SUITE.md>
- Product suite (benign prompts + safety + copy): <https://github.com/byjameson/atom-zero/blob/main/docs/PRODUCT_SUITE.md>
- Server product suite (RAG + copy + multi-turn over HTTP): <https://github.com/byjameson/atom-zero/blob/main/docs/SERVER_PRODUCT_SUITE.md>
- Self-improvement architecture: <https://github.com/byjameson/atom-zero/blob/main/docs/SELF_IMPROVEMENT.md>
- Autonomous loop and sandbox: <https://github.com/byjameson/atom-zero/blob/main/docs/AUTONOMOUS.md>
- First measured loop cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE.md>
- Second measured loop cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_2.md>
- Third loop cycle (agent-bridge consumed): <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_3.md>
- Fourth loop cycle (opt-in runner): <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_4.md>
- Fifth loop cycle (autonomous runner loop): <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_5.md>
- Sixth loop cycle (self-code finding): <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_6.md>
- Seventh loop cycle (targeted audit + regression gate): <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_7.md>
- Eighth loop cycle (fully autonomous, sandboxed): <https://github.com/byjameson/atom-zero/blob/main/docs/LOOP_CYCLE_8.md>
- Improvement loop: <https://github.com/byjameson/atom-zero/blob/main/docs/IMPROVEMENT_LOOP.md>
- Self-improvement scoreboard: <https://github.com/byjameson/atom-zero/blob/main/docs/SCOREBOARD.md>
- Improvement trend: <https://github.com/byjameson/atom-zero/blob/main/docs/TREND.md>
- Calibration drift: <https://github.com/byjameson/atom-zero/blob/main/docs/DRIFT.md>
- Agent handoff bridge: <https://github.com/byjameson/atom-zero/blob/main/docs/AGENT_BRIDGE.md>
- Opt-in agent runner: <https://github.com/byjameson/atom-zero/blob/main/docs/AGENT_RUNNER.md>
- Self-code audit and proposals: <https://github.com/byjameson/atom-zero/blob/main/docs/SELF_CODE.md>
- Self-code diff: <https://github.com/byjameson/atom-zero/blob/main/docs/SELF_CODE_DIFF.md>
- Growth ladder: <https://github.com/byjameson/atom-zero/blob/main/docs/GROWTH_LADDER.md>
- Growth report: <https://github.com/byjameson/atom-zero/blob/main/docs/GROWTH_REPORT.md>
- Escalation router: <https://github.com/byjameson/atom-zero/blob/main/docs/ESCALATION.md>
- Teacher verification: <https://github.com/byjameson/atom-zero/blob/main/docs/TEACHER_VERIFICATION.md>
- Adapter candidates: <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTER_CANDIDATES.md>
- Task-family separation gate: <https://github.com/byjameson/atom-zero/blob/main/docs/FAMILY_GATE.md>
- Real teacher queue and cross-distribution fixture: <https://github.com/byjameson/atom-zero/blob/main/docs/TEACHER_QUEUE.md>
- Teacher task families (JSON, key-value, YAML, arithmetic): <https://github.com/byjameson/atom-zero/blob/main/docs/TEACHER_FAMILIES.md>
- Adapter training cycle 2 (real LoRA): <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTER_TRAINING.md>
- Adapter cross-distribution cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTER_CROSS_CYCLE.md>
- Adapter promotion: <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTER_PROMOTION.md>
- Diverse teacher + replay cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/DIVERSE_REPLAY_CYCLE.md>
- Exact-match content audit: <https://github.com/byjameson/atom-zero/blob/main/docs/CONTENT_AUDIT.md>
- Copy-curriculum value-vs-key cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/COPY_CURRICULUM_CYCLE.md>
- Multi-key rehearsal and adapter soup: <https://github.com/byjameson/atom-zero/blob/main/docs/MULTI_KEY_REHEARSAL.md>
- Curriculum loop evidence: <https://github.com/byjameson/atom-zero/blob/main/docs/CURRICULUM_LOOP_EVIDENCE.md>
- Randomized-key copy cycle: <https://github.com/byjameson/atom-zero/blob/main/docs/RANDOM_KEY_COPY_CYCLE.md>
- Full fine-tune copy control: <https://github.com/byjameson/atom-zero/blob/main/docs/FULL_FINETUNE_CONTROL.md>
- Gradient accumulation: <https://github.com/byjameson/atom-zero/blob/main/docs/GRADIENT_ACCUMULATION.md>
- Replay-aware adapter training: <https://github.com/byjameson/atom-zero/blob/main/docs/REPLAY_TRAINING.md>
- Optional pretokenized training cache: <https://github.com/byjameson/atom-zero/blob/main/docs/PRETOKENIZED.md>
- Adapter growth cycle 1: <https://github.com/byjameson/atom-zero/blob/main/docs/ADAPTER_CYCLE_1.md>
- AISC spec generator: <https://github.com/byjameson/atom-zero/blob/main/docs/AISC_SPEC_GENERATOR.md>
- Multi-agent orchestration: <https://github.com/byjameson/atom-zero/blob/main/docs/ORCHESTRATION.md>
- Branch research: <https://github.com/byjameson/atom-zero/blob/main/docs/BRANCH_RESEARCH.md>
- 1.0.0 parallel batch plan: <https://github.com/byjameson/atom-zero/blob/main/docs/BATCH_1.0.0.md>
- Release cadence: one minor version per release (next: 1.9.0)
- Ten-increment batch plan: <https://github.com/byjameson/atom-zero/blob/main/docs/BATCH_0.7.0.md>
- Self-improvement batch plan: <https://github.com/byjameson/atom-zero/blob/main/docs/BATCH_0.9.0.md>
- Benchmarks: <https://github.com/byjameson/atom-zero/blob/main/docs/BENCHMARKS.md>
- AISC architecture audit: <https://github.com/byjameson/atom-zero/blob/main/docs/AISC_AUDIT.md>
- Public SQuAD retrieval benchmark: <https://github.com/byjameson/atom-zero/blob/main/benchmarks/README.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-180-the-first-7b-teacher-promotion-lo0>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-1-9-0-fewer-refusals-copy-held-coherence-is-next-6bfdf07d7fef>
- LinkedIn: <https://www.linkedin.com/feed/update/urn:li:activity:7531941922496475136/>
- X post: <https://x.com/aigokhan/status/2126176327703179348>
- Contact: **gokhancava@gmail.com**

Apache-2.0 for code and weights; data attribution in `NOTICE`.
