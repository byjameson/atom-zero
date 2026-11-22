# Daily brief and goal planner

Atom Zero can produce a local morning brief and a cited plan skeleton without
calling a model or a provider:

- open tasks from the current session;
- pending knowledge gaps;
- local index size and recent source titles;
- top entities/connections from the second-brain graph;
- a focus item and a short next-action list.

## CLI

```powershell
atomzero-brief --session data/sessions/default.json --docs data/atomzero_demo.db
atomzero-brief --session data/sessions/default.json --docs data/atomzero_demo.db --json
atomzero-brief --session data/sessions/default.json --docs data/atomzero_demo.db `
  --goal "prepare the warranty page"
```

If `--session` is omitted, the CLI uses
`data/sessions/<session-id>.json`. Missing session/document/gap files are
treated as empty, not as errors. Session JSON is read as UTF-8 with or without
a BOM, so files written by PowerShell `Set-Content -Encoding utf8` work
directly.

## Weekly review

```powershell
atomzero-brief --session data/sessions/default.json `
  --docs data/atomzero_demo.db --weekly --days 7
```

```text
POST /v1/weekly-review
{"session_id": "default", "days": 7}
```

A weekly review aggregates:

- open tasks and tasks completed within the window;
- undated completed tasks separately;
- pending gaps;
- sources indexed within the window;
- graph entities/connections and highlights;
- suggested next-week actions.

Completions carry `completed` timestamps when tasks are finished through
`ChatHuman`; older undated completions are listed separately rather than
silently dropped. The UI **Weekly review** button uses the same endpoint.

Chat also runs the review directly:

```text
weekly review
weekly review for 14 days
haftalık özet
haftalık özet 14 gün
son 14 günün özeti
```

The reply contains the same review text and exposes the full `review` payload
(tasks, gaps, sources, graph, highlights and next actions) to clients.

### Markdown export

```powershell
atomzero-brief --session data/sessions/default.json `
  --docs data/atomzero_demo.db --weekly --days 7 --markdown
atomzero-brief --session data/sessions/default.json `
  --docs data/atomzero_demo.db --weekly --days 7 --markdown `
  --out reports\weekly-review.md
```

Without `--out` the report goes to stdout. With `--out` the CLI creates parent
directories, writes UTF-8 Markdown through a sibling temporary file and
atomically replaces the target, then prints a short confirmation. If the write
or replace fails, the previous report is left intact and the temporary file is
removed. `--out` requires both `--weekly` and `--markdown`.

```text
GET /v1/weekly-review/export?session_id=default&days=7
```

The endpoint returns `text/markdown; charset=utf-8` with a
`Content-Disposition: attachment` filename such as
`atomzero-weekly-review-2026-11-22.md`. The UI **Download .md** button fetches
the same endpoint and saves the report.

The export contains task checkboxes (`[ ]` / `[x]`), pending gaps, recent
sources, graph entities/connections, highlights and next-week actions. It is
generated locally and does not change model weights.

### Report archive

Every successful `--out` export is recorded in a local JSON index:

```powershell
atomzero-brief --reports
atomzero-brief --reports --json
atomzero-brief --reports --prune
atomzero-brief --reports --keep 10
atomzero-brief --reports --keep 10 --dry-run
atomzero-brief --reports --delete w0001
atomzero-brief --reports --reports-index reports\weekly_reports.json --limit 5
```

The index defaults to `data/weekly_reports.json`. Each output path has one
entry, refreshed on later exports to the same path. Entries store the report
path, time, period, task/source counts, graph counts, size and SHA-256.

`--prune` removes entries whose report file no longer exists and prints the
removed count. `--keep N` retains only the newest N entries and atomically
removes older ones. `--keep N --dry-run` lists what would be removed without
changing the index; the output ends with `Archive unchanged: N report(s).`
With `--json`, the result includes `removed`, `trimmed`, or
`would_remove_count`/`would_remove` fields as applicable.

`--delete ID` removes one archive entry atomically and leaves the report file
on disk.

```text
GET  /v1/weekly-review/reports?limit=10
GET  /v1/weekly-review/reports?id=w0001
POST /v1/weekly-review/reports/prune
POST /v1/weekly-review/reports/keep
POST /v1/weekly-review/reports/delete
{"keep": 10}
{"keep": 10, "dry_run": true}
{"id": "w0001"}
```

The UI **Recent reports** panel lists the archive, downloads an indexed
Markdown file, exposes **Delete** for one entry, **Prune missing** for deleted
files and a **Keep newest** retention control with **Preview only**. A dry-run
response includes `would_remove` entries and leaves the archive unchanged.
Keep `--reports-index` pointed at the same file for CLI and the server; both
default to `data/weekly_reports.json`. `quickstart` and `chat_server` accept
`--reports-index`.

## Chat flow

```text
brief me
-> daily brief in the chat: open tasks, pending gaps, index and graph

weekly review
-> weekly review; optionally "for 14 days" or "14 gün"

plan goal prepare the warranty page
-> cited plan skeleton, no task creation
-> "Reply 'confirm plan' to create these tasks, or 'cancel plan'."

confirm plan
-> creates up to three suggested tasks

cancel plan
-> discards the pending plan
```

Task creation requires the explicit `confirm plan` message. There is no
background task creation and no side effect from the initial plan request.

## API

```text
POST /v1/briefing
{"session_id": "default"}

POST /v1/goals/plan
{"session_id": "default", "goal": "prepare the warranty page", "max_steps": 3}
```

Goal planning returns:

- `context`: local snippets with `source`, `page`, `section`, `citation`;
- `confidence`: the existing retrieval match heuristic;
- `suggested_tasks`: up to `max_steps` concrete actions;
- `open_tasks` and `pending_gaps` that may be relevant;
- `created_tasks`: empty unless the request contains `"confirm": true`.

**Side effects require confirmation.** A plan request without `confirm: true`
does not change task memory. With `"confirm": true`, Atom Zero adds up to three
suggested tasks to the session and returns their ids.

## UI

The **Daily brief** panel has:

- **Get daily brief** — tasks, gaps, index, graph and next actions;
- a goal input and **Plan goal** — cited local context plus suggested actions;
- **Create suggested tasks** — explicit second click that writes the tasks.

## Honest limits

- The brief is deterministic and local. It summarizes what the session and
  index already contain; it does not invent priorities.
- Goal planning is a **plan skeleton**, not autonomous execution. It retrieves
  evidence and proposes steps; the user chooses what to do.
- Side effects only happen on explicit confirmation.
- No provider is required. If a local provider is configured, normal chat and
  custom skills can still use it, but the brief/plan endpoints do not call it.
- No model weights are changed.
