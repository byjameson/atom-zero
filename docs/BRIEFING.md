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

## Task due dates

Tasks can carry an optional local due date and priority:

```text
add task submit report due 2026-11-30 priority high
add task call mom due tomorrow
add task review notes priority low due in 3 days
set task 1 priority normal
```

Priority is `low`, `normal` or `high`; the default is `normal`. Existing tasks
without the field are treated as normal. `list tasks` shows due dates, marks
overdue items and tags non-normal priorities as `[low]` or `[high]`.
`list tasks priority` (also `list tasks sort priority` or
`list tasks sorted by priority`) orders open tasks high before normal before
low, then earliest due date, then task id.

`list tasks` also accepts filters: `done`, `all`, `due week`, `due overdue`,
`due today`, `due none`, `priority high|normal|low` and `tag work`. Filters can
be combined, for example `list tasks done`,
`list tasks due week priority high`, `list tasks tag work` or
`list tasks all due none priority high`. Status defaults to open; due filters
match the task-search semantics. A bare `#work` suffix is also accepted.

Chat returns a structured `list_tasks` tool call with `status`, `due`,
`priority`, `tag`, `sort` and `count`. The same view is available over HTTP as
`GET /v1/tasks/list?status=open&due=week&priority=high&tag=work&sort=priority`
and in the UI **Task list** controls.

Filtered task lists export as CSV or Markdown:

```powershell
atomzero-brief --session data/sessions/default.json `
  --task-list --task-status open --task-due week `
  --task-priority high --task-tag work --task-sort priority `
  --task-format csv
```

`--task-format markdown` emits a filtered Markdown table. HTTP uses
`GET /v1/tasks/export?status=open&due=week&priority=high&tag=work&sort=priority&format=csv`
(or `format=markdown`). The UI **Download CSV** and **Download .md** buttons
use the currently selected task-list filters. CSV columns are
`id,status,priority,tags,text,due,overdue,created,completed`.

Tasks can carry up to five tags:

```text
add task submit report tags work,finance
tag task 1 work,urgent
tag task 1 clear
```

Tags are lower-case letters, numbers, `-` and `_`; the UI task-list filter
also accepts `#work`.

`list tags` (also `show tags` or just `tags`) counts the tag vocabulary:

```text
Task tags: 2 distinct | 3 tagged task(s) (2 open, 1 done) | 1 untagged
- work: 2 open, 1 done, 3 total
- finance: 1 open, 1 done, 2 total
```

The CLI form is `atomzero-brief --session ... --list-tags [--json]`, HTTP uses
`GET /v1/tasks/tags?session_id=default`, and the UI **Tags** button shows
clickable counts that fill the task-list tag filter.

Tags can be renamed across every task with an explicit confirmation step:

```text
rename tag work job
# Rename tag 'work' to 'job' on 3 task(s) (2 open, 1 done)?
# Reply "confirm tag rename" to apply, or "cancel tag rename".
confirm tag rename
# Renamed tag 'work' to 'job' on 3 task(s).
```

The CLI previews by default and writes only with an explicit flag:

```powershell
atomzero-brief --session data/sessions/default.json --rename-tag work job
atomzero-brief --session data/sessions/default.json --rename-tag work job --confirm-rename-tag
```

HTTP uses `POST /v1/tasks/tags/rename`; without `"confirm": true` it returns
`400 confirmation required` plus the affected count, and with it applies the
rename. The UI has old/new tag fields and a **Confirm tag rename** checkbox.
Renaming preserves other tags, removes duplicates and leaves untagged tasks
untouched. An unused old tag is reported instead of creating a no-op.

Tags can also be deleted globally with the same preview/confirm gate:

```text
delete tag work
# Delete tag 'work' from 3 task(s) (2 open, 1 done)?
# Reply "confirm tag delete" to remove it, or "cancel tag delete".
confirm tag delete
# Deleted tag 'work' from 3 task(s).
```

```powershell
atomzero-brief --session data/sessions/default.json --delete-tag work
atomzero-brief --session data/sessions/default.json --delete-tag work --confirm-delete-tag
```

HTTP uses `POST /v1/tasks/tags/delete` with `{"tag": "work", "confirm": true}`;
the UI has a tag field, **Confirm tag delete** checkbox and **Delete tag**
button. Other tags on each task are preserved; a task can become untagged. An
unused tag is reported instead of creating a no-op.

The last confirmed tag deletion can be undone from the per-task snapshot:

```text
undo delete tag
# Restored tag 'work' on 2 task(s).
```

```powershell
atomzero-brief --session data/sessions/default.json --undo-delete-tag
```

HTTP uses `POST /v1/tasks/tags/undo-delete`; the UI has an **Undo tag delete**
button. The snapshot stores the affected task ids and their previous tags in
the session, so a separate CLI or HTTP process can undo. Only the last deletion
is kept; a successful undo clears the snapshot. If a task was edited meanwhile,
its current other tags are preserved and the deleted tag is re-inserted; tags
already present are reported separately.

Tag delete and undo events are also kept in a local audit trail (last 20):

```text
tag audit
tag audit 5
tag audit work
tag audit action delete
tag audit tag work action undo 1
# Tag audit (2 of 2 event(s)):
# - 2026-11-23T10:00:00 | undo delete tag 'work' | restored 2 task(s) [#1, #2]
# - 2026-11-23T09:59:00 | delete tag 'work' | 2 task(s) (1 open, 1 done) [#1, #2]
```

CLI uses `--tag-audit [--tag-audit-limit N] [--tag-audit-tag TAG]
[--tag-audit-action delete_tag|undo_tag_delete]`, HTTP uses
`GET /v1/tasks/tags/audit?limit=10&tag=work&action=delete_tag`, and the UI has
a **Tag audit** button with tag and action fields. The weekly review JSON
includes `tag_audit`/`tag_audit_total`, and its Markdown
export adds a `## Tag audit` section, so archived reports keep the trail.

Priority can also be updated outside chat:

```powershell
atomzero-brief --session data/sessions/default.json `
  --set-task-priority 1 high
```

The HTTP form is `POST /v1/tasks/priority` with
`{"session_id": "default", "id": 1, "priority": "high"}`, and the UI has a
task ID / priority selector with a **Set priority** button. Invalid priorities
return `400`; missing tasks return `404`.

Tasks can be rescheduled with `reschedule task 1 due 2026-12-10`; the HTTP
endpoint is `POST /v1/tasks/reschedule` and the UI has a task ID/date form.
Tasks can be deleted only after explicit confirmation: chat previews
`delete task 1` and requires `confirm task delete`; the UI has a
**Confirm delete** checkbox and `POST /v1/tasks/delete` requires
`"confirm": true`.

Task text is searchable with `search tasks report`, `find task report` or
`task search report`. Search is case-insensitive and requires every query word
to appear in the task text; completed tasks are included with their status.

Filters are appended after the query: `search tasks report open`,
`search tasks report due week`, `search tasks report priority high`,
`search tasks report tag work` or
`search tasks report high open due overdue #work`. Status filters are `open`
and `done`; due filters are `overdue`, `today`, `week` and `none`; priority
filters are `low`, `normal` and `high`; tag filters use `tag work` or `#work`.
`week` means today through the next seven days, and `overdue` applies to open
tasks only.

The HTTP endpoint is
`GET /v1/tasks/search?q=report&status=open&due=week&priority=high&tag=work&sort=priority`,
the CLI form is `atomzero-brief --session ... --search-tasks report
--search-status open --search-due week --search-priority high
--search-tag work --task-sort priority [--json]` and the UI has a
**Search tasks** field with status, due, priority, tag and task-order
selectors. `priority` sorting places
high before normal before low, then earliest due date, then task id; the
default keeps session order. Chat uses the same option with
`search tasks report sort priority`, including combinations such as
`search tasks report high open sort priority`.

The daily brief and weekly review add overdue/due-soon counts, a focus item,
priority/due-date labels and dedicated Markdown sections. Chat understands
`overdue tasks`, `overdue tasks priority`, `due soon`, `due soon in 14 days`
and `due soon priority`. The UI **Overdue tasks** and **Due soon** buttons
expose the same session views; HTTP equivalents are
`/v1/tasks/overdue?sort=priority` and
`/v1/tasks/due-soon?days=7&sort=priority`. CLI equivalents are
`atomzero-brief --session ... --overdue --task-sort priority [--json]` and
`atomzero-brief --session ... --due-soon --due-days 7 --task-sort priority
[--json]`. The shared UI **Task order** selector applies to all three views.

Overdue and due-soon views also accept tag filters: CLI
`--overdue --overdue-tag work` and `--due-soon --due-soon-tag work`, HTTP
`tag=work`, chat `overdue tasks tag work` / `due soon tag work` and the UI tag
field next to **Due soon**. `open_task_count` and the rendered header are
scoped to matching open tasks, and text headers show `| tag=work`.

`--task-sort priority` also orders open, overdue and due-soon task lists in the
daily brief and weekly review. The HTTP JSON forms accept
`{"sort": "priority"}` on `POST /v1/briefing` and `POST /v1/weekly-review`;
the Markdown export accepts `GET /v1/weekly-review/export?sort=priority`.
Priority-sorted brief and review titles show `| sort=priority`.

Chat uses the same sort option: `brief me priority`, `daily brief priority`,
`weekly review priority` and `weekly review 14 days priority`. Plain
`brief me` / `weekly review` keep session order, and chat tool calls expose the
selected `sort` value.

The daily brief also accepts a tag filter: CLI `--brief-tag work`, HTTP
`POST /v1/briefing` with `{"tag": "work"}` or
`GET /v1/briefing?session_id=default&tag=work`, chat `brief me tag work`
(also `brief me priority tag work`) and the UI tag field next to **Get daily
brief**. It scopes open/overdue/due-soon tasks, tag counts and next actions to
matching tasks; gaps, index stats and graph stay global. The title shows
`| tag=work`, the text adds `Tags for #work`, and JSON includes `tag`, `tags`,
`tag_count`, `tagged_task_count` and `untagged_task_count`.

Recent tag delete/undo events are included too: the brief text adds
`Tag audit: N event(s)` with the last three events, and JSON includes
`tag_audit`/`tag_audit_total`. A `--brief-tag` filter also scopes the audit
events to that tag.

This is local metadata only: there are no background reminders, notifications
or calendar sync.

## Weekly review

```powershell
atomzero-brief --session data/sessions/default.json `
  --docs data/atomzero_demo.db --weekly --days 7
```

Add `--weekly-tag work` to scope the review to one tag.

```text
POST /v1/weekly-review
{"session_id": "default", "days": 7, "tag": "work"}
```

A weekly review aggregates:

- open tasks and tasks completed within the window;
- active tag counts (open tasks plus completions in the window);
- recent tag delete/undo audit events (last 10);
- undated completed tasks separately;
- pending gaps;
- sources indexed within the window;
- graph entities/connections and highlights;
- suggested next-week actions.

Completions carry `completed` timestamps when tasks are finished through
`ChatHuman`; older undated completions are listed separately rather than
silently dropped. The UI **Weekly review** button and adjacent tag field use
the same endpoint.

Tag counts cover open tasks plus tasks completed in the window. Tags on older
completions are excluded; untagged active tasks are counted separately. The
JSON review includes `tags`, `tag_count`, `tagged_task_count` and
`untagged_task_count`, and the Markdown export gains a Tags table.

With a tag filter, the task sections, counts and Tags table are limited to
matching tasks; the table also shows co-occurring tags on those tasks. Gaps,
recent sources and graph highlights remain global. The JSON review adds
`"tag"`, the title and Markdown heading show `tag=work` / `#work`, and the
archive records the tag so it can be searched and displayed in report listings.

Chat also runs the review directly:

```text
weekly review
weekly review for 14 days
weekly review tag work
weekly review 14 days tag work
haftalık özet
haftalık özet tag work
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
atomzero-brief --reports --query weekly
atomzero-brief --reports --tag work
atomzero-brief --reports --sort oldest
atomzero-brief --reports --sort largest
atomzero-brief --reports --verify
atomzero-brief --reports --verify --json
atomzero-brief --reports --repair
atomzero-brief --reports --repair --json
atomzero-brief --reports --compare w0001 w0002
atomzero-brief --reports --compare w0001 w0002 --json
atomzero-brief --reports --compare w0001 w0002 --markdown
atomzero-brief --reports --reports-index reports\weekly_reports.json --limit 5
```

The index defaults to `data/weekly_reports.json`. Each output path has one
entry, refreshed on later exports to the same path. Entries store the report
path, time, period, tag, task/source counts, graph counts, size and SHA-256.

`--prune` removes entries whose report file no longer exists and prints the
removed count. `--keep N` retains only the newest N entries and atomically
removes older ones. `--keep N --dry-run` lists what would be removed without
changing the index; the output ends with `Archive unchanged: N report(s).`
With `--json`, the result includes `removed`, `trimmed`, or
`would_remove_count`/`would_remove` fields as applicable.

`--delete ID` removes one archive entry atomically and leaves the report file
on disk. `--query TEXT` filters entries by ID, title, filename, path or date.
`--tag TAG` filters by the exact stored report tag (for example a report
created with `--weekly-tag work` has tag `work`); unlike `--query` it does not
match substrings. Tag filters combine with `--query`, `--limit` and `--sort`,
but not with maintenance actions. `--sort newest|oldest|largest` orders the
listing. `--verify` compares each
stored size and SHA-256 with the report file and reports
`ok`/`missing`/`size_mismatch`/`hash_mismatch`. `--repair` recomputes metadata
for existing changed files and writes the index atomically; missing files are
reported but not silently dropped. `--compare LEFT RIGHT` reports task, gap,
source, size and SHA-256 deltas.

```text
GET  /v1/weekly-review/reports?limit=10&sort=newest
GET  /v1/weekly-review/reports?query=weekly&sort=oldest
GET  /v1/weekly-review/reports?tag=work&limit=5
GET  /v1/weekly-review/reports?id=w0001
GET  /v1/weekly-review/reports/compare?left=w0001&right=w0002
GET  /v1/weekly-review/reports/compare/export?left=w0001&right=w0002
GET  /v1/weekly-review/reports/verify
POST /v1/weekly-review/reports/repair
POST /v1/weekly-review/reports/prune
POST /v1/weekly-review/reports/keep
POST /v1/weekly-review/reports/delete
{"keep": 10}
{"keep": 10, "dry_run": true}
{"id": "w0001"}
```

The UI **Recent reports** panel has a search box, a tag field and a
newest/oldest/largest sort selector, lists the archive, downloads an indexed
Markdown file, exposes **Delete** for one entry, **Prune missing** for deleted
files, **Verify files** for size/SHA-256 checks, **Repair metadata** for changed
files, a **Compare reports** form with **Download .md** / **Copy Markdown** and
a **Keep newest** retention control with **Preview only**. A dry-run response includes `would_remove`
entries and leaves the archive unchanged. Keep `--reports-index` pointed at
the same file for CLI and the server; both default to
`data/weekly_reports.json`. `quickstart` and `chat_server` accept
`--reports-index`.

## Chat flow

```text
brief me
-> daily brief in the chat: open tasks, pending gaps, index and graph

weekly review
-> weekly review; optionally "for 14 days" or "14 gün"

find report weekly
-> archived report matches with id, date, path and missing-file marker
-> Turkish: "rapor ara weekly"

find report tag work
list reports tag work
find report weekly tag work oldest
-> exact report-tag filter; combines with text query, sort and limit
-> the tool call and reply include the tag; invalid tags are rejected

list reports
list reports oldest
show report archive 5 largest
find report weekly largest
-> recent archived reports, optionally limited to N entries
-> sort by newest, oldest or largest

show report w0001
-> metadata plus Markdown content for one archived report
-> Turkish: "rapor göster w0001"

verify report archive
verify report w0001
-> size/SHA-256 integrity for all entries or one id
-> Turkish: "rapor arşivini doğrula", "rapor w0001 doğrula"

repair report archive
repair report w0001
-> preview of changed metadata and missing files
-> single-id form only touches that entry
-> requires "confirm reports"; Turkish: "rapor arşivini onar",
   "rapor w0001 onar"

compare reports w0001 w0002
-> task, gap, source, size and SHA-256 deltas

prune report archive
keep reports 2
keep reports 1 dry-run
delete report w0001
-> previews the entries affected
-> destructive keep/delete/prune changes require "confirm reports"
-> "cancel reports" discards the pending action
-> dry-run never changes the archive

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
