# atomzero 0.6.0 - weekly review and task workflow

Status: **prepared for publication; publishing is pending fresh PyPI,
Hugging Face, ModelScope and dev.to credentials plus GitHub release/tag
steps.** No model or runtime weights changed; the weights remain `0.8.0`.

This is the second batched release under the ~10-increment policy. It covers
the weekly-review, task-workflow, export and source-comparison work accumulated
after `0.5.0`.

## Weekly review and report archive

- `atomzero-brief --weekly --days N`, `POST /v1/weekly-review` and chat
  `weekly review` / `haftalık özet`.
- Open/recently completed tasks, pending gaps, sources indexed in the window,
  graph highlights and next actions.
- Active tag counts across open and recently completed tasks, plus a Tags table
  in Markdown exports.
- Optional weekly-review tag filters through `--weekly-tag`, HTTP/UI `tag` and
  chat `weekly review tag work`, with scoped counts and archive metadata.
- Optional daily-brief tag filters through `--brief-tag`, GET/POST `/v1/briefing`
  `tag`, chat `brief me tag work` and a UI field, with scoped tasks and counts.
- Optional tag filters for overdue and due-soon views through
  `--overdue-tag` / `--due-soon-tag`, HTTP/UI `tag` and chat commands.
- Weekly Markdown export through `--weekly --markdown [--out PATH]`,
  `GET /v1/weekly-review/export` and a UI download.
- Local report archive with report metadata, size and SHA-256.
- Archive list/download/search/sort, retention, dry-run preview, deletion,
  integrity verification, metadata repair and report comparison.
- Exact report-tag filtering through `--reports --tag`, HTTP/UI `tag`.
- Chat archive commands accept report tags (`find report tag work`,
  `list reports tag work`) with `tag` in the structured tool call.
- Priority-first brief/review sorting through `--task-sort priority`,
  HTTP `sort=priority`, chat `brief me priority` / `weekly review priority`
  and a shared UI selector.

## Local task workflow

- Task due dates (`due YYYY-MM-DD`, `today`, `tomorrow`, `in N days`) with
  overdue and due-soon views.
- Task priorities (`low`, `normal`, `high`), priority updates and priority
  labels in lists, briefs, reviews and exports.
- Task tags (up to five per task), `tag task` updates, tag filters in task
  lists/search, tags in CSV/Markdown task exports and a `list tags` inventory
  with open/done counts plus clickable UI chips.
- Confirmed global tag rename in chat, CLI, HTTP and UI; renaming preserves
  other tags and deduplicates.
- Confirmed global tag delete in chat, CLI, HTTP and UI; deleting preserves
  other tags and can leave a task untagged.
- Tag parity in the legacy `research/chat_human.py` runtime: add/tag/list
  filters, inventory and confirmed rename with persistence.
- Rescheduling and confirmed deletion.
- Task text search plus status/due/priority filters.
- Priority-first sorting in search, overdue, due-soon, list, brief and review
  views.
- Chat `list tasks` filters with a structured `list_tasks` tool call.
- `GET /v1/tasks/list` and UI **Task list** controls using the same view.
- Filtered task-list export as CSV or Markdown:
  `atomzero-brief --task-list --task-format csv|markdown`,
  `GET /v1/tasks/export` and UI Download CSV/.md buttons.
- Task tags are included as a `tags` column in CSV and Markdown exports.
- Optional AISC architecture specs and an audit script for the critical read
  and task-action paths; AISC remains an external, optional tool.

## Source comparison and reliability

- Deterministic source comparison via `atomzero-docs compare`,
  `GET /v1/documents/compare`, UI Compare sources and chat
  `compare sources A and B`.
- Flags numeric, identifier and negation conflicts with snippets and locators.
- Multi-hop graph paths through `atomzero-docs path`, `GET /v1/graph/path`,
  chat and UI.
- BOM-tolerant session, feedback, gap, queue, skill and recipe JSON reads.
- Atomic writes for reports and session priority updates.

## Verified regressions

- `378 passed` in the assistant test suite.
- Weekly chat regression: 7/7.
- Weekly export/archive regression: 29/29.
- Weekly review regression: 6/6.
- Graph-path regression: 6/6.
- Contradiction regression: 5/5.
- Source-comparison chat regression: 6/6.
- Product benchmark remains 21/21; raw-model probe remains 2/10.
- Document benchmark remains 80/80 answerable and 20/20 no-answer.
- No model weights were changed.

## Install

```powershell
pip install atomzero==0.6.0
atomzero-quickstart
```

## Honest limits

- Weekly reviews and task lists are local, deterministic summaries; they do
  not create background reminders or calendar events.
- Task due dates and priorities are local metadata only.
- CSV/Markdown task exports reflect local session data; they do not sync to a
  remote account.
- The graph remains statistical co-occurrence, not semantic NER.
- Source comparison is deterministic and rule-based; it is not a full
  contradiction-resolution model.
- The PyPI, Hugging Face, ModelScope and dev.to updates, together with the
  GitHub tag/release, still require the owner's fresh credentials.

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.6.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.6.0>
- Weekly review: <https://github.com/byjameson/atom-zero/blob/main/docs/BRIEFING.md>
- Tasks and exports: <https://github.com/byjameson/atom-zero/blob/main/docs/BRIEFING.md>
- AISC audit: <https://github.com/byjameson/atom-zero/blob/main/docs/AISC_AUDIT.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
