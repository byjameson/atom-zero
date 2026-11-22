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
treated as empty, not as errors.

## Chat flow

```text
brief me
-> daily brief in the chat: open tasks, pending gaps, index and graph

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
