# Custom skills

Skills are the main way to shape Atom Zero without editing the core. A skill is
a small JSON file that matches trigger phrases, optionally retrieves local
context, and optionally calls the configured provider.

**Skills are declarative.** They cannot execute arbitrary code, open sockets or
touch the filesystem beyond the normal document index. For deeper changes,
Atom Zero is Apache-2.0 and can be forked; you are responsible for what you run.

## Install an example

```powershell
mkdir data\skills -Force
copy examples\skills\*.json data\skills\
atomzero-skills --dir data\skills list
atomzero-skills --dir data\skills validate
atomzero-skills --dir data\skills show daily-brief
atomzero-skills --dir data\skills run daily-brief "launch code" `
  --docs data\atomzero_demo.db
atomzero-skills --dir data\skills run daily-brief "launch code" `
  --docs data\atomzero_demo.db `
  --provider-url http://127.0.0.1:11434/v1 --provider-model llama3.2:3b
```

Restart the server after adding or changing files.

## Schema

```json
{
  "name": "daily-brief",
  "description": "Turn local notes into a short daily brief.",
  "triggers": ["daily brief", "brief me"],
  "system": "You are a concise local planning assistant.",
  "prompt": "Context:\n{context}\n\nRequest:\n{query}",
  "use_context": true,
  "max_chunks": 5,
  "provider": true,
  "enabled": true
}
```

| field | required | meaning |
|---|---|---|
| `name` | yes | `[A-Za-z0-9_-]`, max 64 chars |
| `description` | no | short UI description |
| `triggers` | yes | literal phrases, or regex with `re:` prefix |
| `prompt` | yes | supports `{query}` and `{context}` |
| `system` | no | prepended to the provider prompt |
| `use_context` | no | retrieve local chunks, default `true` |
| `max_chunks` | no | 1-12, default 4 |
| `provider` | no | call the configured provider, default `true` |
| `enabled` | no | default `true` |

The longest matching trigger wins. Trigger matching is case-insensitive.

## Modes

- **Provider skill** (`provider: true`): local context is retrieved and the
  rendered prompt is sent to the configured OpenAI-compatible endpoint.
  `atomzero-quickstart --provider` enables the default local Ollama backend.
- **Template skill** (`provider: false`): the rendered prompt/context is
  returned directly, without a model call. Useful for deterministic checklists.
- **Context-free skill** (`use_context: false`): the provider prompt is built
  from the request only.

If a provider skill is used without a configured provider, Atom Zero returns a
clear “needs a local provider” message and still shows the retrieved sources.

## Explicit name/query execution

Skills can run without a trigger:

```powershell
atomzero-skills run daily-brief "launch code" --docs data\atomzero_demo.db
```

```text
POST /v1/skills/run
{"name": "daily-brief", "query": "launch code"}
```

The UI **Custom skills** panel also has name/query fields and a **Run skill**
button. Responses include `provider_called`, `sources`, `evidence` and the
rendered `prompt`. Provider, template and no-provider fallback all use the same
`execute_skill` core.

## Skill chains

Run skills in order with explicit context passing:

```powershell
atomzero-skills --dir data\skills chain "launch code" `
  --skills study-tutor,brainstorm --docs data\atomzero_demo.db
```

```text
POST /v1/skills/chain
{"skills": ["study-tutor", "brainstorm"], "query": "launch code",
 "carry_context": true}
```

Semantics:

- later steps receive the previous reply as a labelled block, or via the
  `{previous}` placeholder if their prompt contains it;
- `carry_context: false` (or `--no-carry-context`) removes the hand-off;
- a chain supports at most six steps;
- a response includes every step, the final reply, merged sources/evidence and
  `provider_calls`;
- provider, template and no-provider fallback steps can be mixed.

The UI **Custom skills** panel has a chain field and **Run chain** button.

## Named chain recipes

Persist a chain under `data/skills/chains/<name>.json`:

```json
{
  "name": "research-brief",
  "description": "tutor then brainstorm",
  "steps": [
    "study-tutor",
    {"name": "brainstorm", "query": "improve this explanation: {previous}"}
  ],
  "carry_context": true
}
```

Each step can be a skill name or `{ "name": ..., "query": ... }`. The recipe
stores the default `carry_context`; a caller can override it.

CLI:

```powershell
atomzero-skills --dir data\skills recipes
atomzero-skills --dir data\skills recipe research-brief "warranty policy" `
  --docs data\atomzero_demo.db
```

HTTP:

```text
GET  /v1/skills/chains
POST /v1/skills/chains/run
{"name": "research-brief", "query": "warranty policy"}
```

The UI **Custom skills** panel loads named recipes into a select and has a
**Run recipe** button. The response includes `recipe`, every step, merged
sources/evidence and `provider_calls`.

## UI and API

- `GET /v1/skills` lists enabled skills and the skills directory.
- The UI **Custom skills** panel refreshes the list.
- A skill reply includes `tool_calls: [{"name": "skill", "skill": ...}]`,
  `sources`, `evidence` and the skill name.

## Responsibility

Skills can encode any prompt or workflow you write. That is the point: you can
shape the assistant for your own work, language, domain and provider. You are
responsible for the skills you add, the provider you point at and how you use
the output. Atom Zero does not phone home, does not impose a quota and does not
silently call a hosted service.

For changes beyond declarative skills, fork the Apache-2.0 source. The local
plugin/skill layer is intentionally not a hidden permission system.
