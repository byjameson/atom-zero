# atomzero 0.5.0 - second brain and skill workflows

Status: **published to PyPI as `atomzero 0.5.0` on 2026-11-22.** No model or runtime
weights changed; the weights remain `0.8.0`.

This is the first batched release under the ~10-increment policy. It includes
the graph, skills, chains, recipes, daily brief and goal planner accumulated
after 0.4.2.

## Headline: second brain

- Local **co-occurrence graph** over capitalized phrases, code-like
  identifiers and years.
- Source counts on nodes/edges and evidence lookup for an entity.
- `atomzero-docs graph` / `entity`, `GET /v1/graph`,
  `GET /v1/graph/entity` and a UI graph panel.
- Chat relation queries: `what is related to Ankara?`
- Diverse search suppresses near-identical chunks.

The graph is statistical navigation, not a trained NER or semantic relation
model. An edge means “appeared together in a chunk”.

## Custom skills

- Declarative JSON skills under `data/skills`.
- Trigger phrases or explicit name/query execution.
- Provider, deterministic template and no-provider fallback modes.
- `atomzero-skills list/validate/show/run`.
- `GET /v1/skills`, `POST /v1/skills/run`.
- UI skill list, Run skill, Run chain and Run recipe controls.
- Skill chains with explicit `{previous}` hand-off and a no-carry mode.
- Named chain recipes under `data/skills/chains`:
  - `atomzero-skills recipes` / `recipe`;
  - `GET /v1/skills/chains`;
  - `POST /v1/skills/chains/run`.

Skills are declarative; they cannot execute arbitrary code. For deeper changes
the Apache-2.0 source can be forked.

## Daily brief and goal planner

- `atomzero-brief` and `POST /v1/briefing` combine open tasks, pending gaps,
  index stats and graph entities.
- Chat `brief me` / `daily brief` returns the same local brief.
- `POST /v1/goals/plan` returns cited local context and suggested tasks.
- Chat flow: `plan goal ...` → `confirm plan` → tasks; `cancel plan` discards.
- Task creation requires explicit confirmation; the initial plan has no side
  effects.

## End-to-end regression coverage

- scoped answer/summary regression (10/10);
- briefing/goal-plan regression (9/9);
- goal-flow HTTP plan/confirm/cancel (6/6);
- brief-me HTTP chat (7/7);
- skills flow/provider/template/fallback (9/9);
- explicit skill run (9/9);
- skill-chain HTTP (6/6) and UI contract (6/6);
- named recipe HTTP (6/6) and UI contract (7/7).

## Honest limits

- The graph is statistical co-occurrence; it does not label relationships.
- Skills are declarative and provider-dependent; answers inherit the provider's
  quality and limits.
- The goal planner is a deterministic plan skeleton, not autonomous execution.
- No model weights are changed. `weights_changed: false` is reported by the
  relevant endpoints.
- SQuAD retrieval remains at the 0.4.x level: hybrid top-1 context 0.975 and
  top-1 answer-in-visible-snippet 0.816 on the 50-context subset.
- 286 tests pass; wheel/sdist pass `twine check`.

## Install

```powershell
pip install atomzero==0.5.0
atomzero-doctor
atomzero-quickstart
```

Quick examples:

```powershell
atomzero-docs graph --db data/atomzero_demo.db
atomzero-brief --session data/sessions/default.json --docs data/atomzero_demo.db
atomzero-skills --dir data/skills recipes
atomzero-skills --dir data/skills recipe research-brief "warranty policy"
atomzero-hungry --dry-run --json
```

Chat:

```text
brief me
what is related to Ankara?
plan goal prepare the warranty page
confirm plan
```

## Support

- **EVM-compatible networks:** `0xa4dEc84Fe6F13c8A3b8707446D6892a6E272285c`
- **Solana:** `EDqs8w4JkS5vAAfxmsB7fAQwoauEYufYXvMWFW9aYPdc`
- **Bitcoin:** `bc1qsq4mdtf0dlfvppllqwdaknxrq6xkjgsrx6ctgm`

Full details: <https://github.com/byjameson/atom-zero/blob/main/docs/DONATE.md>

## Links

- PyPI: <https://pypi.org/project/atomzero/0.5.0/>
- GitHub: <https://github.com/byjameson/atom-zero>
- GitHub release: <https://github.com/byjameson/atom-zero/releases/tag/v0.5.0>
- Knowledge graph: <https://github.com/byjameson/atom-zero/blob/main/docs/KNOWLEDGE_GRAPH.md>
- Skills: <https://github.com/byjameson/atom-zero/blob/main/docs/SKILLS.md>
- Briefing: <https://github.com/byjameson/atom-zero/blob/main/docs/BRIEFING.md>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Contact: **gokhancava@gmail.com**
