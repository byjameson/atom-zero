# Atom Zero benchmarks

## Source-comparison regression

`benchmarks/contradiction_regression.py` checks the served compare controls,
numeric/negation conflicts, identifier conflicts, formatted-number, date and
unit equivalence (`$1,500,000` vs `1.5 million`, `5 January 2020` vs
`January 5, 2020`, `24 months` vs `2 years`), magnitude/date/unit conflicts
and a no-conflict comparison through the real handler.

```powershell
python benchmarks/contradiction_regression.py
```

Latest result: **11/11 checks pass**. Raw result:
[`results/contradiction_regression.json`](results/contradiction_regression.json).

`benchmarks/compare_chat_regression.py` checks the chat intent, both snippets,
tool-call payload, sources, no-conflict behavior and missing-source honesty.

```powershell
python benchmarks/compare_chat_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/compare_chat_regression.json`](results/compare_chat_regression.json).

## Weekly-review regression

`benchmarks/weekly_review_regression.py` checks the served weekly-review
control, then adds/completes tasks, requests `/v1/weekly-review`, and verifies
task counts, gaps, recent sources, graph entities, text and no-weight behavior.

```powershell
python benchmarks/weekly_review_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/weekly_review_regression.json`](results/weekly_review_regression.json).

`benchmarks/weekly_chat_regression.py` checks the chat intent, task counts,
text sections, sources, graph and the 14-day Turkish window.

```powershell
python benchmarks/weekly_chat_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/weekly_chat_regression.json`](results/weekly_chat_regression.json).

`benchmarks/weekly_export_regression.py` checks the UI export controls,
Markdown content type/filename, task checkboxes, gaps, sources, graph, next
actions, the 14-day window, invalid day input,
`atomzero-brief --markdown --out PATH` file output, the archive
listing/download flow, pruning, retention, dry-run preview, deletion, query
filtering, chat search/list/maintenance, chat/HTTP sorting, report viewing,
integrity verification, all/single-entry metadata repair, comparison and
comparison Markdown export.

```powershell
python benchmarks/weekly_export_regression.py
```

Latest result: **29/29 checks pass**. Raw result:
[`results/weekly_export_regression.json`](results/weekly_export_regression.json).

## Graph-path regression

`benchmarks/graph_path_regression.py` checks the UI path controls, the
multi-hop `/v1/graph/path` result, chat intent, evidence sources and the
missing-entity path.

```powershell
python benchmarks/graph_path_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/graph_path_regression.json`](results/graph_path_regression.json).

## Scoped answer and summary regression

`benchmarks/scoped_regression.py` is a deterministic product regression gate
for the document workflow. It builds a temporary corpus with normal notes, a
PDF-style page locator and a teacher note, then checks:

- source-scoped answers do not leak other documents;
- source-prefix and kind filters stay inside their scope;
- single-source summaries keep page/section locators;
- collection summaries include only the selected prefix;
- no-answer questions return nothing;
- exact code identifiers (`L-7781` vs `W-7781`) rank the right source;
- missing source/collection summaries raise `SourceNotFound`.

```powershell
python benchmarks/scoped_regression.py
```

Latest result: **10/10 checks pass**. Raw result:
[`results/scoped_regression.json`](results/scoped_regression.json).

## Custom skills end-to-end regression

`benchmarks/skills_flow_regression.py` starts the real HTTP chat handler and
checks three skill modes:

- provider skill: local context is rendered and the provider is called;
- template skill: no provider call, rendered prompt returned;
- no-provider fallback: explicit message plus local sources;
- `GET /v1/skills` lists the loaded skills.

```powershell
python benchmarks/skills_flow_regression.py
```

Latest result: **9/9 checks pass**. Raw result:
[`results/skills_flow_regression.json`](results/skills_flow_regression.json).

## Named skill-recipe regression

`benchmarks/skills_recipe_regression.py` loads a recipe from
`data/skills/chains`, lists it over HTTP, runs it by name/query, checks
provider -> template hand-off, sources, no-carry override and the unknown
recipe 404.

```powershell
python benchmarks/skills_recipe_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/skills_recipe_regression.json`](results/skills_recipe_regression.json).

## Named recipe UI regression

`benchmarks/skills_recipe_ui_regression.py` checks the served recipe select,
Run recipe button and request contract, then runs the exact UI request through
the real handler: recipe list, context hand-off, metadata, carry override and
unknown-recipe 404.

```powershell
python benchmarks/skills_recipe_ui_regression.py
```

Latest result: **7/7 checks pass**. Raw result:
[`results/skills_recipe_ui_regression.json`](results/skills_recipe_ui_regression.json).

## Skill-chain UI regression

`benchmarks/skills_chain_ui_regression.py` fetches the served page, checks the
chain controls/request contract, then executes the exact UI request through the
real handler: context hand-off, step metadata, sources and no-carry behavior.

```powershell
python benchmarks/skills_chain_ui_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/skills_chain_ui_regression.json`](results/skills_chain_ui_regression.json).

## Skill-chain regression

`benchmarks/skills_chain_regression.py` calls `POST /v1/skills/chain` and
checks provider -> template hand-off, source merging, per-step provider flags,
no-carry mode and the unknown-skill 400 path.

```powershell
python benchmarks/skills_chain_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/skills_chain_regression.json`](results/skills_chain_regression.json).

## Explicit skill-run regression

`benchmarks/skills_run_regression.py` calls `POST /v1/skills/run` by explicit
name/query and verifies provider mode, template mode, no-provider fallback and
the unknown-skill 404 path.

```powershell
python benchmarks/skills_run_regression.py
```

Latest result: **9/9 checks pass**. Raw result:
[`results/skills_run_regression.json`](results/skills_run_regression.json).

## `brief me` end-to-end regression

`benchmarks/brief_me_flow_regression.py` starts the real HTTP chat handler,
adds a task, then checks the `brief me` response for the open task, a pending
gap, document counts, graph entities, the briefing tool call and source items.

```powershell
python benchmarks/brief_me_flow_regression.py
```

Latest result: **7/7 checks pass**. Raw result:
[`results/brief_me_flow_regression.json`](results/brief_me_flow_regression.json).

## Goal-flow end-to-end regression

`benchmarks/goal_flow_regression.py` starts the real HTTP chat handler and
checks the full confirmation contract:

- `plan goal ...` creates no tasks;
- the plan cites local evidence and asks for confirmation;
- one `confirm plan` creates exactly three tasks;
- a second confirmation is a no-op;
- `cancel plan` leaves memory unchanged;
- confirming after cancel is a no-op.

```powershell
python benchmarks/goal_flow_regression.py
```

Latest result: **6/6 checks pass**. Raw result:
[`results/goal_flow_regression.json`](results/goal_flow_regression.json).

## Briefing and goal-plan regression

`benchmarks/briefing_regression.py` checks the daily brief and goal planner
without a model:

- open-task and pending-gap counts;
- focus selection and next actions;
- index and graph summaries;
- cited goal context and suggested tasks;
- plans do not mutate session tasks;
- empty/no-doc plans remain honest;
- missing session files are empty, not errors.

```powershell
python benchmarks/briefing_regression.py
```

Latest result: **9/9 checks pass**. Raw result:
[`results/briefing_regression.json`](results/briefing_regression.json).

## SQuAD v2 retrieval

This is a **public retrieval benchmark**, not a generation benchmark. It uses
the `rajpurkar/squad_v2` validation split and reports whether the answer
survives in the snippet the product actually shows.

```powershell
pip install datasets
python benchmarks/squad_retrieval.py `
  --max-contexts 20 --max-questions 100 --per-context 5 --seed 0
```

Small subset (20 contexts, 100 questions: 50 answerable, 50 no-answer):

| mode | top-1 context | top-5 context | top-1 answer in chunk | top-1 answer in snippet | MRR |
|---|---:|---:|---:|---:|---:|
| FTS | **0.98** | 1.00 | 0.96 | 0.84 | **0.990** |
| hybrid | 0.96 | 1.00 | 0.96 | 0.84 | 0.978 |

The raw result is in
[`results/squad_retrieval.json`](results/squad_retrieval.json).

Larger subset (50 contexts, 200 questions: 103 answerable, 97 no-answer):

| mode | top-1 context | top-5 context | top-1 answer in chunk | top-1 answer in snippet | MRR |
|---|---:|---:|---:|---:|---:|
| FTS | 0.960 | 1.000 | 0.922 | 0.845 | 0.974 |
| hybrid | **0.975** | 1.000 | **0.961** | **0.884** | **0.984** |

The hybrid path wins on the larger subset but ties or loses on the smaller one.
SQuAD questions share vocabulary with their source paragraphs, so this is a
public retrieval sanity check, not a hard semantic benchmark. Snippets score
contiguous sentence windows (up to four sentences, capped at 500 characters)
instead of always expanding around one local maximum. Raw result:
[`results/squad_retrieval_50x200.json`](results/squad_retrieval_50x200.json).

The heuristic confidence score is **not** an answerability classifier. On the
50-context subset the mean was 0.585 for answerable and 0.583 for no-answer
questions, because no-answer questions still share the paragraph vocabulary.
The product uses it only to flag a weak match.

This benchmark measures **retrieval**. It does not measure whether the 49.2M
model generates the answer; that is a separate product benchmark.

## Turkish morphology retrieval

This is an **authored** Turkish retrieval check, not a public benchmark. Queries
use the base form (`kitap`, `ağaç`, `ev`) while target documents use an
inflected form (`kitabın`, `ağacın`, `evler`); distractor documents share the
attribute words so the entity term decides the ranking.

```powershell
python benchmarks/turkish_retrieval.py
```

| mode | before morphology | after morphology |
|---|---:|---:|
| FTS | 0.583 | **0.833** |
| hybrid | 0.833 | **1.000** |

12 queries over 18 documents. The "before" column was measured with the
pre-morphology code on the same queries. Raw result:
[`results/turkish_retrieval.json`](results/turkish_retrieval.json).
