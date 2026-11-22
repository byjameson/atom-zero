# Atom Zero benchmarks

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
| FTS | 0.960 | 1.000 | 0.922 | 0.777 | 0.974 |
| hybrid | **0.975** | 1.000 | **0.961** | **0.816** | **0.984** |

The hybrid path wins on the larger subset but ties or loses on the smaller one.
SQuAD questions share vocabulary with their source paragraphs, so this is a
public retrieval sanity check, not a hard semantic benchmark. Raw result:
[`results/squad_retrieval_50x200.json`](results/squad_retrieval_50x200.json).

The heuristic confidence score is **not** an answerability classifier. On the
50-context subset the mean was 0.585 for answerable and 0.583 for no-answer
questions, because no-answer questions still share the paragraph vocabulary.
The product uses it only to flag a weak match.

This benchmark measures **retrieval**. It does not measure whether the 49.2M
model generates the answer; that is a separate product benchmark.
