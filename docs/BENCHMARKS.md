# Benchmarks and honest limits

Atom Zero is a **narrow local assistant**, not a general model. The benchmark
here measures product behavior: tools, memory, tasks, source-cited retrieval,
teacher-note review, missing-source detection and one short generation. It is
**not MMLU**, a coding benchmark or a general-reasoning score. The shareable
summary and model card are in [`MODEL_CARD.md`](MODEL_CARD.md).

## Public snapshot

| variant | raw knowledge | product | behavior | source-cited retrieval |
|---|---:|---:|---:|---:|
| base 49.2M | 2/10 | 21/21 | 16/16 | — |
| product 49.2M (40k) | 2/10 | 21/21 | 16/16 | — |
| + adapter `v0001` | 2/10 | 21/21 | 16/16 | — |
| + RAG/tools | — | 21/21 | 16/16 | top-1 1.000 |
| + optional ML safety | — | 26/26 | 16/16 | filter off by default |

Raw means no RAG, no tools and no agent. "Product" adds the runtime. The
adapter does not add factual knowledge; it is a style/skill layer with a
reviewed gate. See the [model card](MODEL_CARD.md) for the full distinction.

## Typed relation regression

`benchmarks/relations_regression.py` checks the served relations route, the
supported relation types, entity/relation filters, provenance, a control
sentence without a relation, missing-source behavior, direct/multi-hop relation
paths, the UI control and English/Turkish chat intents.

```powershell
python benchmarks/relations_regression.py
```

Current result: **12/12 checks pass**. Raw result:
[`results/relations_regression.json`](results/relations_regression.json).

## Citation gate policy simulation

`benchmarks/citation_gate_eval.py` evaluates the calibrated CitationGate on
SQuAD v2 (50 contexts, 200 questions: 103 answerable, 97 no-answer). Citation
precision counts released answers whose visible snippet contains the gold
answer. This is a distribution-shift simulation, not a production calibration
proof.

```powershell
python benchmarks/citation_gate_eval.py
```

| profile | threshold | answered / correct | no-answer refused | precision | overall |
|---|---:|---:|---:|---:|---:|
| authored_v1 | 0.5 | 69 / 64 | 34/97 | 0.485 | 0.490 |
| authored_v1 | 0.6 | 58 / 56 | 44/97 | 0.504 | 0.500 |
| **squad_v2_stress** | **0.5** | **92 / 82** | **18/97** | **0.479** | **0.500** |
| squad_v2_stress | 0.6 | 2 / 2 | 97/97 | 1.000 | 0.495 |

The default `squad_v2_stress` / `0.5` keeps most answerable questions but only
refuses 18/97 no-answer questions. Precision stays near the release-all
baseline. The `0.6` threshold is not portable: it refuses almost all no-answer
questions but also almost all answerable questions. A separate lexical
answerability gate was tested and rejected (context-grouped CV AUC 0.562); see
[`CITATION_GATE.md`](CITATION_GATE.md#answerability-is-not-a-separate-shipped-gate).
Target-distribution calibration is still required. Raw result:
[`results/citation_gate_eval.json`](results/citation_gate_eval.json).

## Public retrieval benchmark: SQuAD v2

`benchmarks/squad_retrieval.py` runs a public retrieval check on
`rajpurkar/squad_v2` validation. It reports context recall, answer recall and
whether the answer survives in the snippet the product actually shows.

Small subset (20 contexts, 100 questions: 50 answerable, 50 no-answer):

| mode | top-1 context | top-5 context | top-1 answer in chunk | top-1 answer in snippet | MRR |
|---|---:|---:|---:|---:|---:|
| FTS | **0.98** | 1.00 | 0.96 | 0.90 | **0.990** |
| hybrid | 0.96 | 1.00 | 0.96 | 0.90 | 0.978 |

Larger subset (50 contexts, 200 questions: 103 answerable, 97 no-answer):

| mode | top-1 context | top-5 context | top-1 answer in chunk | top-1 answer in snippet | MRR |
|---|---:|---:|---:|---:|---:|
| FTS | 0.960 | 1.000 | 0.922 | 0.845 | 0.974 |
| hybrid | **0.975** | 1.000 | **0.961** | **0.884** | **0.984** |

This is **retrieval**, not generation. SQuAD questions share vocabulary with
their contexts, so the result is a public sanity check, not a hard semantic
benchmark. Snippets score every contiguous window of up to four sentences and
keep the best query coverage plus a small length bonus, capped at 500
characters; this recovers answers that span sentence boundaries. The hybrid
path wins on the larger subset but ties or loses on the smaller one; the
difference is within the noise of a 100-question sample.

### Confidence is not answerability

The hybrid path also returns a heuristic confidence score. On the 50-context
subset:

| group | mean confidence | low-confidence rate |
|---|---:|---:|
| answerable questions | 0.585 | 0.0% |
| no-answer questions | 0.583 | 1.0% |

The score does **not** separate answerable from unanswerable questions here,
because SQuAD no-answer questions share vocabulary with the same paragraph.
It is a match-strength signal, not an answerability classifier; the product
uses it only to warn that a match is weak. A real answerability gate needs a
different model or a different benchmark. Raw results:
[`results/squad_retrieval.json`](results/squad_retrieval.json) and
[`results/squad_retrieval_50x200.json`](results/squad_retrieval_50x200.json).

Reproduce with:

```powershell
pip install datasets
python benchmarks/squad_retrieval.py `
  --max-contexts 20 --max-questions 100 --per-context 5 --seed 0
```

## Turkish morphology retrieval

`benchmarks/turkish_retrieval.py` is an authored Turkish check: queries use the
base form while target documents use inflected forms (`kitap` / `kitabın`,
`ağaç` / `ağacın`, `ev` / `evler`), and distractors share the attribute words.

| mode | before vowel drop | after |
|---|---:|---:|
| FTS | 0.700 | **0.950** |
| hybrid | 0.800 | **1.000** |

20 queries / 26 documents. The before column is the same query set run with
the prior suffix/consonant-mutation morphology (no vowel drop); recall probes
(`Şehir?`, `Burun?`, `Ağız?`, `İsim?`) exercise `şehir` / `şehrin` and
`burun` / `burnu`. Raw result:
[`results/turkish_retrieval.json`](results/turkish_retrieval.json).

## Source-comparison regression

`benchmarks/contradiction_regression.py` checks the served compare controls,
numeric/negation conflicts, identifier conflicts, formatted-number, date and
unit equivalence (`$1,500,000` vs `1.5 million`, `5 January 2020` vs
`January 5, 2020`, `24 months` vs `2 years`), magnitude/date/unit conflicts,
topic-paraphrase alignment (`24 months` vs `3 years` under the same title) and
no-conflict controls through the real handler.

```powershell
python benchmarks/contradiction_regression.py
```

Current result: **16/16 checks pass**. Raw result:
[`results/contradiction_regression.json`](results/contradiction_regression.json).

`benchmarks/compare_chat_regression.py` checks the chat intent, both snippets,
tool-call payload, sources, no-conflict behavior and missing-source honesty.

```powershell
python benchmarks/compare_chat_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/compare_chat_regression.json`](results/compare_chat_regression.json).

## Document timeline regression

`benchmarks/timeline_regression.py` checks the served timeline route, sorted
dates, source filtering, entity filtering, year-only precision, limits,
missing-source behavior, the UI timeline control and English/Turkish chat
intents.

```powershell
python benchmarks/timeline_regression.py
```

Current result: **10/10 checks pass**. Raw result:
[`results/timeline_regression.json`](results/timeline_regression.json).

## Weekly-review regression

`benchmarks/weekly_review_regression.py` checks the served weekly-review
control, then adds/completes tasks, requests `/v1/weekly-review`, and verifies
task counts, gaps, recent sources, graph entities, text and no-weight behavior.

```powershell
python benchmarks/weekly_review_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/weekly_review_regression.json`](results/weekly_review_regression.json).

`benchmarks/weekly_chat_regression.py` checks the chat intent, task counts,
text sections, sources, graph and the 14-day Turkish window.

```powershell
python benchmarks/weekly_chat_regression.py
```

Current result: **6/6 checks pass**. Raw result:
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

Current result: **29/29 checks pass**. Raw result:
[`results/weekly_export_regression.json`](results/weekly_export_regression.json).

## Graph-path regression

`benchmarks/graph_path_regression.py` checks the UI path controls, the
multi-hop `/v1/graph/path` result, chat intent, evidence sources and the
missing-entity path.

```powershell
python benchmarks/graph_path_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/graph_path_regression.json`](results/graph_path_regression.json).

## Scoped retrieval and summary regression

`benchmarks/scoped_regression.py` is a deterministic regression gate for the
document workflow, separate from the public SQuAD retrieval check. It builds a
temporary corpus with normal notes, a page/section locator and a teacher note,
then checks source/prefix/kind scoping, no-answer behavior, summary locators,
collection boundaries and exact code identifiers (`L-7781` vs `W-7781`).

```powershell
python benchmarks/scoped_regression.py
```

Current result: **10/10 checks pass**. Raw result:
[`results/scoped_regression.json`](results/scoped_regression.json).

## Custom skills end-to-end regression

`benchmarks/skills_flow_regression.py` exercises the real HTTP chat handler:
provider mode renders retrieved context and calls the provider; template mode
does not call it; the no-provider fallback is explicit and still returns local
sources.

```powershell
python benchmarks/skills_flow_regression.py
```

Current result: **9/9 checks pass**. Raw result:
[`results/skills_flow_regression.json`](results/skills_flow_regression.json).

## Named skill-recipe regression

`benchmarks/skills_recipe_regression.py` loads a recipe from
`data/skills/chains`, lists it over HTTP, runs it by name/query, checks the
provider -> template hand-off, sources, no-carry override and unknown-recipe
404.

```powershell
python benchmarks/skills_recipe_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/skills_recipe_regression.json`](results/skills_recipe_regression.json).

## Named recipe UI regression

`benchmarks/skills_recipe_ui_regression.py` checks the served recipe controls
and request contract, then runs the exact UI request through the real handler:
recipe list, context hand-off, metadata, carry override and unknown-recipe 404.

```powershell
python benchmarks/skills_recipe_ui_regression.py
```

Current result: **7/7 checks pass**. Raw result:
[`results/skills_recipe_ui_regression.json`](results/skills_recipe_ui_regression.json).

## Skill-chain UI regression

`benchmarks/skills_chain_ui_regression.py` checks the served UI chain controls
and then executes the same request through the real HTTP handler: context
hand-off, steps, sources and no-carry behavior.

```powershell
python benchmarks/skills_chain_ui_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/skills_chain_ui_regression.json`](results/skills_chain_ui_regression.json).

## Skill-chain regression

`benchmarks/skills_chain_regression.py` calls `POST /v1/skills/chain` and checks
the provider -> template hand-off, source merging, per-step provider flags,
no-carry mode and the unknown-skill 400 path.

```powershell
python benchmarks/skills_chain_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/skills_chain_regression.json`](results/skills_chain_regression.json).

## Explicit skill-run regression

`benchmarks/skills_run_regression.py` calls `POST /v1/skills/run` with an
explicit name/query: provider mode calls the provider, template mode does not,
fallback is explicit, and an unknown skill returns 404.

```powershell
python benchmarks/skills_run_regression.py
```

Current result: **9/9 checks pass**. Raw result:
[`results/skills_run_regression.json`](results/skills_run_regression.json).

## `brief me` end-to-end regression

`benchmarks/brief_me_flow_regression.py` exercises the real HTTP chat handler:
it adds a task, asks `brief me`, and verifies the task, pending gap, index
counts, graph entity, briefing tool call and source items.

```powershell
python benchmarks/brief_me_flow_regression.py
```

Current result: **7/7 checks pass**. Raw result:
[`results/brief_me_flow_regression.json`](results/brief_me_flow_regression.json).

## Goal-flow end-to-end regression

`benchmarks/goal_flow_regression.py` exercises the real HTTP chat handler:
plan creates no tasks, one confirmation creates exactly three, a second
confirmation is a no-op, and cancel leaves memory unchanged.

```powershell
python benchmarks/goal_flow_regression.py
```

Current result: **6/6 checks pass**. Raw result:
[`results/goal_flow_regression.json`](results/goal_flow_regression.json).

## Briefing and goal-plan regression

`benchmarks/briefing_regression.py` is a deterministic gate for the local
daily brief and goal planner: task/gap counts, focus/actions, index and graph
summary, cited goal context, no hidden task mutation, honest empty cases and
missing-session handling.

```powershell
python benchmarks/briefing_regression.py
```

Current result: **9/9 checks pass**. Raw result:
[`results/briefing_regression.json`](results/briefing_regression.json).

## Product benchmark

```bash
python research/benchmark_assistant.py \
  --model checkpoints/transformer_behavior_50m_long \
  --out research/results/assistant_benchmark_base.json

python research/benchmark_assistant.py \
  --model checkpoints/transformer_behavior_50m_long \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/assistant_benchmark_adapter.json
```

The benchmark creates a temporary document index and a temporary gap store, so
it does not touch personal notes. It runs 21 checks:

- greeting, help/limits, arithmetic, time
- name and likes memory
- add/list/complete tasks and empty-task list
- source-cited retrieval from five documents
- unverified and verified teacher notes
- missing-source gap detection
- one short generation without role-tag leakage

With `--safety-filter` it adds five safety checks and runs 26 in total:

- violence, self-harm and sexual-minors requests are refused
- benign child-abuse reporting and Wi-Fi security questions are not blocked

### Result on 2026-11-21

| model | score |
|---|---|
| 49.2M base | **21/21 (100%)** |
| 49.2M + `adapters_50m/v0001` | **21/21 (100%)** |
| 49.2M + safety filter | **26/26 (100%)** |

The base and the product adapter tie on these deterministic behaviors. That is
expected: the adapter was trained for a narrow skill queue, not to replace the
runtime tools. The adapter's value is shown by the loss gates and holdout, not
by this table. The hybrid neural path keeps the same gates: product **21/21**
and safety **26/26** (`assistant_benchmark_scaled_40k_hybrid_neural*.json`);
the hashed fallback is also 21/21 and 26/26.

## Retrieval reranker benchmark

`research/benchmark_retrieval.py` builds a 20-document authored office corpus
and asks 20 paraphrased questions in a temporary database. No reranker is
trained on these pairs. It reports top-1, top-3, MRR, citation accuracy
(top-1 equals the expected source) and full retrieval latency.

```bash
python research/benchmark_retrieval.py \
  --neural-model models/all-MiniLM-L6-v2 \
  --out research/results/retrieval_benchmark_hybrid.json
```

| candidate stage + reranker | top-1 | top-3 | MRR | citation | median latency |
|---|---:|---:|---:|---:|---:|
| raw FTS | 0.850 | 0.900 | 0.875 | 0.850 | 0.34 ms |
| FTS + hashed pool | 0.850 | 0.900 | 0.875 | 0.850 | 1.36 ms |
| FTS + neural pool | **1.000** | **1.000** | **1.000** | **1.000** | 8.8 ms |

Candidate-pool recall@10:

| stage | recall@10 |
|---|---:|
| FTS only | 0.900 |
| FTS + MiniLM neural pool | **1.000** |

The two misses in the previous FTS-only run were "Why are my documents printing
pale?" (printer) and "What should I wear to a customer meeting?" (dress code).
Both are absent from FTS, so no reranker could recover them. The neural
candidate pool supplies both, and the reranker puts them at rank 1. The hashed
pool stays FTS-anchored: it did not answer "population of Paris" from hash
collisions, and it refuses semantic-only queries rather than risking a false
answer. That is deliberate: the dependency-free fallback is conservative, and
the neural path is the promoted hybrid retriever.

The neural path is optional and local:

- `--reranker auto` (server default) uses the local model when
  `models/all-MiniLM-L6-v2` exists and falls back to hashed n-grams.
- `--reranker hashed` is the dependency-free path.
- `--reranker neural` fails loudly if the model is missing.

```bash
python -c "from huggingface_hub import snapshot_download; snapshot_download('sentence-transformers/all-MiniLM-L6-v2', local_dir='models/all-MiniLM-L6-v2')"
```

`models/` is gitignored. The reranker is retrieval only; it does not change the
49.2M core model. CPU latency is about 8.8 ms per query here after the vector
index is warm, with a few seconds of model warm-up. Chunk vectors are cached in
SQLite (`doc_vectors`) by model key and pruned when a source is replaced or
forgotten. See [`PRODUCT_AUDIT.md`](PRODUCT_AUDIT.md) for the remaining gaps:
the hashed fallback is still lexical, the ANN index is IVF rather than HNSW,
and multi-document conflict resolution is not implemented.

## Document and citation benchmark (100 questions)

`research/benchmark_documents.py` generates a deterministic 10-document corpus
in a temporary directory: Markdown, PDF, DOCX and HTML. It asks **80 answerable
questions** with an expected source and page/section, plus **20 questions with
no local source**. The corpus is authored; this is a regression benchmark, not
a public leaderboard.

```bash
python research/benchmark_documents.py \
  --out research/results/document_benchmark.json --check
```

| metric | result |
|---|---:|
| top-1 source | **1.000** (80/80) |
| top-1 page/section locator | **1.000** (80/80) |
| top-1 contains expected answer | **1.000** (80/80) |
| complete citation (source + locator + answer) | **1.000** (80/80) |
| no-answer refusal | **1.000** (20/20) |
| median retrieval latency | **~1-2 ms** |
| p95 retrieval latency | **~2-3 ms** |

By format: Markdown 24/24, PDF 24/24, DOCX 16/16, HTML 16/16 for source,
locator and answer. The `--check` mode enforces source ≥ 0.95, locator ≥ 0.90,
answer ≥ 0.90, complete citation ≥ 0.90, refusal ≥ 0.95 and median latency
≤ 100 ms. CI runs the same check.

The PDF path uses page numbers; DOCX/HTML/Markdown use heading sections. The
corpus is small and the questions are authored, so this does not measure open
domain robustness. It does catch regressions in ingestion, provenance, source
routing and the no-answer path. Results:
[`document_benchmark.json`](../research/results/document_benchmark.json).

### Prewarmed exact vector index

`research/benchmark_vector_index.py` compares the legacy linear scan (read every
SQLite blob and loop in Python) with a one-time numpy matrix built at startup.
It uses hashed 512-dim vectors, so the benchmark needs no neural model.

```bash
python research/benchmark_vector_index.py \
  --chunks 3000 --repeats 20 \
  --out research/results/vector_index_benchmark.json
```

| chunks | prewarm | matrix query | linear scan | speedup | top-5 overlap |
|---:|---:|---:|---:|---:|---:|
| 3000 | 747 ms | **0.77 ms** | 127.6 ms | **166x** | **1.000** |

Both paths are exact cosine over the same vectors; the top-5 sources match.
Full `search` (FTS + matrix + rerank) is 6.9 ms at this size. The matrix is
about 6 MB for 3000 hashed chunks. The server builds it after loading the model
and reranker; `--no-prewarm` skips it for very large corpora or slow disks.

### IVF ANN index

Above `ANN_MIN_CHUNKS` (256), auto mode builds deterministic k-means IVF
buckets. A query scores centroids, probes the best `nprobe` buckets and exactly
reranks only their members. `--vector-index ann|exact|auto` selects the path.
Anything below the auto threshold, an empty/short bucket result, an ANN error,
or an ANN top score below the hybrid threshold falls back to the exact matrix.

```bash
python research/benchmark_ann_index.py \
  --vectors 20000 --clusters 128 --probes 4,8,12 \
  --out research/results/ann_index_benchmark.json
```

Clustered synthetic normalized vectors, 20k vectors, 128 buckets, build 0.24 s:

| probe | ANN median | exact median | speedup | recall@10 |
|---:|---:|---:|---:|---:|
| 4 | **0.64 ms** | 3.22 ms | **5.0x** | **1.000** |
| 8 | 1.43 ms | 3.22 ms | 2.3x | **1.000** |
| 12 | 1.90 ms | 3.22 ms | 1.7x | **1.000** |

This is approximate IVF, not an HNSW graph; the exact fallback is the safety
net. On the small authored retrieval set `--vector-index ann` keeps top-1 at
**1.000** (two low-confidence queries used the exact fallback). A live 300-doc
server test reported `mode=ann`, `nlist=17`, `nprobe=5`, `fallbacks=0` and
returned `CODE-137`/`CODE-299` with the right sources.

## Raw model knowledge probe

The product score is high because tools and RAG do the work. Without them, the
50M model is weak. This probe asks ten simple general questions directly, with
no DocMemory, no calculator and no agent:

```bash
python research/benchmark_raw.py \
  --model checkpoints/transformer_behavior_50m_long \
  --out research/results/raw_knowledge_base.json
```

| model | raw score |
|---|---|
| 49.2M base | **2/10 (20%)** |
| 49.2M product 40k | **2/10 (20%)** |
| 49.2M product 40k + adapter `v0001` | **2/10 (20%)** |

Examples:

```text
PASS What is the capital of France? -> The capital of France is Paris.
FAIL What is the capital of Turkey? -> The capital of Norway is the capital of Sea...
FAIL What is 2 + 2? -> To find 2 + 2, we need to find the total number of possible values...
FAIL Who wrote Hamlet? -> Mia
FAIL What is the speed of light? -> The speed of light is about 1.5 degrees.
```

This is the honest reason for the architecture: **facts belong in RAG, not in
the 50M weights**. It is also why the product does not claim general knowledge.
Results: [`raw_knowledge_base.json`](../research/results/raw_knowledge_base.json),
[`raw_knowledge_scaled_40k.json`](../research/results/raw_knowledge_scaled_40k.json)
and [`raw_knowledge_scaled_40k_adapter.json`](../research/results/raw_knowledge_scaled_40k_adapter.json).

## Teacher -> RAG learning demo

`research/teacher_learning_demo.py` asks a question with no local source, asks
the local Ollama teacher, stores the answer as an unverified teacher note, and
asks again:

```text
before:  I do not have a source for that yet...
teacher: The capital of New Zealand is Wellington.
after:   I found these notes:
         - ... Wellington. (teacher note, unverified: teacher://ollama/...)
weights_changed: false
```

This is **retrieval-memory learning**: the answer enters `DocMemory` as
unverified data and can be cited immediately. No gradient update happens, and
the model does not autonomously decide what to study. Evidence:
[`teacher_rag_learning.json`](../research/results/teacher_rag_learning.json).

The **curiosity queue** wraps this in a review gate: pending knowledge gaps are
proposed as `kind=curiosity` research items; nothing runs until approval. A live
run recorded `unapproved_run: 0`, `approved_run: done`, and
`weights_changed: false`:
[`curiosity_live.json`](../research/results/curiosity_live.json). See
[`CURIOSITY.md`](CURIOSITY.md).

## Data-scaling early result

The scaled 50M checkpoint at 10k steps was evaluated on the **new** scaled
validation shard and on the product behaviors:

| check | base | scaled 10k |
|---|---|---|
| scaled-valid chat bpb | 1.503 | **1.382** |
| scaled-valid assistant loss | 4.215 | **3.855** |
| scaled-valid stories bpb | 1.622 | **1.414** |
| product benchmark | 21/21 | **21/21** |
| safety benchmark | 26/26 | **26/26** |
| behavior checks | 16/16 | **16/16** |
| raw-model probe | 2/10 | 2/10 |

The 40k checkpoint is promoted as `checkpoints/transformer_behavior_50m_scaled`;
training stopped at 40k because 38k/40k plateaued. **Do not use the old clean
valid shard for this comparison:** the larger corpus was regenerated from the
same source datasets, so that shard is not a clean holdout for the new run.
Details: [`DATA_SCALING.md`](DATA_SCALING.md).

## Safety red-team

```bash
python research/redteam_safety.py --ml \
  --model checkpoints/transformer_behavior_50m_scaled \
  --adapter checkpoints/adapters_50m/v0001 \
  --out research/results/redteam_safety_ml_agent.json

python research/redteam_moderation.py \
  --out research/results/redteam_moderation.json
```

| check | rules only | ML hybrid |
|---|---:|---:|
| harmful/benign filter cases | **72/72** | **72/72** |
| output-filter cases | **4/4** | **4/4** |
| agent refusal path | **72/72** | **72/72** |
| benign false blocks | 0/26 | **0/26** |
| held-out semantic paraphrases | 1/10 | **10/10** |
| product benchmark with filter | **26/26** | **26/26** |

The hybrid filter is rule-first; the local MiniLM + linear head only adds
blocks. On the authored 72-case set ML-only is 61.1% (multilingual/category
cases remain rule-covered), but it blocks 10/10 held-out semantic paraphrases
the rules deliberately do not encode, with 0/26 benign false blocks. It is a
guardrail, not proof of general safety; see [`SAFETY.md`](SAFETY.md).

## Adapter loss/holdout evidence

```text
8-example gate:        target +16.74%
                       clean bpb +1.58%, assistant +1.54%
                       behavior and smoke PASS
6-train / 2-holdout:   base 3.853 -> adapter 3.139 (-18.5%)
```

Sources:

- [`gated_feedback_history_test.json`](../research/results/gated_feedback_history_test.json)
- [`holdout_feedback_test.json`](../research/results/holdout_feedback_test.json)
- [`adapter_comparison_feedback.json`](../research/results/adapter_comparison_feedback.json)

These are small-sample measurements. They show that the reviewed update loop
works; they do not make the model general. See [`ADAPTERS.md`](ADAPTERS.md) for
the external-trainer adapter format, local validation commands and the
`adapter_holdout.json` result.

Negative control: a 213-train/54-holdout format-skill queue was already known
to the base (holdout loss 0.224). A 100-step adapter regressed to 0.302
(+34.7%) and a 20-step adapter to 0.228 (+1.6%); both were rejected by the
local holdout comparison. Low training loss alone is not evidence; use a novel
per-domain queue and keep the holdout. Raw result:
[`results/adapter_targeted_holdout.json`](results/adapter_targeted_holdout.json).

Answerability was tested as a separate lexical/type classifier on the SQuAD v2
stress sample. Context-grouped 5-fold CV AUC was **0.562** (logistic) and
**0.497** (boosted trees). A later 42-feature + tiny-QA hybrid reached held-out
AUC **0.663**, Brier **0.244**, ECE **0.130**, but still missed the refusal
recall/precision and coverage targets at every threshold. The optional gate is
off by default; CitationGate remains the default refusal gate at 0.5. See
[`ANSWERABILITY.md`](ANSWERABILITY.md),
[`CITATION_GATE.md`](CITATION_GATE.md#answerability-is-a-separate-optional-signal),
and [`answerability_eval.py`](../benchmarks/answerability_eval.py).

## What is not measured

- MMLU/GSM8K/HumanEval or coding ability.
- Open-ended factual reasoning without sources.
- A large human-labelled moderation benchmark; the current sets are authored
  and small (72 rule cases, 10 semantic paraphrases, 26 benign controls).
- Long-context or multi-document synthesis at scale.
- Turkish or multilingual quality.

Do not claim a general benchmark score from the product benchmark. The 50M
model remains a narrow daily assistant; facts stay in RAG and skills stay in
gated adapters.
