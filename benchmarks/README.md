# Atom Zero benchmarks

## SQuAD v2 retrieval

This is a **public retrieval benchmark**, not a generation benchmark. It uses
the `rajpurkar/squad_v2` validation split: 20 contexts, 100 questions
(50 answerable, 50 no-answer), up to 5 questions per context.

```powershell
pip install datasets
python benchmarks/squad_retrieval.py `
  --max-contexts 20 --max-questions 100 --per-context 5 --seed 0
```

Result on 2026-11-22:

| mode | top-1 context | top-5 context | top-1 answer in chunk | top-5 answer in chunk | MRR |
|---|---:|---:|---:|---:|---:|
| FTS | 0.98 | 1.00 | 0.96 | 1.00 | 0.99 |
| hybrid | 0.98 | 1.00 | 0.96 | 1.00 | 0.99 |

The raw result is in
[`results/squad_retrieval.json`](results/squad_retrieval.json).

A larger subset (50 contexts, 200 questions: 103 answerable, 97 no-answer):

| mode | top-1 context | top-5 context | top-1 answer in chunk | top-5 answer in chunk | MRR |
|---|---:|---:|---:|---:|---:|
| FTS | 0.960 | 1.000 | 0.922 | 1.000 | 0.974 |
| hybrid | **0.970** | 1.000 | **0.951** | 1.000 | **0.980** |

The hybrid path is slightly better on the harder subset, but both modes remain
high because SQuAD questions share vocabulary with their source paragraphs. This
is a public retrieval sanity check, not a hard semantic benchmark. A harder
paraphrase/no-answer benchmark and the optional MiniLM reranker are the next
steps. Raw result:
[`results/squad_retrieval_50x200.json`](results/squad_retrieval_50x200.json).

This benchmark measures **retrieval**. It does not measure whether the 49.2M
model generates the answer or refuses a no-answer question; those are separate
product benchmarks.
