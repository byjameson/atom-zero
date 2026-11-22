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

Why both modes tie: the 20 selected contexts produce only 25 chunks, the
questions share vocabulary with the source paragraphs, and FTS already recovers
the right chunk. SQuAD is a useful public sanity check, not a hard semantic
retrieval benchmark. A harder paraphrase/no-answer benchmark and the optional
MiniLM reranker are the next steps.

This benchmark measures **retrieval**. It does not measure whether the 49.2M
model generates the answer or refuses a no-answer question; those are separate
product benchmarks.
