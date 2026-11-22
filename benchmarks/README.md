# Atom Zero benchmarks

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
| FTS | 0.960 | 1.000 | 0.922 | 0.786 | 0.974 |
| hybrid | **0.975** | 1.000 | **0.961** | **0.825** | **0.984** |

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
