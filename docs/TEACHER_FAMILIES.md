# Teacher task families

`research/build_diverse_teacher_queue.py` builds a verified teacher queue for
cross-distribution adapter experiments. The family table is importable without
loading a model, so validators, counts and splits can be tested on any machine.

## Families

| family | split | entity | keys | format | example answer |
|---|---|---|---|---|---|
| `person_json` | train | person | `name`, `score` | JSON | `{"name": "Ada", "score": 7}` |
| `product_json` | train | product | `product`, `price` | JSON | `{"product": "lamp", "price": 42}` |
| `person_kv` | train | person | `name`, `score` | key-value | `name=Ada;score=7` |
| `product_kv` | train | product | `product`, `price` | key-value | `product=lamp;price=42` |
| `arith_plain` | train | operand | `sum` | plain | `The sum of 12 and 30 is 42.` |
| `arith_json` | train | operand | `sum` | JSON | `{"sum": 42}` |
| `city_json` | cross | city | `city`, `population` | JSON | `{"city": "Ankara", "population": 500000}` |
| `city_kv` | cross | city | `city`, `population` | key-value | `city=Ankara;population=500000` |
| `city_yaml` | cross | city | `city`, `population` | YAML-like | `city: Ankara` + newline + `population: 500000` |

`TRAIN_FAMILIES` and `CROSS_FAMILIES` are explicit tuples. The split enforces
that no cross family can appear in train; an overlapping configuration raises
`ValueError`.

Arithmetic rows store the left operand as `entity` and the right operand as
`number`. The prompt asks the teacher to add them; the expected answer is the
computed sum. Attribute rows, in contrast, ask the teacher to transcribe the
given values into the requested format.

## Formats and validators

All validators are deterministic, standard-library only, and used through
`validate_answer(family, answer, entity, number)`.

| format | accepted | rejected |
|---|---|---|
| `json` | one JSON object with the required keys and values; an enclosing Markdown fence is stripped | arrays, missing or wrong keys, wrong values, prose around the object |
| `kv` | one `key1=value;key2=value` pair, whitespace tolerated | missing pairs, wrong values, JSON, prose |
| `yaml` | exactly two `key: value` lines with the required keys; order and optional quotes are tolerated | braces, lists, extra lines, missing keys, wrong values |
| `plain` | one sentence following `The sum of A and B is C` with an optional final period | a different sum, a different template, extra prose |

The YAML validator is a strict two-line subset. It is deliberately not a YAML
parser and does not depend on PyYAML.

## Reusable API

```python
from research.build_diverse_teacher_queue import (
    CROSS_FAMILIES, FAMILIES, TRAIN_FAMILIES,
    build_rows, oracle_answer, split_rows, validate_answer,
)

def stub_teacher(prompt: str) -> str:
    return '{"sum": 42}'

rows = build_rows(["arith_json"], stub_teacher, count=2, seed=17)
files = split_rows(rows)
```

`build_rows` accepts one callable that maps a prompt to raw text. Pass `None`
to use the deterministic oracle formatter. Counts can be uniform
(`count=...`) or per family (`counts={...}`). The function never imports
`torch` or `transformers`; those imports live inside the CLI model loader.

## Command line

```powershell
python research/build_diverse_teacher_queue.py `
  --out-dir data/diverse_teacher `
  --per-family 40 --city-per-family 25 `
  --city-yaml-count 25 --arith-count 20 `
  --seed 17
```

- `--per-family` controls the original train families.
- `--city-per-family` controls `city_json` and `city_kv`.
- `--city-yaml-count` controls `city_yaml`; it defaults to
  `--city-per-family`.
- `--arith-count` controls each arithmetic family.
- `--family-count FAMILY=COUNT` overrides one family and can be repeated.
- `--dry-run` uses the oracle formatter and does not load a model.

The output files are unchanged: `train.jsonl`, `indist.jsonl`, `cross.jsonl`
and `all.jsonl`. Train contains 80% of each verified train family, `indist`
contains the remaining 20%, `cross` contains every verified cross-family row,
and `all` contains every verified row.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m pytest tests/test_diverse_teacher_queue.py -q
```

The tests use stub teachers and require no GPU, CUDA, model files or network.

## Honest boundary

- A validator checks the requested format and values, not factual truth. For
  arithmetic it checks the sum; it does not inspect the teacher's reasoning.
- `city_yaml` changes both the entity family and the output format relative to
  the train families. A cross result therefore does not isolate format
  transfer from entity transfer.
- Generated city populations are synthetic, not census data. They are in a
  realistic range but are not facts.
- The real Qwen teacher is only exercised by the CLI. Dry-run rows show that
  plumbing works; they are not teacher-quality evidence.
