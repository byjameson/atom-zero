# Teacher task families

`research/build_diverse_teacher_queue.py` builds a verified teacher queue for
cross-distribution adapter experiments. The family table is importable without
loading a model, so validators, counts and splits can be tested on any machine.

## Families

| family | split | entity | keys | format | example answer |
|---|---|---|---|---|---|
| `person_json` | train | person | `name`, `score` | JSON | `{"name":"Ada","score":7}` |
| `product_json` | train | product | `product`, `price` | JSON | `{"product":"lamp","price":42}` |
| `person_kv` | train | person | `name`, `score` | key-value | `name=Ada;score=7` |
| `product_kv` | train | product | `product`, `price` | key-value | `product=lamp;price=42` |
| `person_yaml` | train | person | `name`, `score` | YAML-like | `name: Ada` + newline + `score: 7` |
| `product_yaml` | train | product | `product`, `price` | YAML-like | `product: lamp` + newline + `price: 42` |
| `arith_plain` | train | operand | `sum` | plain | `The sum of 12 and 30 is 42.` |
| `arith_json` | train | operand | `sum` | JSON | `{"sum":42}` |
| `city_json` | cross | city | `city`, `population` | JSON | `{"city":"Ankara","population":500000}` |
| `city_kv` | cross | city | `city`, `population` | key-value | `city=Ankara;population=500000` |
| `city_yaml` | cross | city | `city`, `population` | YAML-like | `city: Ankara` + newline + `population: 500000` |

`TRAIN_FAMILIES` and `CROSS_FAMILIES` are explicit tuples. The split enforces
that no cross family can appear in train; an overlapping configuration raises
`ValueError`. Person and product YAML families are train families, so YAML is
seen before `city_yaml` is evaluated.

Arithmetic rows store the left operand as `entity` and the right operand as
`number`. The prompt asks the teacher to add them; the expected answer is the
computed sum. Attribute rows, in contrast, ask the teacher to transcribe the
given values into the requested format.

## Copy-generalization key pairs

`KEY_PAIRS` contains twelve key pairs:

```
name:score      product:price   label:points    code:rank
team:wins       author:year     song:length     model:params
river:km        planet:mass     recipe:minutes  city:population
```

`--key-pairs` selects which pairs to generate. A bare key (`label`) selects
its canonical pair, and `key1:key2` selects an explicit pair. `all` selects
every pair. Hand-written person, product and city families are reused for
their pairs; the remaining pairs generate `{key1}_json` and `{key1}_kv`
families (for example `label_json`, `label_kv`). Each family has its own
entity pool and number range.

`--holdout-keys` names key values, not family names. Every family whose key
pair contains a held-out key becomes eval-only and is never emitted into
`train.jsonl`. The city pair is always eval-only.

## Randomized-key copy family

`research/build_random_copy_queue.py` builds a second copy curriculum in which
the keys themselves are random. The fixed `KEY_PAIRS` curriculum repeats twelve
key pairs; a small model can memorize those key names and, when asked about a
held-out entity family, copy the right values while still emitting the training
keys. A randomized-key corpus removes that shortcut: the only way to answer is
to read the keys from the prompt.

Every row contains 2-4 pairs with the default settings. Keys are drawn from
three generators (`k_<8 hex>`, `field_<6 base36>`, short consonant-vowel
strings), checked against the fixed key vocabulary and an unused-key registry,
and unique within a row. Values come from five synthetic pools: short tokens,
names, cities, integers in `1..999999`, and a multi-word string. One multi-word
value appears in every row. The generator uses only the standard library and
does not load a model.

The prompt lists the exact `key=value` pairs and asks for one of three answer
formats (JSON, KV, YAML). The canonical answer is parsed and compared key by key
before it is kept; a failing row is regenerated. `train.jsonl` and `eval.jsonl`
are built against one shared key/value registry, so the default build reports
`key_overlap: 0` and `value_overlap: 0`.

```powershell
python research/build_random_copy_queue.py `
  --out-dir data/random_copy_v1 `
  --rows 2000 --eval-rows 200 `
  --formats json,kv --keys-per-row 2 --seed 23 --shuffle
```

`--rows` is the number of training rows and `--eval-rows` the number of held-out
rows. `--holdout PATH` moves the eval file (default `<out-dir>/eval.jsonl`).
`--keys-per-row` is the minimum pair count; rows sample from that up to four
pairs. The CLI writes `manifest.json` with the same summary it prints, including
row counts, per-format counts, overlap counts and key/value style
distributions.

This family differs from the fixed key-pair curriculum in three ways: keys are
fresh per row instead of one of twelve memorizable pairs, the train/eval split
is enforced on keys as well as entities, and the generator emits verified
oracle answers without a teacher model. It is still a transcription task; the
validator checks format and copying, not world knowledge.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m pytest tests/test_random_copy_queue.py -q
```

## Staged randomized-key copy family

`research/build_staged_copy_queue.py` splits randomized-key copying into five
stages, so a model can learn one-pair copying before it has to hold two or four
pairs and JSON/KV syntax at the same time:

| # | stage dir | pairs | format | values |
|---|---|---|---|---|
| 1 | `stage_echo_kv` | 1 | KV | short token or single word |
| 2 | `stage_echo_json` | 1 | JSON | short token or single word |
| 3 | `stage_two_kv` | 2 | KV | short token or single word |
| 4 | `stage_two_json` | 2 | JSON | short token or single word |
| 5 | `stage_multi` | 2-4 | JSON/KV | mixed single word, multiword, number |

Every stage has its own `train.jsonl`, `eval.jsonl` and `manifest.json`. Keys
are fresh random strings, checked against `RESERVED_KEYS`, and unique across
the entire build, so train and eval share no keys or values. Each answer is
parsed back and compared key by key before a row is kept; invalid rows are
regenerated. The per-stage manifest reports row counts, format counts, key and
value overlap, value-pool counts and stage order.

```powershell
python research/build_staged_copy_queue.py `
  --out-dir data/staged_copy_v1 `
  --stages all --rows 2000 --eval-rows 200 `
  --seed 23 --key-style short --shuffle
```

`--key-style` chooses `short` (3-6 lowercase alphanumeric), `cv` (3-4
consonant-vowel syllables) or `word` (4-8 lowercase letters). `--stages` takes
a comma list of `echo_kv,echo_json,two_kv,two_json,multi`, or `all`.
`data/staged_copy_v1/` is gitignored. See `docs/STAGED_COPY_CURRICULUM.md` for
the full curriculum design and verification details.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m pytest tests/test_staged_copy_queue.py -q
```

## Multi-key short-value ladder

`research/build_multi_short_queue.py` is a second staged copy family aimed at
the `multi` bottleneck. The earlier `multi` stage varies pair count, value
length and syntax at the same time, so its exact-match score cannot show which
factor is limiting. This generator holds values short and walks pair count up:

| # | stage dir | pairs | values | formats |
|---|---|---|---|---|
| 1 | `stage_2short` | 2 | single token only | JSON/KV |
| 2 | `stage_2mixed` | 2 | one single token + one multiword | JSON/KV |
| 3 | `stage_3short` | 3 | single token only | JSON/KV |
| 4 | `stage_4short` | 4 | single token only | JSON/KV |

`2short` vs `2mixed` isolates the multiword-value penalty; `2short`, `3short`
and `4short` isolate the cost of an additional key. Keys are 3-5 characters:
`short` is lowercase alphanumeric, `cv` is a five-letter consonant-vowel
alternation, and `word` is 3-5 lowercase letters. The default key style is
`short`.

The prompt lists the exact pairs with an explicit instruction:

```text
Copy these pairs exactly into one-line KV: [hhxbz=glowing meadow; fiz=brindtras].
Return only one line of semicolon-separated key=value pairs and keep exactly
these keys and values. ...
```

A single key registry and a single value registry span the whole build, so
train and eval share no key names or values per stage or across stages. Every
answer is rendered canonically, parsed back, compared key by key, and hashed
before the row is kept. Invalid answers are regenerated.

```powershell
python research/build_multi_short_queue.py `
  --out-dir data/multi_short_v1 `
  --rows 2000 --eval-rows 200 `
  --difficulty all --key-style short --seed 23 --shuffle
```

`--difficulty` selects `2short`, `2mixed`, `3short`, `4short`, a comma list, or
`all`. Each stage writes `stage_<name>/train.jsonl`, `eval.jsonl` and
`manifest.json`; the root `manifest.json` records the ladder order, row counts,
format counts and zero-overlap evidence. `data/multi_short_v1/` is gitignored.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m pytest tests/test_multi_short_queue.py -q
```

This family does not reuse or modify the staged-copy generator, and it does not
change the trainer. It is a data-isolation tool: stage-level model results are
required before drawing conclusions about which factor limits multi-key
copying.

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

`canonical_answer(family, answer, entity, number)` validates first and then
rewrites JSON with `json.dumps(..., separators=(",", ":"))` and KV with no
spaces around `=` or `;`. Invalid answers and YAML/plain answers are returned
unchanged. The CLI applies this to every verified row with `--canonical-json`.

## Reusable API

```python
from research.build_diverse_teacher_queue import (
    CROSS_FAMILIES, FAMILIES, TRAIN_FAMILIES,
    build_rows, oracle_answer, resolve_family_plan, split_rows,
    split_summary, validate_answer,
)

def stub_teacher(prompt: str) -> str:
    return '{"sum": 42}'

rows = build_rows(["arith_json"], stub_teacher, count=2, seed=17)
files = split_rows(rows)
```

`build_rows` accepts one callable that maps a prompt to raw text. Pass `None`
to use the deterministic oracle formatter. Counts can be uniform (`count=...`)
or per family (`counts={...}`). The function never imports `torch` or
`transformers`; those imports live inside the CLI model loader.

For an entity split, resolve the plan first and pass the same family lists to
both calls:

```python
plan = resolve_family_plan("all", "population")
rows = build_rows(
    plan["train_families"] + plan["cross_families"], None,
    count=5, seed=17, split_by="entity",
    train_families=plan["train_families"],
    cross_families=plan["cross_families"])
files = split_rows(
    rows, train_families=plan["train_families"],
    cross_families=plan["cross_families"], split_by="entity")
report = split_summary(files, mode="entity")
assert report["entity_overlap"] == 0
```

## Command line

Legacy (no curriculum flags) behaviour is unchanged apart from the two added
YAML train families:

```powershell
python research/build_diverse_teacher_queue.py `
  --out-dir data/diverse_teacher `
  --per-family 40 --city-per-family 25 `
  --city-yaml-count 25 --arith-count 20 `
  --seed 17
```

Copy-generalization curriculum:

```powershell
python research/build_diverse_teacher_queue.py `
  --out-dir data/copy_curriculum `
  --key-pairs all --holdout-keys points,population `
  --split-by entity --canonical-json --shuffle `
  --per-family 20 --seed 17 `
  --replay-out data/copy_curriculum/replay.jsonl
```

- `--key-pairs` is a comma list of `key1:key2` (or bare `key1`) pairs, or
  `all`. The default keeps the original families.
- `--holdout-keys` makes matching families eval-only.
- `--canonical-json` compacts verified JSON and KV answers.
- `--split-by row|entity` selects the per-family row slice (legacy) or the
  entity-first split.
- `--shuffle` deterministically shuffles generated rows with `--seed`.
- `--replay-out PATH` writes legacy-family train rows to PATH, marks them
  `"replay": true`, and removes them from `train.jsonl` so replay never
  duplicates a primary train row.
- `--per-family` controls generated families. Its default is 20 in curriculum
  mode and 40 in legacy mode. `--city-per-family` and `--arith-count` default
  to `--per-family` in curriculum mode, and to 25/20 in legacy mode.
- `--family-count FAMILY=COUNT` overrides one family and can be repeated.
- `--dry-run` uses the oracle formatter and does not load a model.

The primary output files are `train.jsonl`, `indist.jsonl`, `cross.jsonl` and
`all.jsonl`. In row mode train contains 80% of each verified train family,
indist contains the remaining 20%, cross contains every verified eval-only
row, and all contains every verified row. In entity mode the 80/20 split is
applied to entity pools before rows are emitted; every entity is only on one
side, and the same entity gets the same side for JSON, KV and YAML families
that share a pool.

The printed build summary includes machine-readable split fields:

```json
"split": {
  "mode": "entity",
  "entities_train": 44,
  "entities_eval": 16,
  "entities_primary_train": 44,
  "entities_replay": 0,
  "overlap": 0,
  "entity_overlap": 0,
  "entity_overlap_examples": []
}
```

`entities_train` includes replay rows because replay is a training stream.
`entity_overlap` must be 0 for an entity split; the CLI rejects a build whose
entity split overlaps.

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
- Generated cities, populations, years, masses and lengths are synthetic, not
  factual data. They are in a realistic range but are not facts.
- The real Qwen teacher is only exercised by the CLI. Dry-run rows show that
  plumbing works; they are not teacher-quality evidence.
- `--replay-out` moves legacy training rows out of `train.jsonl`; it does not
  decide the replay weight used by adapter training.
