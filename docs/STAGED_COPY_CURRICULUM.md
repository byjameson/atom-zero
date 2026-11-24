# Staged copy curriculum

Date: 2026-11-24

## Why stage the task

The randomized-key corpus (`data/random_copy_v1`) starts at the hardest
setting: two to four arbitrary key/value pairs, random key strings, multiword
values, and JSON or KV syntax. After 1,000 optimizer steps a LoRA adapter
reached only 19.0% key recall and 1.5% exact match on held-out keys
(`docs/RANDOM_KEY_COPY_CYCLE.md`). The model was learning values and syntax
while still failing to copy the key strings.

The staged curriculum isolates the sub-skills. A model first copies one pair
in the simplest format, then wraps that pair in JSON, then copies two pairs,
then two pairs in JSON, and only then sees 2-4 mixed pairs. Each stage has its
own train/eval split and can be trained or measured independently.

## Stages

| # | stage | directory | pairs | format | value types |
|---|---|---|---|---|---|
| 1 | `echo_kv` | `stage_echo_kv/` | 1 | KV | short token or single word |
| 2 | `echo_json` | `stage_echo_json/` | 1 | JSON | short token or single word |
| 3 | `two_kv` | `stage_two_kv/` | 2 | KV | short token or single word |
| 4 | `two_json` | `stage_two_json/` | 2 | JSON | short token or single word |
| 5 | `multi` | `stage_multi/` | 2-4 | JSON/KV | mixed single word, multiword, number |

Stage 1 is the echo case: the prompt contains one bracketed pair such as
`[o5ubr=sim]` and the answer must be exactly `o5ubr=sim`. Stage 2 asks for
`{"o5ubr":"sim"}`. Stages 3-4 repeat the same progression with two pairs.
Stage 5 alternates KV and JSON and samples 2-4 pairs; every row includes a
multiword value, and rows with three or more pairs also include a number and
a single word.

The generator is `research/build_staged_copy_queue.py`. It uses only the
standard library and never imports `torch`, a tokenizer or a model.

## Keys and values

- Keys are random strings drawn per row in one of three styles:
  - `short`: 3-6 lowercase alphanumeric characters, first character a letter
    (`q7f3`, `zorp`);
  - `cv`: 3-4 consonant-vowel syllables (`bavotir`, 6-8 letters);
  - `word`: 4-8 lowercase letters (`brantok`).
- Keys are checked against the fixed key vocabulary (`RESERVED_KEYS`), which
  includes `name`, `score`, `city`, `population`, `river`, `km` and the other
  names from the earlier curricula.
- A single key registry is shared by every stage, train split and eval split.
  Keys are unique within a row and across the whole build, so train/eval key
  overlap is exactly zero, not just for the aggregated corpus but per stage.
- Values use four pools: short synthetic tokens, single synthetic words,
  adjective+noun multiwords, and integers in `1..999999`. A single value
  registry also spans the whole build, so value overlap is zero.
- Stages 1-4 use only tokens and single words. Stage 5 mixes all four pools.

The values are synthetic. The task is transcription, not world knowledge; a
row does not assert a fact about the world.

## Row schema and verification

Every JSONL row contains:

```json
{
  "user": "... [o5ubr=sim] ...",
  "assistant": "o5ubr=sim",
  "family": "staged_copy_echo_kv",
  "stage": "echo_kv",
  "stage_index": 1,
  "format": "kv",
  "partition": "train",
  "key_style": "short",
  "keys": ["o5ubr"],
  "values": ["sim"],
  "pairs": [["o5ubr", "sim"]],
  "key_styles": ["short"],
  "value_pools": ["token"],
  "verification": {
    "verified": true,
    "promotable": true,
    "status": "verified",
    "method": "deterministic_staged_copy_roundtrip_v1",
    "checks": ["prompt_contains_exact_pairs", "answer_parses",
               "keys_match", "values_match", "keys_unique",
               "verification_hashes_match"],
    "prompt_sha256": "...",
    "answer_sha256": "..."
  }
}
```

Generation builds the canonical answer with `render_answer`, then
`validate_row` checks all of the following before the row is kept:

1. the stage, format and pair count match the stage spec;
2. keys are unique and all match the chosen key style and reserved-key
   vocabulary;
3. the bracketed prompt pairs equal the metadata pairs exactly;
4. the answer parses as JSON or KV, has exactly the requested key set and
   exact values (numbers are compared numerically, strings exactly);
5. for `echo_kv`, the answer string is the prompt pair verbatim;
6. the prompt and answer hashes in the verification block match.

A failing row is regenerated. The manifest records the number of rejected
attempts, and the final build aborts if any train/eval key or value overlap is
detected.

## Commands

Generate all stages with the defaults:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python research/build_staged_copy_queue.py `
  --out-dir data/staged_copy_v1 `
  --stages all --rows 2000 --eval-rows 200 `
  --seed 23 --key-style short --shuffle
```

The exact command is idempotent: the same seed, stage list and key style
produce byte-identical `train.jsonl` and `eval.jsonl` files.

Generate or inspect only the early stages:

```powershell
python research/build_staged_copy_queue.py `
  --out-dir data/staged_copy_echo `
  --stages echo_kv,echo_json `
  --rows 500 --eval-rows 100 --seed 23 --key-style short
```

Other key styles:

```powershell
python research/build_staged_copy_queue.py `
  --out-dir data/staged_copy_cv `
  --stages all --seed 23 --key-style cv --shuffle
```

Run the tests:

```powershell
python -m pytest tests/test_staged_copy_queue.py -q
```

## Output layout

```text
data/staged_copy_v1/
  manifest.json                 # root: stage order, totals, overlap
  stage_echo_kv/
    train.jsonl
    eval.jsonl
    manifest.json               # rows, formats, overlap, value pools
  stage_echo_json/
  stage_two_kv/
  stage_two_json/
  stage_multi/
```

The root manifest includes `stage_order`, per-stage rows and overlap, aggregate
format counts and global overlap. Each stage manifest repeats the same fields
for that stage and adds `paths` for its three files.

`data/` is gitignored. Generated queues are local artifacts and are not
committed.

## Default build summary

The default command above produces:

| stage | train | eval | format(s) | key overlap | value overlap |
|---|---:|---:|---|---:|---:|
| `echo_kv` | 2,000 | 200 | KV | 0 | 0 |
| `echo_json` | 2,000 | 200 | JSON | 0 | 0 |
| `two_kv` | 2,000 | 200 | KV | 0 | 0 |
| `two_json` | 2,000 | 200 | JSON | 0 | 0 |
| `multi` | 2,000 | 200 | KV/JSON | 0 | 0 |
| **total** | **10,000** | **1,000** | JSON 5,500 / KV 5,500 | **0** | **0** |

The build has 19,824 unique keys and 19,824 unique values, 11,000 verified
rows and 0 rejected attempts. The `multi` stage has 2-4 pairs (mean 3.011) and
contains all four value pools.

## How to use the stages

Suggested training sequence:

1. Train or adapt on `stage_echo_kv` until exact match is stable.
2. Evaluate on `stage_echo_kv/eval.jsonl`, then move to `stage_echo_json`.
3. Repeat for `two_kv`, `two_json` and `multi`.
4. Keep each stage's `eval.jsonl` untouched. A stage counts as learned only
   when exact match is high on its own held-out keys.

The files use the teacher-queue `user`/`assistant`/`family` schema, so existing
research trainers can consume them without changes to the trainer. This branch
does not add a trainer or change the random-copy generator.

## Honest boundary

- The generator proves that the corpus parses and copies exactly; it says
  nothing about model quality until a training run is done.
- Random short keys stress tokenization as well as copying. If `short` remains
  too hard, `word` or `cv` separates tokenization difficulty from copy-circuit
  difficulty, but those styles are also easier to memorize in shape.
- Stage 5 mixes 2-4 pairs, so a two-pair row cannot contain all three of
  single, multiword and number values. The guarantee is per row for multiword
  and per stage for the mixed distribution; rows with 3+ pairs contain all
  three categories.
- The curriculum schedules data; it does not decide replay weight, learning
  rate, adapter locus or step counts.
