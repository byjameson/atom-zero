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

## Curriculum runner

`research/run_staged_curriculum.py` chains the five stages instead of relying
on manual trainer commands. It uses only the standard library plus
subprocesses; the runner never imports `torch` or loads a model. Training and
evaluation run through the existing
`research/train_adapter_candidate.py` and `research/compare_adapters.py`.

### Exact command

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python research/run_staged_curriculum.py `
  --data-dir data/staged_copy_v1 `
  --out-dir checkpoints/staged_curriculum `
  --base checkpoints/transformer_behavior_50m_long `
  --stages echo_kv,echo_json,two_kv,two_json,multi `
  --steps 2000 --batch-size 4 --lr 3e-4 --content-weight 2 `
  --replay-weight 0 --rank 8 --alpha 8 --rank-attn 4 --alpha-attn 8 `
  --seed 23 `
  --result research/results/staged_curriculum_run.json
```

Add `--replay-jsonl data/...` together with a non-zero
`--replay-weight` to mix older rows into every stage. The runner passes the
same replay settings to all stages.

### Stage order

The canonical order is fixed and enforced even if `--stages` lists the names
in another order:

1. `echo_kv` - one KV pair;
2. `echo_json` - one JSON pair;
3. `two_kv` - two KV pairs;
4. `two_json` - two JSON pairs;
5. `multi` - two to four mixed KV/JSON pairs.

Every stage trains into `out_dir/<stage>/`. The first stage starts a fresh
adapter. Every later stage adds `--init-adapter` pointing at the previous
stage's output directory, so the LoRA weights continue across the curriculum.
Explicit rank/alpha flags must match the saved adapter geometry; the defaults
above do.

### Evaluation

After training a stage, the runner evaluates the adapter twice:

- on that stage's own `stage_<stage>/eval.jsonl`;
- on the final `stage_multi/eval.jsonl`, unless the current stage is `multi`
  or the file does not exist.

Each comparison runs `compare_adapters.py` with `--per-family`,
`--exact-match` and `--key-copy`. The runner extracts only aggregate numbers:
full/content/key/value/syntax loss, exact-match rate, key-copy rate, key
recall and per-family aggregates. The comparator's raw output contains
`user`/`assistant` rows; the runner reads that temporary file, strips it and
deletes it. No prompt or answer text is written to the ledger.

### Expected output

```text
checkpoints/staged_curriculum/
  echo_kv/
    adapter.pt
    config.json
    trainer_result.json
  echo_json/
  two_kv/
  two_json/
  multi/
  ledger.json
research/results/staged_curriculum_run.json
```

Both JSON files contain the same content-free ledger. A stage record looks
like this (abridged):

```json
{
  "stage": "echo_kv",
  "index": 1,
  "adapter": "checkpoints/staged_curriculum/echo_kv",
  "init_adapter": null,
  "trainer": {
    "returncode": 0,
    "elapsed_seconds": 41.27,
    "adapter_config": {"r_mlp": 8, "alpha_mlp": 8,
                       "r_attn": 4, "alpha_attn": 8},
    "result": {"train_loss": 0.041, "train_steps": 2000,
               "optimizer_steps": 500, "lora_parameters": 405504}
  },
  "evals": {
    "stage": {
      "returncode": 0,
      "elapsed_seconds": 3.4,
      "metrics": {
        "adapter": {
          "loss": 0.182, "content_loss": 0.201, "key_loss": 0.154,
          "value_loss": 0.166, "syntax_loss": 0.097,
          "exact_match_rate": 0.94, "key_copy_rate": 0.98,
          "key_recall": 0.99
        },
        "base": {"loss": 3.71},
        "per_family": {"staged_copy_echo_kv": {"adapter": {"loss": 0.182}}}
      }
    },
    "multi": {"returncode": 0, "metrics": {"adapter": {"loss": 1.83}}}
  },
  "key_copy_rate": 0.98,
  "key_recall": 0.99,
  "exact_match_rate": 0.94,
  "key_copy_gate": null
}
```

The exact numbers depend on the base checkpoint and step budget; the shape and
the field names are stable. `stages` is ordered, `stage_order` lists the stages
that actually ran, and `stopped_early`/`stop_stage` report a gate stop.

### Key-copy gate

Add a threshold in `[0, 1]` to stop the curriculum as soon as a stage shows
the copy behavior:

```powershell
python research/run_staged_curriculum.py `
  --base checkpoints/transformer_behavior_50m_long `
  --stop-on-key-copy 0.8
```

The gate uses the adapter's key-copy rate on the current stage's own held-out
eval split, not on the final `multi` split. A stage is recorded with
`"key_copy_gate": {"threshold": 0.8, "passed": true}`, the runner sets
`stopped_early` and `stop_stage`, and no later stage is trained. Reaching the
gate is a schedule stop, not an adapter promotion: it does not change the
trainer, the comparator or the product weights.

### Dry run and tests

`--dry-run` skips all subprocesses, writes fake aggregate metrics and needs no
data files or checkpoint. It is useful to check option validation and produce
a sample result:

```powershell
python research/run_staged_curriculum.py `
  --dry-run --stages echo_kv,echo_json,multi `
  --out-dir checkpoints/staged_curriculum_dry_run `
  --result research/results/staged_curriculum_dry_run.json
```

```powershell
python -m pytest tests/test_curriculum_runner.py -q
```

### How to use the stages

Suggested training sequence:

1. Train or adapt on `stage_echo_kv` until exact match is stable.
2. Evaluate on `stage_echo_kv/eval.jsonl`, then move to `stage_echo_json`.
3. Repeat for `two_kv`, `two_json` and `multi`.
4. Keep each stage's `eval.jsonl` untouched. A stage counts as learned only
   when exact match is high on its own held-out keys.

The runner automates exactly this sequence and records the evidence. The files
use the teacher-queue `user`/`assistant`/`family` schema, so existing research
trainers can consume them without changes to the trainer. This branch does not
change the trainer, the comparator or the random-copy generator.

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
