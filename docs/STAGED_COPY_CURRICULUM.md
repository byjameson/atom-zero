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
full/content/key/value/syntax loss, content perplexity, exact-match rate,
key-copy rate, key recall and per-family aggregates. The comparator's raw
output contains `user`/`assistant` rows; the runner reads that temporary file,
strips it and deletes it. No prompt or answer text is written to the ledger.

After the last stage the runner repeats this comparison on the final mixed
eval and feeds the content-free aggregate to the adapter gate; see
[Final eval, gate and promotion](#final-eval-gate-and-promotion).

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
  candidate.json
  gate.json
  ledger.json
research/results/staged_curriculum_run.json
data/curriculum_active_adapter.json   # only with --promote and a passing gate
```

Both ledger JSON files contain the same content-free ledger. A stage record
looks like this (abridged):

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

### Final eval, gate and promotion

After the last trained stage the runner closes the loop:

1. choose the final held-out eval:
   - `--final-eval PATH` when given (missing path is an error);
   - otherwise `<data-dir>/mixed_eval.jsonl` when it exists;
   - otherwise `<data-dir>/stage_multi/eval.jsonl` when it exists;
   - otherwise the gate is recorded as `skipped` and no promotion happens.

   `mixed_eval.jsonl` is the optional 250-row mixed held-out file (50 rows per
   stage) used by the manual run. When it is absent, the runner falls back to
   the held-out `stage_multi/eval.jsonl`, so a standard build still produces a
   gate result.

2. run `research/compare_adapters.py` on that file with `--per-family`,
   `--exact-match` and `--key-copy`, against the last stage adapter;
3. write a content-free aggregate (loss, ppl, content/key/value/syntax loss,
   content perplexity, exact-match and key-copy rates, per-family numbers);
4. write the candidate metadata and call
   `python -m atomzero.adapter_candidates evaluate` with the requested
   thresholds;
5. when `--promote` is set and the gate passes, call
   `python -m atomzero.adapter_candidates promote` with the gate JSON, the
   candidate JSON and the last adapter directory.

The exact command, including a promotion into a **separate curriculum
registry**:

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
  --final-eval data/staged_copy_v1/mixed_eval.jsonl `
  --require-content --min-exact-match 0.25 --min-key-copy-rate 0.25 `
  --min-content-delta 0.05 --max-content-ppl-ratio 1.0 `
  --candidate-json checkpoints/staged_curriculum/candidate.json `
  --gate-json checkpoints/staged_curriculum/gate.json `
  --result research/results/staged_curriculum_run.json `
  --promote --active-file data/curriculum_active_adapter.json
```

`--promote` is deliberately strict: it requires `--active-file`,
`--require-content`, `--min-exact-match` and `--min-key-copy-rate`, because
the promotion command rejects a gate that lacks
`content_loss_improved`, `content_ppl_not_worse`, `exact_match_ok` or
`key_copy_ok`. The gate also checks the full loss/perplexity improvement.
A failing gate leaves the registry untouched and records the failure in the
ledger; a promotion subprocess failure is also recorded and exits with code 2.

#### Safe curriculum registry

Use `data/curriculum_active_adapter.json` for curriculum experiments. It is a
separate file from the product registry `data/active_adapter.json`:

- `data/active_adapter.json` is read by `atomzero --adapter auto` and changes
  what the product serves;
- `data/curriculum_active_adapter.json` records only the last passing
  curriculum adapter and does not affect the product;
- a curriculum run with no `--promote` (or a failing gate) never touches
  either registry.

To inspect the curriculum result without activating it:

```powershell
Get-Content data/curriculum_active_adapter.json
```

To use it explicitly, point the server at that file:

```powershell
atomzero --adapter auto `
  --active-adapter-file data/curriculum_active_adapter.json
```

Do not point a curriculum promotion at `data/active_adapter.json` unless the
adapter has also passed the product review process described in
`docs/ADAPTER_PROMOTION.md`.

#### Candidate metadata and family separation

The gate needs `candidate_id` and `eval_examples`; promotion additionally
needs the family separation evidence. Neither file stores prompt or answer
text. The final
mixed eval intentionally uses the same task-family names as training, so the
runner does not claim family-name disjointness. Instead it records
`family_separation: true` only when:

- every final-eval row has an explicit family label;
- no final-eval row appears in any stage train split;
- no final-eval key or value appears in any stage train split.

The metadata also carries the counts (`row_overlap`, `key_overlap`,
`value_overlap`, `train_rows_checked`, `eval_rows_checked`) and
`family_separation_basis`, so the claim is auditable. On the default
`data/staged_copy_v1` build the mixed eval has 250 held-out rows and the
overlap counts are all zero.

#### Ledger fields

The ledger and `--gate-json`/`--candidate-json` outputs are content-free. The
top-level fields include:

```json
{
  "final_eval": {
    "data": "data/staged_copy_v1/mixed_eval.jsonl",
    "examples": 250,
    "metrics": {"adapter": {"loss": 0.9, "content_loss": 1.0,
                            "exact_match_rate": 0.4,
                            "key_copy_rate": 0.548}}
  },
  "candidate_id": "ad-...",
  "candidate_json": "checkpoints/staged_curriculum/candidate.json",
  "gate_json": "checkpoints/staged_curriculum/gate.json",
  "gate": {
    "status": "pass",
    "ok": true,
    "reasons": ["loss_improved", "ppl_not_worse",
                "content_loss_improved", "content_ppl_not_worse",
                "exact_match_ok", "key_copy_ok"],
    "adapter_exact_match": 0.4,
    "adapter_key_copy_rate": 0.548
  },
  "exact_match_rate": 0.4,
  "key_copy_rate": 0.548,
  "promotion": {"requested": true, "promoted": true,
                "reason": "promoted",
                "active_file": "data/curriculum_active_adapter.json"},
  "active_file": "data/curriculum_active_adapter.json"
}
```

The stage records are unchanged. `promotion.promoted` is false with
`reason: "gate_failed"` when the final gate fails, `"dry_run"` in a dry run,
`"gate_skipped"` when no final eval was available and `"not_requested"` when
`--promote` was not set.

### Autonomous self-improve integration

`atomzero-self-improve` can invoke this runner as one bounded step. The
curriculum command runs only when `--opt-in` and
`--adapter-curriculum-command` are both set; without the flag the loop behaves
exactly as before.

| flag | meaning | default |
|---|---|---|
| `--adapter-curriculum-command` | shell command template | unset |
| `--adapter-curriculum-result` | runner JSON ledger read back by the loop | `research/results/staged_curriculum_run.json` |
| `--adapter-curriculum-out` | adapter output directory (`{out_dir}`) | `checkpoints/staged_curriculum` |
| `--adapter-curriculum-active-file` | curriculum registry (`{active_file}`) | `data/curriculum_active_adapter.json` |
| `--adapter-curriculum-final-eval` | final mixed eval (`{final_eval}`) | `data/staged_copy_v1/mixed_eval.jsonl` |
| `--adapter-curriculum-data-dir` | staged queue root (`{data_dir}`) | `data/staged_copy_v1` |
| `--adapter-curriculum-min-exact-match` | loop-side exact-match floor | `0.25` |
| `--adapter-curriculum-min-key-copy-rate` | loop-side key-copy floor | `0.25` |

The template supports `{base}`, `{data_dir}`, `{out_dir}`, `{result}`,
`{active_file}` and `{final_eval}`. `{base}` comes from `--adapter-base`; use
`--adapter-curriculum-data-dir` to point `{data_dir}` at another staged queue.

Exact command:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
atomzero-self-improve `
  --signal data/staged_signals.json --opt-in `
  --adapter-promotable 1 `
  --adapter-base checkpoints/transformer_behavior_50m_long `
  --adapter-curriculum-command "python research/run_staged_curriculum.py --data-dir {data_dir} --out-dir {out_dir} --base {base} --stages echo_kv,echo_json,two_kv,two_json,multi --steps 2000 --batch-size 4 --lr 3e-4 --content-weight 2 --seed 23 --final-eval {final_eval} --require-content --min-exact-match 0.25 --min-key-copy-rate 0.25 --candidate-json {out_dir}/candidate.json --gate-json {out_dir}/gate.json --result {result} --promote --active-file {active_file}" `
  --adapter-curriculum-result research/results/staged_curriculum_run.json `
  --adapter-curriculum-out checkpoints/staged_curriculum `
  --adapter-curriculum-active-file data/curriculum_active_adapter.json `
  --adapter-curriculum-final-eval data/staged_copy_v1/mixed_eval.jsonl `
  --json
```

When the planner selects `adapter_candidate_ready`, the curriculum command
replaces the built-in adapter training step. On any other action it runs as a
bounded pre-step before the code change. The loop records a content-free
`curriculum` block:

```json
"curriculum": {
  "status": "pass",
  "returncode": 0,
  "elapsed_seconds": 41.27,
  "gate_status": "pass",
  "exact_match": 0.4,
  "key_copy": 0.548,
  "promotion": {"requested": true, "promoted": true,
                "reason": "promoted"},
  "result_path": "research/results/staged_curriculum_run.json",
  "reasons": []
}
```

`status` is `pass` only when the command exits 0, the ledger gate is `pass`
and the loop-side exact-match/key-copy floors are met. A nonzero exit, missing
or invalid ledger is `error`; a failing or skipped gate is `fail`. In both
cases the loop status becomes `curriculum_failed` and it does not claim
success. The loop never copies command output, rows or prompt text into this
block.

### Dry run and tests


`--dry-run` skips all subprocesses, writes fake trainer, comparator and gate
outputs, needs no data files or checkpoint, and never writes a registry. The
fake final eval mirrors the manual 250-row mixed run (exact match 0.40,
key-copy 0.548, content loss 4.04 -> 1.00). It is useful to check option
validation and produce a sample result:

```powershell
python research/run_staged_curriculum.py `
  --dry-run --stages echo_kv,echo_json,multi `
  --out-dir checkpoints/staged_curriculum_dry_run `
  --result research/results/staged_curriculum_dry_run.json `
  --candidate-json research/results/staged_curriculum_dry_candidate.json `
  --gate-json research/results/staged_curriculum_dry_gate.json `
  --require-content --min-exact-match 0.25 --min-key-copy-rate 0.25 `
  --promote --active-file data/curriculum_active_adapter_dry_run.json
```

The dry run records `promotion.reason: "dry_run"` and does not create the
active file. Without explicit `--candidate-json`/`--gate-json`, a dry run does
not create the output directory either.

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
