# Verified teacher distillation for copy

`research/build_distill_copy.py` builds **training data** in which a larger
local teacher copies the exact key/value pairs from a staged copy prompt. The
49M student is not changed by this builder. No weights are trained, merged or
modified here.

The purpose is narrow: the 49M model copies three or four arbitrary keys at
0-6% exact match. A local Qwen2.5-0.5B-Instruct teacher can copy them much
more often, but its raw output is not automatically usable. The builder keeps
a teacher answer only after a deterministic parser recovers the exact source
pairs and the answer is rewritten into the canonical requested format.

## Host-GPU command (2000 requested rows)

Run from the repository root on the host with the CUDA GPU and the cached
Qwen model:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python research/build_distill_copy.py `
  --source data/multi_short_v1/stage_2short/train.jsonl `
           data/multi_short_v1/stage_3short/train.jsonl `
           data/multi_short_v1/stage_4short/train.jsonl `
  --out-dir data/distill_copy_v1 `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --device cuda `
  --rows 2000 `
  --seed 23 `
  --formats json,kv,yaml `
  --max-new-tokens 96 `
  --local-files-only `
  --shuffle
```

`--rows` is the number of source prompts attempted, not a promise of 2000
written rows. Teacher acceptance is measured and reported. Add
`--fallback-oracle` if the output must also retain prompts the teacher failed;
those rows are labelled as oracle fallbacks, not teacher-verified rows. If the
model is not in the local Hugging Face cache, pass `--no-local-files-only` or
`--model` with a local path.

Outputs are written under `data/`, which is gitignored:

```text
data/distill_copy_v1/
  train_distill.jsonl
  manifest.json
```

## Verification rule

For each source row the builder takes the expected pairs from the row's
`pairs` field (or `keys`/`values`) and then:

1. sends the row's existing `user` prompt to the teacher with greedy decoding;
2. parses the answer tolerantly. Accepted shapes include flat JSON,
   `{"pairs": [["key", "value"], ...]}`,
   `{"pairs": [{"source": ..., "target": ...}, ...]}`, numbered pair objects,
   code fences, one-line or multiline KV (`key=value` or colon-separated), and
   YAML lines;
3. requires the parsed keys and values to equal the source pairs exactly.
   Order is ignored. Renamed keys, missing keys, extra keys, changed values,
   case changes and truncated answers are rejected. A truncated outer JSON
   object is not salvaged from a nested fragment;
4. canonicalizes an accepted answer to the row's requested format: compact
   flat JSON with no spaces, one-line `key=value;...` KV with no spaces, or
   one `key: value` YAML line per pair.

A teacher answer that fails parsing or exact matching is counted as rejected.
With `--fallback-oracle`, the source `assistant` is used instead, but only
after it is reparsed and rechecked against the expected pairs. Fallback rows
carry `teacher_verified: false` and `canonicalized: false`; accepted teacher
rows carry both as `true`.

Rows contain `user`, `assistant`, `family`, `stage`, `format`, `keys`,
`values`, `source_family`, `teacher_model`, `teacher_verified`,
`canonicalized`, and a small verification block. The canonical `assistant` is
the only answer written to the corpus.

## Manifest

`manifest.json` is content-free. It reports aggregate counters only:

- `requested`: source rows attempted after filtering and `--rows`;
- `generated`: rows written (`teacher_valid + fallback_oracle`);
- `teacher_valid`: rows whose raw teacher answer passed exact verification;
- `fallback_oracle`: rows filled from the source assistant;
- `rejected`: rows written by neither path;
- `by_format` and `by_stage`: generated-row counts;
- a note restating that the manifest contains no prompt, answer, key or value
  text.

It also records the model, seed, source file paths, selected formats, decode
budget and fallback setting. It does not contain row text or examples.

## No eval leakage

The default sources are staging **train** files only. The loader refuses file
names or path components that identify eval data, and it refuses rows whose
`partition` is not `train`. Eval files are never read by this builder and no
eval row is mixed into `train_distill.jsonl`.

## Teacher data is not a capability claim

A verified teacher row means only that the teacher's answer passed the exact
copy check. It says nothing about the 49M student's ability. Do not cite the
builder's acceptance count as a model benchmark, a student score or evidence
that the product model can copy more keys. Any student claim requires a
separate training run followed by the normal held-out evaluation. This branch
does not train, merge, publish or push anything.

## Tests

The builder accepts an injected teacher callable and an injectable teacher
factory for the CLI, so tests do not download or load Qwen:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_build_distill_copy.py -q
```
