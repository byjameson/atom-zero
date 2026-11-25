# New-task teacher distillation

**Status:** research data pipeline. `research/build_distill_new_tasks.py`
generates and validates new copy tasks; it never trains, changes weights,
promotes an adapter or makes a product claim.

The earlier hard-label distillation experiment reused the prompts and exact
key/value pairs of the staged copy corpus. Qwen could copy those rows more
often than the 49M student, but the accepted rows were mostly new surface forms
of known content. This pipeline tests a different idea: use Qwen as a
**new-task generator**. The teacher invents new keys and values, and ordinary
Python verifies that each generated task is well formed before the builder
chooses the answer format.

```text
teacher prompt (fresh keys/values requested)
  -> Qwen2.5-0.5B-Instruct (host GPU, chat template, greedy)
  -> tolerant parse (JSON, pretty JSON, fences, KV/YAML)
  -> deterministic constraint validation
  -> canonical JSON / one-line KV / YAML lines
  -> train_new_tasks.jsonl + content-free manifest.json
```

No eval file is read or sent to the teacher. `--avoid-source` accepts train
JSONL files only; a path that looks like an eval split is refused before it is
opened.

## Distinction from hard-label distillation

| | hard-label distillation | new-task generation |
|---|---|---|
| task content | existing staged prompts | teacher-invented prompts |
| keys and values | exact source pairs | new teacher pairs |
| verification | parsed pairs equal the source pairs exactly | deterministic shape/value constraints |
| what it tests | surface variation of known rows | learning from unseen rows |
| output | `distill_copy_v1/train_distill.jsonl` | `distill_new_v1/train_new_tasks.jsonl` |

Hard-label data can only be as novel as the source corpus. New-task data can be
novel, but it is also easier for a small teacher to produce malformed rows, so
the acceptance rate is part of the result.

## Data contract

The builder writes:

- `data/distill_new_v1/train_new_tasks.jsonl` — accepted training rows
- `data/distill_new_v1/manifest.json` — aggregate counts only

Each row contains:

- `user` — an instruction that lists every `key=value` pair and the requested
  answer format;
- `assistant` — compact flat JSON, one-line semicolon KV, or YAML
  `key: value` lines;
- `keys`, `values`, `pairs`, `key_count`;
- `family` — `newtask_<n>key_<fmt>`, for example `newtask_3key_json`;
- `format` — `json`, `kv` or `yaml`;
- `partition` — always `train`;
- `teacher_model`, `teacher_verified: true`, `canonicalized: true`;
- `verification` — method name and the checks that passed.

`teacher_verified` means the row passed the deterministic checks below. It does
not mean that the teacher is correct in general, that the task is interesting,
or that training on it will improve held-out copy behavior.

## Validation rules

For each parsed teacher object the builder requires:

1. the answer parses to a flat object (a JSON object, a single-object array, or
   a KV/YAML line or block);
2. the object has **exactly** the requested number of pairs (2, 3 or 4);
3. keys are unique inside the object;
4. every key matches `[a-z][a-z0-9]{2,5}` (3-6 lowercase alphanumeric
   characters, starting with a letter so KV/YAML span weighting works);
5. no key is in the fixed blocklist of structured-output and generic words
   (`key`, `value`, `json`, `pair`, `answer`, `key1`, `value1`, ...);
6. every value is a string or finite number, non-empty, at most 40 characters,
   contains a letter or digit, and contains none of `; = [ ] { } ,` or control
   whitespace; numbers are normalized to their decimal text so the prompt,
   metadata and answer agree;
7. the row's order-insensitive pair signature has not already been accepted;
8. if `--avoid-source` was supplied, the rendered prompt, rendered answer and
   pair signature do not occur in that train corpus.

A JSON object with duplicate keys, a truncated object, an object with the wrong
pair count, or prose with no object is rejected. A rejected line is counted; it
is not salvaged from a prefix.

Duplicates are exact rows: the same key/value pairs, ignoring order. Two rows
may reuse a key set with different values. Qwen often repeats its two prompt
shape examples or reuses a key set across a batch; those rows are still valid
copy tasks, but they are not proof of broad key diversity. The manifest reports
duplicates and per-key-count counts so this is visible.

`--yaml-count N` reserves `N` output slots for YAML. The rest rotate through the
non-YAML formats. YAML answers are ordinary `key: value` lines, one pair per
line, matching the repository's YAML answer-span parser.

## Exact host-GPU command for 1000 requested tasks

Run from the repository root on the Windows host with the GPU. The model is
already cached in the smoke environment; `--local-files-only` prevents a
download.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_distill_new_tasks.py `
  --out-dir data/distill_new_v1 `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --device cuda `
  --rows 1000 `
  --batch-size 5 `
  --keys-per-row 2,3,4 `
  --formats json,kv `
  --yaml-count 0 `
  --max-new-tokens 512 `
  --seed 23 `
  --local-files-only `
  --shuffle
```

`--rows` is the number of task slots requested. Validation can reject some
slots, so the accepted JSONL is smaller than 1000. The manifest reports
`requested`, `accepted` and `rejected`.

`--batch-size` defaults to 20, but the 20-task smoke used 5: Qwen2.5-0.5B
underproduced or changed the pair count when asked for 20 tasks in one prompt.
Five tasks per prompt gave usable acceptance. Increase the batch size only after
checking a small run.

To reject overlap with existing staged train text, add the train files. Never
pass an `eval.jsonl` path; the builder refuses it.

```powershell
  --avoid-source data/multi_short_v1/stage_2short/train.jsonl `
                 data/multi_short_v1/stage_3short/train.jsonl `
                 data/multi_short_v1/stage_4short/train.jsonl
```

The command above does not read eval files. It writes only the two files under
`--out-dir`; the directory is under the gitignored `data/` tree.

## Measured 20-task smoke

Host GPU, cached Qwen2.5-0.5B-Instruct, greedy decoding. The smoke used
`--rows 20`, `--batch-size 5`, `--keys-per-row 2,3,4`, `--formats json,kv`,
`--max-new-tokens 512`, `--seed 23`, `--device cuda` and
`--local-files-only`, without `--shuffle`; a repeat run produced the same
manifest counts.

```text
requested:        20
accepted:         16
rejected:          4
duplicate_count:   0
invalid_task_count: 2
teacher_calls:     6
by_format:        json 8, kv 8, yaml 0
by_keys:          2-key 5, 3-key 6, 4-key 5
```

The two invalid candidates had the wrong pair count or an invalid key. This is
a data-pipeline smoke, not a capability measurement. It does not show that the
49M student learns or improves; that requires a separate training and held-out
evaluation cycle.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_distill_new_tasks.py -q
```

The tests inject a fake teacher. They never load torch, download a model or
touch an eval file. They cover tolerant parsing, dedupe, key/value validation,
exact canonical formats, deterministic seeds, a content-free manifest, invalid
teacher output and refusal of eval sources.

## Honest limits

- Validation is literal and deterministic; it does not judge whether a task is
  useful, natural or semantically correct.
- Qwen may echo the two shape examples, reuse keys across a batch, or produce
  fewer tasks than requested. Acceptance is below 100% by design.
- Three- and four-key generation is harder than two-key generation. A small
  run can over-represent easy two-key rows; read `by_keys`.
- `--avoid-source` only checks the train files you pass. With no avoid source,
  novelty is by construction (new teacher prompts) but is not checked against
  an external corpus.
- The output is training data. It is not a model result, a product claim or
  evidence that the 49M model has improved.
