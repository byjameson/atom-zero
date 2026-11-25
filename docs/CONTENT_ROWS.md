# Content replay rows

## Why this exists

Mixed copy + content full fine-tuning needs a content rehearsal corpus. The
repository does not ship large pretraining text, but the desktop workspace has a
local facts file at:

```text
C:\Users\biris\Desktop\opencode\ai gelistir\data\rag\world_facts.jsonl
```

Rows in that file have the shape:

```json
{"id": "...", "title": "...", "lang": "en", "text": "..."}
```

`research/build_content_rows.py` turns those facts into deterministic
prompt/answer rows. Every answer is a trimmed extract from the fact's own
`text`; the builder does not call a model, download data, or invent content.

**Content rows are rehearsal data, not a factuality claim.** They help a
candidate rehearse factual, source-grounded wording while it learns copy
skills. The builder does not verify the facts, the source license, or whether
the model has stored them in its weights. Retrieval, citation and evaluation
stay separate concerns.

## Row shape

Each output line uses the repository schema plus provenance:

```json
{"user": "...", "assistant": "...", "family": "content_qa",
 "source_id": "en:...", "source_title": "...", "style": "qa"}
```

| style | prompt | answer |
|---|---|---|
| `qa` | `What does the source say about <title>?` | a short extract from `text` |
| `summary` | `Summarize the source about <title>.` | `In summary: ` plus the extract |
| `citation` | `Give a source-cited answer about <title>.` | `According to the source "<title>", ` plus the extract |

Answers are capped at `--max-answer-chars`. The builder prefers the last
sentence boundary that fits; if the first sentence is longer than the cap, it
cuts on a word boundary. Extracts are contiguous prefixes of the
whitespace-normalized source text.

## Filters and determinism

The builder reads the facts file as input only and skips rows that are:

- malformed JSON, non-objects, or missing non-empty `id`/`title`/`lang`/`text`;
- shorter than `--min-chars` after whitespace normalization;
- not in the selected language. The default keeps English and accepts
  regional variants such as `en-US`; use `--lang fr`, `--lang de`, etc. to keep
  another primary language;
- duplicate titles (NFKC-normalized and case-folded; the first eligible title
  wins).

Eligible rows are shuffled with `random.Random(seed)` and then capped at
`--max-rows`, so the same input and seed produce byte-identical output. The
manifest has counts, skip reasons, style, seed and the output SHA-256, but no
prompt, answer, title or full path text.

## Exact command

From the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_content_rows.py `
  --facts "C:\Users\biris\Desktop\opencode\ai gelistir\data\rag\world_facts.jsonl" `
  --out data/content_replay_v1/train.jsonl `
  --manifest data/content_replay_v1/manifest.json `
  --max-rows 5000 `
  --min-chars 200 `
  --max-answer-chars 400 `
  --style qa `
  --seed 13
```

The output defaults to `data/content_replay_v1/train.jsonl`, which is
gitignored generated data. The command refuses to overwrite an existing
`--out` unless `--force` is passed. A missing facts file, invalid argument or
I/O error exits non-zero without writing the output.

Build a separate file for each style you want to mix, for example
`train_summary.jsonl` and `train_citation.jsonl`, with the corresponding
`--style` values.

## Mix with copy data

Use the existing deterministic mixer. Weights are sampling multipliers, not
target fractions; see [`MIXED_REPLAY.md`](MIXED_REPLAY.md) for the full rules.

```powershell
python research/mix_training_data.py `
  --copy data/copy_curriculum_v1/train.jsonl `
  --content data/content_replay_v1/train.jsonl `
  --copy-weight 1.0 `
  --content-weight 1.0 `
  --max-rows 120000 `
  --seed 13 `
  --shuffle `
  --out data/mixed_replay_v1/train.jsonl `
  --manifest data/mixed_replay_v1/manifest.json
```

The mixed file is still research training data. It must pass the retention
evaluation and the checkpoint gate before it is used for a candidate base; see
[`RETENTION.md`](RETENTION.md), [`CHECKPOINT_GATE.md`](CHECKPOINT_GATE.md) and
[`FULL_FINETUNE_CONTROL.md`](FULL_FINETUNE_CONTROL.md). This builder never
trains, promotes, merges or pushes anything.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_content_rows.py -q
```

The tests cover deterministic output, all three styles, the answer cap,
sentence-boundary trimming, skip reasons, the content-free manifest, `--force`
and a missing facts file.
