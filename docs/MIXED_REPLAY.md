# Mixed copy + content replay training data

## Why this exists

The 0.80.0 8L/24L upcycled checkpoints were fully fine-tuned on copy-only
curriculum data. Copy exact match improved, but content/QA behaviour regressed
on the retention set. More copy data alone does not fix that: the next full
fine-tune needs copy rows **and** content/replay rows in a controlled ratio.

`research/mix_training_data.py` builds that training file deterministically. It
reads the repository row schema, removes duplicate pairs, refuses eval paths,
and writes a content-free manifest. It does not train, download, merge or
promote anything. The output must still pass the retention evaluation and the
checkpoint gate before it is used for a candidate base.

## Row schema and behaviour

Each JSONL line may contain:

```json
{"user": "...", "assistant": "...", "family": "optional", "keys": ["optional"]}
```

- `user` and `assistant` are required non-empty strings.
- `family` and `keys` are optional; when present they must be valid (a
  non-empty string and a list/tuple of strings).
- Invalid JSON, non-objects and rows with missing/invalid required fields are
  skipped and counted. Blank lines are ignored.
- Rows are deduplicated by the NFKC-normalized, whitespace-collapsed
  `(user, assistant)` pair. The **first** occurrence wins. Case and punctuation
  are preserved because copy examples can legitimately differ by case.
- Copy files are read first, then content files, in the order given. A
  duplicate in a later file is dropped and reported under that file.

With no `--content` file the command is a copy-only dedupe/cleanup pass.

## Exact command: copy + content mix

Replace `data/content_v1/train.jsonl` with the real content/replay corpus. The
corpus is local and gitignored, so the command works only when that file exists.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/mix_training_data.py `
  --copy data/copy_curriculum_v1/train.jsonl `
         data/copy_curriculum_v2/train.jsonl `
  --content data/content_v1/train.jsonl `
  --copy-weight 1.0 `
  --content-weight 1.0 `
  --max-rows 120000 `
  --seed 13 `
  --shuffle `
  --out data/mixed_replay_v1/train.jsonl `
  --manifest data/mixed_replay_v1/manifest.json
```

The command prints a content-free summary and returns non-zero on an invalid
path, an eval path, a bad weight or an I/O error. `--manifest` is optional, but
record it with every run: it contains the exact weights, seed and output hash.

Copy-only variant:

```powershell
python research/mix_training_data.py `
  --copy data/copy_curriculum_v1/train.jsonl `
  --out data/mixed_replay_v1/copy_only.jsonl `
  --seed 13 --shuffle `
  --manifest data/mixed_replay_v1/copy_only_manifest.json
```

## How weights work

Weights are sampling multipliers, not target fractions. For each file:

```text
requested = round(weight * unique_rows_in_that_file)
selected  = min(requested, unique_rows_in_that_file)
```

Rows are never duplicated to satisfy a weight. If a file cannot supply its
requested rows, the mixer fills the shortfall from other files that still have
spare rows, and reports `exhausted`, `filled` and `unfilled_rows`. Weights above
`1.0` therefore do not oversample a source; they only flag that it is too small.
A weight of `0.0` excludes that source without deleting it from the manifest.

`--max-rows` caps the final count. When the plan exceeds the cap, quotas are
reduced proportionally (largest-remainder allocation) so the intended ratio is
preserved instead of truncating the last source. `--shuffle` shuffles the final
rows with `random.Random(seed)`; without it the output keeps source order.

## Ratio guidance

With `copy_weight = c`, `content_weight = k`, `C` unique copy rows and `N`
unique content rows, the planned mix is approximately `c*C : k*N`.

- Start with `1.0 / 1.0`. That uses every unique row and gives the corpus's
  natural content share, `N / (C + N)`.
- To reach a target content share `f`, keep `k = 1.0` and set
  `c = ((1 - f) / f) * (N / C)`. For example, 100k copy rows and 20k content
  rows with 25% content needs `c = (0.75 / 0.25) * (20 / 100) = 0.6`.
- With the same corpus, `c = 1.0` gives only 16.7% content. To raise the content
  share, **lower the copy weight or add more content rows**; increasing
  `content_weight` above `1.0` cannot exceed the available content rows.
- A sensible first experiment is to keep all content rows and cap copy at
  3-9x the content count (10-25% content), then compare retention. Do not
  increase the copy share just because copy loss is lower.
- The previous copy-only 63M checkpoint failed content retention. If retention
  still fails, change the corpus or the weights, not just the step count.

## Leakage rule

The mixer never reads eval rows. Before opening **any** input it validates all
paths, and refuses:

- any file name containing `eval`, case-insensitive (`eval.jsonl`,
  `heldout_EVAL_rows.jsonl`, ...);
- any path with an `eval` or `eval_frozen` directory component;
- any path under `data/eval_frozen/`.

A single bad path aborts the whole run before the first file is read, so a
valid copy file listed earlier cannot be read first. Frozen eval rows belong
only in retention and checkpoint-gate commands, never in `--copy` or
`--content`.

## Manifest

The manifest is content-free. It contains paths and counts, but no prompts,
answers, keys, values or family names:

```json
{
  "schema": "atomzero.mix_training_data.v1",
  "weights": {"copy": 1.0, "content": 1.0},
  "seed": 13,
  "shuffle": true,
  "final_rows": 118402,
  "duplicates": 37,
  "skipped": 4,
  "filled_rows": 0,
  "unfilled_rows": 0,
  "sources": [
    {"role": "copy", "path": "data/copy_curriculum_v1/train.jsonl",
     "rows_read": 100000, "skipped": 3, "duplicates": 35,
     "available": 99962, "requested": 99962, "selected": 99962,
     "filled": 0, "exhausted": false}
  ],
  "output": {"path": "data/mixed_replay_v1/train.jsonl",
             "sha256": "..."},
  "eval": {"rows_used": 0,
           "note": "No eval rows were read or used. Eval-looking input paths are refused before opening."}
}
```

Use `output.sha256` (also mirrored as `output_sha256`) to identify the exact
training file in experiment notes.

## The result must pass retention and the checkpoint gate

Building a mix is not evidence that forgetting is fixed. A candidate trained on
the mix must pass both gates:

1. **Retention** against the product base, on content/frozen eval rows:

   ```powershell
   python research/retention_eval.py `
     --baseline "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
     --candidate checkpoints/mixed_replay_v0001 `
     --data data/eval_frozen/heldout_city_yaml.jsonl `
            data/eval_frozen/heldout_city_river.jsonl `
            data/eval_frozen/heldout_arithmetic.jsonl `
     --max-rows 200 --device auto --per-family `
     --out research/results/retention_mixed_replay.json
   ```

2. **Checkpoint gate** with held-out copy and content eval files:

   ```powershell
   python research/checkpoint_gate.py `
     --candidate checkpoints/mixed_replay_v0001 `
     --baseline checkpoints/transformer_behavior_50m_long `
     --content-eval data/checkpoint_gate/content.jsonl `
     --copy-eval data/checkpoint_gate/copy.jsonl `
     --safety-eval data/checkpoint_gate/safety.jsonl `
     --out research/results/checkpoint_gate_mixed_replay.json --json
   ```

A pass is necessary but not sufficient for promotion. The mixer never writes
`data/active_adapter.json`, never changes product weights and never promotes a
checkpoint. Copy skill, content loss, key copy and safety must all hold on
held-out rows before a human promotion decision.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_mix_training_data.py -q
```

The tests cover deterministic output, weight quotas, dedupe, malformed-row
skipping, eval-path refusal before any read, fill/shortfall reporting, a
content-free manifest with the output hash, copy-only operation with no content
files, `--max-rows` ratio preservation and invalid arguments.

## Limitations

- Weights count rows, not tokens. A long row contributes more tokens than a
  short one, so the token-level content share can differ from the row share.
- Dedupe uses normalized text but does not detect semantic duplicates or
  near-duplicates with different wording.
- The mixer cannot create content data. If the content corpus is too small or
  unrepresentative, the mix cannot preserve behaviour it never contains.
- `--max-rows` proportional trimming is deterministic but can shift counts by
  one row per source due to rounding.
- This is an external research harness. It does not run training, and no output
  is a product artifact.
