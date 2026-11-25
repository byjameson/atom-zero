# Optional pretokenized training cache

Status: opt-in research tooling. Default training behaviour is unchanged.

## Purpose

The Vast.ai 24-layer full fine-tune run is data-loader bound: every optimizer
step formats rows, runs the tokenizer, builds the assistant mask and computes
per-token weights on the CPU. When that work is slower than the forward and
backward pass, GPU utilization fluctuates and paid GPU time is wasted.

`research/pretokenize.py` moves the per-row work out of the training loop. The
full fine-tune control (`research/finetune_copy_control.py`) can then draw
input ids, labels and token weights from a compact memmap cache. Tokenization
happens once per dataset instead of once per epoch.

This does not change the model, the loss or the default path. It is a loader
optimization, not a new training recipe.

## What the cache contains

A cache directory holds four flat little-endian `.npy` memmap arrays and a
content-free `meta.json`:

| file | dtype | contents |
|---|---|---|
| `input_ids.npy` | int32 | concatenated model inputs, one row after another |
| `labels.npy` | int32 | concatenated targets, `-100` on ignored labels |
| `weights.npy` | float16 | per-token loss weights from `_token_weights` |
| `lengths.npy` | int32 | token count per cached row |
| `meta.json` | JSON | counts, max_seq, content weight, tokenizer/data SHA-256, schema, per-file hashes |

Rows are encoded with the trainer's own
`research.train_adapter_candidate._encode` and `_token_weights`, so the cache
round-trip is equivalent to the on-the-fly path. Nothing is padded: the
trainer already encodes rows individually, and the cache preserves that.

`meta.json` contains no prompt or answer text. It records how many rows were
read, cached and skipped, the token count, the tokenizer SHA-256, the source
data SHA-256, the schema (`atomzero.pretokenized.v1`), the per-file SHA-256
values and an aggregate `sha256`. The output is deterministic for the same
input and flags: rebuilding the cache twice produces byte-identical files.

Rows that produce no scorable label are skipped and counted under
`skip_reasons`. `--manifest` writes an optional per-row manifest with only
`index`, `kept`, token count and skip reason; it never includes text.

## Build a cache

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/pretokenize.py `
  --data data/copy_curriculum_v1/train.jsonl `
  --tokenizer "$env:USERPROFILE\.cache\atomzero\atom-zero-49m\chat_en_bpe8k.json" `
  --out data/pretok/copy_curriculum_v1 `
  --max-seq 256 `
  --content-weight 5.0 `
  --manifest
```

A bare `--manifest` writes `manifest.json` inside the cache directory. Pass a
path (`--manifest some\dir\manifest.json`) to write it elsewhere. The summary
printed to stdout is content-free:

```json
{
  "cache": "data/pretok/copy_curriculum_v1",
  "cached_rows": 12000,
  "skipped_rows": 0,
  "skip_reasons": {"too_short": 0, "no_scorable_token": 0},
  "tokens": 2138400,
  "max_seq": 256,
  "content_weight": 5.0,
  "sha256": "..."
}
```

`data/` is gitignored, so caches built there are not committed.

## Train from a cache

Add `--cache-dir` to the existing full fine-tune control command:

```powershell
python research/finetune_copy_control.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/copy_curriculum_v1/train.jsonl `
  --eval data/copy_curriculum_v1/cross.jsonl `
  --cache-dir data/pretok/copy_curriculum_v1 `
  --out checkpoints/fullft_copy_control/v0001 `
  --steps 100 --lr 1e-4 --batch-size 4 --max-seq 256 `
  --content-weight 5.0 --seed 23 --freeze none `
  --result research/results/full_finetune_control.json `
  --comparison research/results/full_finetune_control_comparison.json
```

The trainer:

- loads and hash-checks the cache before training;
- requires `--max-seq` and `--content-weight` to match the cache metadata,
  because the cached labels and weights are fixed at build time;
- refuses a cache whose recorded source-data SHA-256 differs from `--train`;
- records `cache_rows`, `cache_dir` and the cache SHA-256 in the training
  summary / checkpoint metadata.

Without `--cache-dir`, the trainer tokenizes exactly as before. No default
flag, command or result field changes.

## How to measure the speedup

Do not assume a speedup; measure it on the same machine, data, seed and step
count. A useful protocol:

1. Run the on-the-fly command and the `--cache-dir` command with `--steps 200`
   (or another fixed count), `--seed 23`, the same batch size, `--max-seq` and
   `--content-weight`.
2. Add a short warm-up (for example the first 20 steps) and compare the
   remaining steps. The trainer already prints `elapsed_seconds` every
   `log_every` steps.
3. Repeat each configuration three times and report the median wall-clock
   seconds per optimizer step and tokens per second
   (`cached_rows * max_seq / seconds`, or the logged token count).
4. Sample GPU utilization in parallel, for example
   `nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv -l 1`.
   The cache should raise the average and reduce the dips; if utilization is
   already near saturation, expect little gain.
5. Confirm correctness first: a one-step cached run and a one-step on-the-fly
   run with the same seed must produce the same loss within float tolerance.
   `tests/test_pretokenize.py::test_trainer_accepts_cache_and_matches_on_the_fly_loss`
   does this on a tiny model.

Expected result: a modest end-to-end gain, largest when CPU tokenization
dominates. With `--content-weight > 1`, `_token_weights` decodes every scored
token individually, so removing it from the loop is the biggest saving. On the
24L Vast.ai runs expect something in the low single digits (roughly 1.1–2x
per step) rather than an order of magnitude; the forward/backward pass and the
remaining tensor copies still cost time. The only honest number is the one
measured with the protocol above.

## Correctness caveat

- The per-row weighted cross-entropy is the same expression as the on-the-fly
  path: weighted token loss divided by the summed token weights, averaged over
  valid rows. There is no padding in either path.
- If every row is scorable, a fixed seed produces the same row order and the
  same optimizer step. Skipped rows are absent from the cache, so the epoch
  length and batch composition can differ from the on-the-fly path. The
  summary reports both `train_examples` (source rows read) and `cache_rows`
  (rows actually drawn).
- Token weights are stored as float16. `1.0`, `2.0`, `5.0` and other
  exactly-representable values round-trip exactly. An arbitrary
  `--content-weight` may introduce a tiny weight rounding difference; use the
  equivalence test or rebuild with a representable value when bitwise identity
  matters.
- The cache is tied to the tokenizer, `--max-seq`, `--content-weight` and the
  source JSONL. Rebuild it after changing any of them. The trainer rejects
  mismatched `--max-seq` / `--content-weight`, and rejects a different
  `--train` SHA-256.
- The cache stores token ids, which are reversible in principle. `meta.json`
  and the manifest are content-free, but treat the payload as sensitive
  training data.

## Eval files are never cached as training data

`--data` is a **training** JSONL only. Evaluation stays on the direct path:
`--eval` is read by `read_rows` and scored by `research.compare_adapters`
without going through the cache. Never point `--data` at an eval file, and
never mix eval rows into a training cache.

The source-data SHA-256 in `meta.json` makes an accidental mismatch visible:
the trainer refuses a cache built from a different `--train` file. This guards
against reusing an eval-derived cache for training, but it is not a substitute
for keeping train/eval files separate.

The cache is also never used as an evaluation cache: `compare_adapters`
tokenizes the eval JSONL directly so metrics stay comparable with earlier
runs.

## Failure modes

- Missing cache directory: `FileNotFoundError: pretokenized cache directory
  not found ...` with the build command.
- Missing or corrupt `meta.json`, wrong schema, modified payload or hash
  mismatch: `ValueError` naming the cache and the rebuild command.
- Empty cache (every row skipped): `ValueError: no scorable rows ...`.
- Mismatched trainer settings: `ValueError` naming `--max-seq` or
  `--content-weight` and asking for a rebuild.

## Scope

This is research tooling for the full fine-tune control. It does not train a
model by itself, does not download anything, does not change base or product
weights, and does not auto-promote a checkpoint.
