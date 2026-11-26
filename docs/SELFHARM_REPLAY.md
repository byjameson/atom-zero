# Contrastive self-harm replay

`research/build_selfharm_replay.py` builds paired **crisis** and **benign
support** rows for a narrow safety rehearsal. It is aimed at a specific failure:
Variant A (63M mixed-replay) has the best product profile so far
(95.6% benign, 61.5% 4-key exact, retention 0.759) but a crisis prompt can still
receive unrelated coding advice. Generic safety replay did not teach the
boundary, so this builder contrasts a crisis prompt that must get a supportive
refusal with a near-neighbour everyday support question that must get a helpful
answer.

This is narrow, template-based safety rehearsal. It is **not** a moderation
benchmark, not crisis care and not evidence that a checkpoint is generally
safe. Every generated crisis response must be reviewed by a human before it is
used for training or evaluation. See the caveats at the end.

## What it builds

Each selected template pair emits exactly two adjacent rows:

| kind | family | prompt | answer |
|---|---|---|---|
| `crisis` | `self_harm_crisis` | hopelessness or a request for help with self-harm | 2-3 sentence supportive refusal |
| `benign` | `benign_support` | a distinct everyday support question | short, concrete, helpful answer |

The default is 8 pairs: 8 crisis rows and 8 benign rows in the training file.
The optional eval file uses a separate, held-out pair pool with the same shape.

Crisis answers:

- start with empathy;
- contain an explicit boundary such as `I can't help with ending your life`
  or `I won't help you end your life`;
- point to a local crisis line or emergency services;
- offer to stay with the person while they take a safe next step;
- are under 400 characters and contain no method, instruction or graphic
  detail.

Benign answers are short and concrete. They never contain refusal markers, so
the model sees that ordinary support questions should not be answered like a
crisis.

## Row schema

Training and eval rows contain exactly:

```json
{"user": "<prompt>", "assistant": "<answer>", "family": "self_harm_crisis", "kind": "crisis", "source": "synthetic_contrastive_selfharm"}
```

The benign rows use `"family": "benign_support"` and `"kind": "benign"`.

The prompts are synthetic and contain no real people, method detail or
operational content. The benign prompts are separate questions about grief,
warning signs, withdrawal, burnout and similar topics; they are not crisis
prompts with a mild rewrite. Every prompt is unique, and no benign prompt
contains a crisis term such as `suicide`, `self-harm` or `end my life`.

## Exact command

Run from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_selfharm_replay.py `
  --train-out data/selfharm_replay_v1/train.jsonl `
  --manifest data/selfharm_replay_v1/manifest.json `
  --per-family 8 `
  --seed 53
```

To write the optional held-out eval split as well:

```powershell
python research/build_selfharm_replay.py `
  --train-out data/selfharm_replay_v1/train.jsonl `
  --eval-out data/selfharm_replay_v1/eval.jsonl `
  --manifest data/selfharm_replay_v1/manifest.json `
  --per-family 8 `
  --seed 53
```

The command refuses to overwrite an existing train file, eval file or
manifest. Use `--force` to replace all of them:

```powershell
python research/build_selfharm_replay.py --force
```

| flag | default | meaning |
|---|---|---|
| `--train-out` | `data/selfharm_replay_v1/train.jsonl` | paired training JSONL |
| `--eval-out` | off | optional held-out eval JSONL, never for training |
| `--per-family` | `8` | crisis/benign pairs to emit (max 12 train, 8 eval) |
| `--seed` | `53` | deterministic selection, ordering and answer variant seed |
| `--manifest` | `data/selfharm_replay_v1/manifest.json` | content-free manifest |
| `--force` | off | overwrite existing output and manifest files |

The same seed, `--per-family` and generator version produce byte-identical
files. A different seed changes the selected pairs, their order and the answer
variants.

## Mixing into `data/mixed_benign_*`

The training rows use the repository `user` / `assistant` schema, so
`research/mix_training_data.py` can add them to an existing mixed benign file.
Replace `data/mixed_benign_a/train.jsonl` with the actual mixed file you are
using. The mixer preserves `user`, `assistant` and `family`; it drops the
extra `kind` and `source` fields.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/mix_training_data.py `
  --copy data/mixed_benign_a/train.jsonl `
  --content data/selfharm_replay_v1/train.jsonl `
  --copy-weight 1.0 `
  --content-weight 1.0 `
  --seed 53 `
  --shuffle `
  --out data/mixed_benign_selfharm_v1/train.jsonl `
  --manifest data/mixed_benign_selfharm_v1/manifest.json
```

Weights are sampling multipliers, not target fractions. With `1.0 / 1.0` the
mixer uses every unique row from both files. The 16 self-harm rows are still a
tiny share of a multi-thousand-row benign mix (about 0.2% for a 7k-row base).
The mixer never duplicates rows to raise that share. If you need a stronger
signal, use a smaller targeted mix, repeated rehearsal cycles with different
seeds, or a larger human-reviewed crisis corpus. Do not oversample these
templates and call the result safe.

For a quick one-off experiment you can append the 16 rows to a copy of a
`data/mixed_benign_*` file with a small inline script, then shuffle with a
recorded seed and write a content-free manifest. Do not edit the original mix in
place, and keep `data/selfharm_replay_v1/eval.jsonl` out of every training mix.
The mixer refuses eval-looking paths before opening them; an inline script must
enforce the same rule itself.

## Manifest

The manifest is content-free. The top-level `counts`, `total`, `sha256` and
`bytes` describe the training file. If an eval file is written,
`splits.eval` describes it and `total_all` reports both splits. No prompt,
answer or source text is copied into the manifest.

```json
{
  "schema": "selfharm_replay_v1",
  "source": "synthetic_contrastive_selfharm",
  "seed": 53,
  "per_family": 8,
  "kinds": ["crisis", "benign"],
  "counts": {"by_kind": {"crisis": 8, "benign": 8}},
  "total": 16,
  "sha256": "<64 lowercase hex characters>",
  "bytes": 6297,
  "content_free": true,
  "paired": true,
  "splits": {
    "train": {
      "count": 16,
      "sha256": "<64 lowercase hex characters>",
      "bytes": 6297,
      "counts": {"by_kind": {"crisis": 8, "benign": 8}}
    }
  }
}
```

Check the generated manifest instead of copying the example hash or byte count;
they depend on the seed, `--per-family` and the generator version.

## Caveats

- **Narrow template rehearsal with human review.** The builder uses a small,
  visible set of templates. It has no sampling frame, clinical validity or
  claim of representativeness. A human must review the generated crisis
  responses and the trained behaviour before use.
- **Crisis responses must be reviewed.** A refusal marker plus a crisis-line
  phrase is not crisis care. Check tone, clarity, local resource accuracy and
  whether the response stays supportive under paraphrase and multi-turn
  pressure.
- **Lexical checks only.** The tests check refusal markers, help-seeking
  phrases, method words and refusal-free benign answers. They do not detect
  every unsafe paraphrase, translation or context-dependent failure.
- **Visible templates.** The eval split is held out from training, not from the
  reader. It is not a hidden benchmark.
- **Small.** Sixteen training rows cannot cover the space of crisis language.
  The signal is easily diluted in a large mixed file.
- **No real people or operational content.** Every prompt and answer is
  synthetic. No prompt contains method detail, instructions or graphic content.
- **No training is run here.** The builder only writes JSONL and a manifest.
  Any resulting checkpoint must still pass retention and the checkpoint gate
  before a human promotion decision.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_selfharm_replay.py -q
```

The tests cover deterministic paired counts, seed and size controls, prompt
uniqueness and crisis/benign separation, supportive refusal content, method-word
exclusion, refusal-free benign answers, the local safety filter, a content-free
manifest with hashes, disjoint optional eval rows and `--force` behavior.
