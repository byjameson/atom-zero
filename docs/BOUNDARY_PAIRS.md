# Contrastive boundary pairs

`research/build_boundary_pairs.py` builds paired **unsafe** and **benign** rows
for four safety families. It applies the contrastive self-harm idea to the rest
of the safety surface: an unsafe request gets a short refusal that names the
boundary, while a near-neighbour benign request in the same topic area gets a
helpful, concrete answer.

The failure this targets is the trade-off seen with self-harm replay. A blanket
refusal style can fix a crisis probe and still cost ordinary conversation. A
pair teaches the boundary instead: the model sees two requests that look alike
in topic but differ in intent, and only one should be refused.

This is narrow, template-based rehearsal. It is **not** a moderation benchmark,
not evidence that a checkpoint is generally safe, and not a substitute for
human review. See the caveats at the end.

## What it builds

| family | unsafe request | benign near-neighbour |
|---|---|---|
| `prompt_injection` | asks to ignore rules, reveal hidden setup text or expose system prompts | asks how to write, version, test or document prompts |
| `secret_exfiltration` | asks for keys, tokens, passwords or secret values | asks how to store, rotate, revoke or audit credentials safely |
| `privacy_personal_data` | asks to find, profile or expose a private person | asks how to protect, redact or request personal data |
| `harmful_or_illegal` | asks for help with a crime, evasion or bypass | asks about home security, reporting, permits or lawful recovery |

Each selected template pair emits exactly two adjacent rows:

| kind | answer |
|---|---|
| `unsafe` | one or two sentences: a refusal marker, the family boundary, and a safe alternative |
| `benign` | one to three sentences: a helpful, concrete answer with no refusal markers |

The default is 12 pairs per family: 24 rows per family and 96 rows total. The
default seed is `67`. A different seed selects and orders different template
pairs and answer variants. `--per-family` controls the pair count; up to 14
pairs per family are available.

## Row schema

Every row contains exactly:

```json
{"user": "<prompt>", "assistant": "<answer>", "family": "prompt_injection", "kind": "unsafe", "source": "synthetic_template"}
```

The benign row uses `"kind": "benign"` and the same family. The source is
always `synthetic_template`.

Unsafe answers are short refusals. They:

- contain a refusal marker such as `I can't` or `I won't`;
- name the boundary, for example `I won't expose system prompts` or
  `I can't share passwords, API keys, or access tokens`;
- offer a safe alternative, for example help with prompt design or credential
  rotation;
- contain no method, procedure, credential value or operational fragment.

Benign prompts are ordinary requests about the same topic, not mild rewrites
of the unsafe prompt. The builder rejects any pair whose prompt token overlap
is 0.5 or higher. Benign answers are concrete and useful, and they contain no
refusal marker and no boundary marker.

No prompt contains a real secret, a real person, an operational method or an
instruction. Every prompt is unique, and every `(user, assistant)` pair is
unique.

## Exact command

Run from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_boundary_pairs.py
```

To choose paths and size explicitly:

```powershell
python research/build_boundary_pairs.py `
  --out data/boundary_pairs_v1/train.jsonl `
  --manifest data/boundary_pairs_v1/manifest.json `
  --per-family 12 `
  --seed 67
```

The command refuses to overwrite an existing output or manifest. Use `--force`
to replace both:

```powershell
python research/build_boundary_pairs.py --force
```

| flag | default | meaning |
|---|---|---|
| `--out` | `data/boundary_pairs_v1/train.jsonl` | JSONL output |
| `--per-family` | `12` | unsafe/benign pairs per family (max 14) |
| `--seed` | `67` | deterministic selection, ordering and answer variants |
| `--manifest` | `data/boundary_pairs_v1/manifest.json` | content-free manifest |
| `--force` | off | overwrite existing output and manifest files |

The same seed and `--per-family` produce byte-identical files.

## Mixing into the benign mix

The rows use the repository `user` / `assistant` schema, so
`research/mix_training_data.py` can add them to an existing benign mix. Replace
`data/mixed_benign_a/train.jsonl` with the actual mixed file you are using.
The mixer keeps `user`, `assistant` and `family`; it drops `kind` and `source`.

Start with one copy:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/mix_training_data.py `
  --copy data/mixed_benign_a/train.jsonl `
  --content data/boundary_pairs_v1/train.jsonl `
  --copy-weight 1.0 `
  --content-weight 1.0 `
  --seed 67 `
  --shuffle `
  --out data/mixed_benign_boundary_v1/train.jsonl `
  --manifest data/mixed_benign_boundary_v1/manifest.json
```

Ninety-six rows are a tiny share of a multi-thousand-row benign mix. If the
safety probes still drift, a **small repeat** is reasonable: repeat the file
twice, never more, for 192 rows. Build the repeated copy first and keep it out
of version control:

```powershell
$rows = Get-Content data/boundary_pairs_v1/train.jsonl
($rows + $rows) | Set-Content data/boundary_pairs_v1/train_x2.jsonl -Encoding utf8
```

Then pass `data/boundary_pairs_v1/train_x2.jsonl` as `--content`. A 2x repeat is
still a narrow signal, not a guarantee. Do not raise the multiplier repeatedly
until the probes pass and call the result safe.

Keep monitoring the benign suite. The point of the contrast is to preserve
helpful answers, so an improvement on unsafe probes that comes with a new
refusal on ordinary requests is a regression. The eval and probe files are
never part of a training mix.

## Manifest

The manifest is content-free. It records counts per family and kind, the seed,
the pair count, the total, the file hash and the byte count. No prompt or answer
text is copied into it.

```json
{
  "schema": "boundary_pairs_v1",
  "source": "synthetic_template",
  "seed": 67,
  "per_family": 12,
  "families": ["prompt_injection", "secret_exfiltration", "privacy_personal_data", "harmful_or_illegal"],
  "kinds": ["unsafe", "benign"],
  "counts": {
    "by_family": {
      "prompt_injection": {"unsafe": 12, "benign": 12},
      "secret_exfiltration": {"unsafe": 12, "benign": 12},
      "privacy_personal_data": {"unsafe": 12, "benign": 12},
      "harmful_or_illegal": {"unsafe": 12, "benign": 12}
    },
    "by_kind": {"unsafe": 48, "benign": 48}
  },
  "pairs": 48,
  "total": 96,
  "sha256": "<64 lowercase hex characters>",
  "bytes": 12345,
  "content_free": true,
  "paired": true
}
```

Check the generated manifest instead of copying the example hash or byte count;
they depend on the seed, `--per-family` and the generator version.

## Caveats

- **Narrow rehearsal, not a moderation benchmark.** The builder uses a small,
  visible set of templates. It has no sampling frame, no independent labels and
  no claim of representativeness. Passing these rows does not show that a
  checkpoint is generally safe.
- **Lexical checks only.** The tests check refusal markers, boundary phrases,
  method words, prompt uniqueness and prompt overlap. They do not detect every
  unsafe paraphrase, translation, multi-turn escalation or context-dependent
  failure.
- **Visible templates.** Both kinds are visible to the reader. There is no
  hidden benchmark here.
- **Small and easily diluted.** Ninety-six rows are a tiny share of a large mix.
  Use a small repeat at most and keep measuring benign behaviour.
- **No real secrets, people or methods.** Every prompt and answer is synthetic.
  No prompt contains a credential value, a real identity, a procedure or an
  encoded payload.
- **No training is run here.** The builder only writes JSONL and a manifest.
  Any resulting checkpoint must still pass retention, the benign suite and the
  checkpoint gate before a human promotion decision.
- **Human review still applies.** Read the generated refusals and the benign
  answers before training. A correct marker is not the same as a good reply.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_boundary_pairs.py -q
```

The tests cover deterministic paired counts, seed and size controls, unsafe
refusal markers and method-word exclusion, refusal-free benign answers,
prompt uniqueness and near-neighbour separation, a content-free manifest with
hashes, and `--force` behaviour.
