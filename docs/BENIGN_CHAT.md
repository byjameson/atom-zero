# Benign chat replay

`research/build_benign_chat.py` builds deterministic training rows for ordinary
conversation: safe prompts with short, helpful, non-personal answers.

The 1.1.0 safety-replay ratio sweep failed on the benign side. Even a 1x safety
row mix produced 68% refusals on benign open prompts. The copy and factual
content rows teach copied or source-grounded replies, but the mix contains
almost no ordinary conversational examples. The model therefore learns
"assistant style = refusal" for prompts that are not quoted or grounded.

This builder adds the missing positive examples so a fine-tune sees that a
benign question should receive a useful answer. It is rehearsal data, not a
factuality claim and not evidence of general capability. See the caveats at the
end.

## What it builds

| category | rows | example topics |
|---|---:|---|
| `identity` | 18 | who are you, what can you do |
| `greeting` | 18 | hello, how are you, good morning |
| `explanation` | 18 | transformer, RAG, database |
| `cooking` | 18 | boil an egg, simple pasta, store bread |
| `travel` | 18 | pack a carry-on, plan a day trip, airport arrival |
| `coding` | 18 | what is a function, debug a typo, explain a loop |
| `planning` | 18 | plan a study week, prioritize tasks, make a checklist |
| `general_knowledge` | 18 | capital of France, boiling point, speed of light |

The default is 18 rows per category: 8 categories x 18 = 144 rows.
`data/` is gitignored, so generated files stay local. The generator is the
reviewable source of truth.

Each category has multiple prompt groups and multiple answer variants per
group. Rows are deterministic combinations of a prompt and an answer, selected
with the seed. Every normalized `(user, assistant)` pair is unique.

## Row shape

Each output line contains exactly:

```json
{"user": "...", "assistant": "...", "family": "identity", "source": "synthetic_template", "style": "benign_chat"}
```

`family` is one of the eight categories. `source` is always
`synthetic_template`, and `style` is always `benign_chat`.

Answers are 1-3 short sentences, between 24 and 300 characters. They are safe,
non-personal, and contain a concrete answer rather than a refusal. The builder
rejects answers that contain common refusal markers (`I can't`, `I cannot`,
`I won't`, `I'm sorry`, `unable to`, and similar), credential-like fragments, or
an email address, URL or phone number.

## Exact command

Run from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_benign_chat.py `
  --out data/benign_chat_v1/train.jsonl `
  --manifest data/benign_chat_v1/manifest.json `
  --per-category 18 `
  --seed 41
```

The command refuses to overwrite an existing output or manifest. Use `--force`
to replace both:

```powershell
python research/build_benign_chat.py --force
```

| flag | default | meaning |
|---|---|---|
| `--out` | `data/benign_chat_v1/train.jsonl` | training JSONL |
| `--manifest` | `data/benign_chat_v1/manifest.json` | content-free manifest |
| `--per-category` | `18` | rows per category (the pool has 27 per category) |
| `--seed` | `41` | deterministic combination selection and ordering |
| `--force` | off | overwrite existing output and manifest |

The same seed and `--per-category` produce byte-identical files. A different
seed selects a different subset when the pool is larger than the requested
count. `--per-category` above 27 fails with an explicit error.

## Add the rows to the mixed replay

Pass `data/benign_chat_v1/train.jsonl` as an additional `--content` file to
`research/mix_training_data.py`, next to the content and safety-replay files.
The mixer preserves `user`, `assistant` and `family`, deduplicates normalized
pairs, and never reads eval paths.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/mix_training_data.py `
  --copy data/copy_curriculum_v1/train.jsonl `
  --content data/content_v1/train.jsonl `
            data/safety_replay_v1/train.jsonl `
            data/benign_chat_v1/train.jsonl `
  --copy-weight 1.0 `
  --content-weight 1.0 `
  --max-rows 120000 `
  --seed 13 `
  --shuffle `
  --out data/mixed_replay_v1/train.jsonl `
  --manifest data/mixed_replay_v1/manifest.json
```

Use the real local copy and content corpora in place of the example paths. The
mixer reports each source separately, so check that the benign chat source is
present in `sources` and contributes its rows.

**Dilution warning.** Weights are sampling multipliers, not target fractions.
The mixer never oversamples a source and never duplicates rows. A 144-row file
is tiny next to a 100k-row copy curriculum and contributes roughly 0.14% of a
120k mix. Do not treat one mix as proof that benign answering is restored. Use a
targeted mix or repeated rehearsal cycles, and evaluate both the product suite
and the safety probes.

## Hypothesis

Benign chat replay should restore useful answers while safety replay keeps
refusals:

- **Benign rows** teach the model a positive answer style for ungrounded,
  ordinary prompts. They give the fine-tune examples where the correct
  continuation is a short helpful answer, not a refusal.
- **Safety replay rows** keep the refusal behaviour on the harmful families.
  They are still needed; benign rows alone would not preserve safety.
- **Together**, the two sources make the desired behaviour conditional. A
  benign request should get an answer, and a harmful request should get a
  refusal. The mix should reduce over-refusal on benign open prompts without
  lowering the safety pass rate.

This is a hypothesis, not a measured result. It is only supported if a run
passes both sides: the product suite's benign prompts should show fewer
refusals, and the disjoint safety probes should keep their refusal pass rate.
Report both numbers from the same checkpoint. A benign improvement with a
safety regression is not a successful mix.

## Manifest

The manifest is content-free. It contains the schema, source, style, seed,
`--per-category`, total, counts per category, byte length and a SHA-256 of the
exact JSONL bytes. Prompt and answer text is never copied into it.

```json
{
  "schema": "benign_chat_v1",
  "source": "synthetic_template",
  "style": "benign_chat",
  "seed": 41,
  "per_category": 18,
  "total": 144,
  "categories": ["identity", "..."],
  "counts": {
    "by_category": {"identity": 18, "...": 18}
  },
  "sha256": "<64 lowercase hex characters>",
  "bytes": 37664,
  "content_free": true
}
```

Check the generated manifest instead of copying the example bytes and hashes;
they depend on the seed, `--per-category` and the generator version.

## Caveats

- **Synthetic templates, not a conversation corpus.** The pool is small and
  visible. It has no sampling frame, confidence interval or claim of
  representativeness.
- **Canned answers.** The answers are short and reusable. A model can memorize
  their wording, and the rows do not teach multi-turn conversation, tone
  variation or personalised advice.
- **No factuality guarantee.** General-knowledge answers are common stable
  facts, but the builder does not verify them against an external source. They
  are rehearsal examples, not a knowledge benchmark.
- **Small and easily diluted.** A 144-row file is a tiny fraction of a
  copy-heavy mix. Check the mixer manifest and use a targeted ratio if the
  effect is being tested.
- **Over-correction risk.** More answer-style rows can also make the model more
  willing on prompts it should refuse. Always mix safety replay and gate on both
  the benign product suite and the disjoint safety probes.
- **No personal data or operational content.** The rows contain no real people,
  credentials or harmful procedures.
- **No training is run here.** The builder only writes JSONL and a manifest.
  Any resulting checkpoint must still pass retention and the checkpoint gate
  before a human promotion decision.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_benign_chat.py -q
```

The tests cover determinism, default and custom counts, unique pairs, required
fields, refusal-marker absence, answer length and sentence bounds, safety-filter
compatibility, a content-free manifest with the output hash, and `--force`
behavior.
