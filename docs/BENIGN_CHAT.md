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

| category | default rows | pool | example topics |
|---|---:|---:|---|
| `identity` | 60 | 72 | who are you, what can you do, human or program, local or online, languages, limits |
| `greeting` | 60 | 72 | hello, how are you, good morning, good evening, thanks, goodbye, nice to meet you |
| `explanation` | 60 | 72 | transformer, RAG, database, API, cache, version control, encryption, JSON |
| `cooking` | 60 | 72 | boil an egg, simple pasta, store bread, rice, roast vegetables, tea, pancakes, dressing |
| `travel` | 60 | 72 | pack a carry-on, day trip, airport arrival, jet lag, budget, transit, documents, hotel |
| `coding` | 60 | 72 | functions, typo debugging, loops, lists and dictionaries, commits, tests, naming, tracebacks |
| `planning` | 60 | 72 | study week, priorities, checklists, morning routine, project plan, agenda, weekly review, habits |
| `general_knowledge` | 60 | 72 | France, boiling point, speed of light, largest planet, largest ocean, continents, photosynthesis, blue sky |

The default is 60 rows per category: 8 categories x 60 = 480 rows.
`data/` is gitignored, so generated files stay local. The generator is the
reviewable source of truth.

### 480-row recipe

Each category is authored as a small combinatorial pool rather than a flat
list:

```text
8 subject groups
  x 3 prompt phrasings per subject   (intent / phrasing variants)
  x 3 answer variants per subject     (direct answer, practical tip, short steps)
= 72 candidate (user, assistant) pairs per category
```

The seed selects 60 of the 72 candidates in each category, so the default file
is 8 x 60 = 480 rows. The remaining 12 per category are alternates: a different
seed selects a different subset without changing the authored inventory. The
full pool is validated before selection, and every normalized
`(user, assistant)` pair across all 576 candidates is unique.

Answers are 1-3 short sentences, between 24 and 300 characters. They vary
between a direct answer, one practical tip, and a short ordered list or set of
steps. They are safe, non-personal, and contain a concrete answer rather than a
refusal.

## Row shape

Each output line contains exactly:

```json
{"user": "...", "assistant": "...", "family": "identity", "source": "synthetic_template", "style": "benign_chat"}
```

`family` is one of the eight categories. `source` is always
`synthetic_template`, and `style` is always `benign_chat`.

The builder rejects answers that contain common refusal markers (`I can't`,
`I cannot`, `I won't`, `I'm sorry`, `unable to`, and similar), credential-like
fragments, or an email address, URL or phone number.

## Exact command

Run from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_benign_chat.py `
  --out data/benign_chat_v1/train.jsonl `
  --manifest data/benign_chat_v1/manifest.json `
  --per-category 60 `
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
| `--per-category` | `60` | rows per category (the authored pool has 72 per category) |
| `--seed` | `41` | deterministic combination selection and ordering |
| `--force` | off | overwrite existing output and manifest |

The same seed and `--per-category` produce byte-identical files. A different
seed selects a different subset when the pool is larger than the requested
count. `--per-category` above 72 fails with an explicit error. Raising the
ceiling requires authoring more subject groups, prompts or answer variants, not
duplicating rows.

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
            data/selfharm_replay_v1/train.jsonl `
            data/benign_chat_v1/train.jsonl `
  --copy-weight 1.0 `
  --content-weight 1.0 `
  --max-rows 120000 `
  --seed 13 `
  --shuffle `
  --out data/mixed_replay_v1/train.jsonl `
  --manifest data/mixed_replay_v1/manifest.json
```

Use the real local copy and content corpora in place of the example paths, and
keep the safety sources in the mix. The mixer reports each source separately,
so check that the benign chat source is present in `sources` and contributes
its 480 rows.

### Mixing guidance

- **Size.** 480 rows is 0.40% of a 120k mix, 2.4% of a 20k targeted mix, and
  4.8% of a 10k rehearsal mix. If the benign useful rate is still stuck below
  90%, a smaller targeted mix or repeated benign rehearsal is usually more
  effective than adding unrelated rows.
- **Weights do not duplicate.** The mixer requests
  `round(weight * unique_rows)` per source and caps each source at the rows it
  actually contains. `--content-weight` above 1 does not create extra copies of
  the benign rows. To get more exposure, run additional training epochs or
  repeat the rehearsal cycle, and record the effective passes in the run notes.
- **Keep safety replay fixed.** The goal is to change the benign side without
  changing the safety rows. Do not drop or down-weight safety replay to make
  room for benign rows.
- **Gate both sides.** Evaluate the product suite and the disjoint safety
  probes from the same checkpoint. Report the benign useful rate, safety pass
  rate and self-harm pass rate together.

**Dilution warning.** Weights are sampling multipliers, not target fractions.
A 480-row file is still small next to a 100k-row copy curriculum. Do not treat
one mix as proof that benign answering is restored. Use a targeted mix or
repeated rehearsal cycles, and evaluate both the product suite and the safety
probes.

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
  "per_category": 60,
  "total": 480,
  "categories": ["identity", "..."],
  "counts": {
    "by_category": {"identity": 60, "...": 60}
  },
  "sha256": "<64 lowercase hex characters>",
  "bytes": 125612,
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
- **Small and easily diluted.** A 480-row file is still a small fraction of a
  copy-heavy mix. Check the mixer manifest and use a targeted ratio or repeated
  rehearsal if the effect is being tested.
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

The tests cover determinism, the 480-row default, 60 rows per category, unique
`(user, assistant)` pairs across the full 576-candidate pool, required fields,
refusal-marker absence, answer length and sentence bounds, answer style
variation, safety-filter compatibility, a content-free manifest with the output
hash, the CLI default, and `--force` behavior.
