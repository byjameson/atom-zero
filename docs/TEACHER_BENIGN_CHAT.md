# Teacher-generated benign chat

**Status:** research data pipeline. `research/build_teacher_benign_chat.py`
generates and filters benign chat rows; it never trains, changes weights or
makes a capability claim.

The 1.5.0 default ships 480 authored benign chat rows from
`research/build_benign_chat.py`: 8 categories x 60 rows, selected from a 72-row
combinatorial pool per category. The prompts and answers are small, visible and
canned. That is enough for rehearsal, but a model can memorize the wording and
the pool cannot grow without hand-authoring more variants.

This builder keeps authored, non-personal user prompts and asks a larger local
teacher for the answers. The teacher writes a short, helpful reply to each
prompt; the builder cleans it, validates it and keeps it only if it passes the
safety and style rules below. The gain is freer answer wording, not new topics
or verified facts.

```text
authored benign prompt (eight categories, 96 prompts each)
  -> Qwen2.5-0.5B-Instruct (chat template, greedy)
  -> clean answer text (fences, role prefixes, line breaks)
  -> lexical + atomzero.safety.SafetyFilter validation
  -> unique prompt and unique (user, assistant) pair
  -> train.jsonl + content-free manifest.json
```

## What it builds

- **768 authored prompts**: 96 per category. Six categories use 16 topics x 6
  question frames; `identity` uses 8 topics x 12 frames and `greeting` uses 8
  salutations x 12 contexts.
- **400 accepted rows by default**: 50 per category. When `--rows` is not a
  multiple of eight, the remainder goes to the earlier categories in
  `CATEGORY_ORDER`.
- **One row per prompt**: a prompt is never reused, including after a rejected
  answer. Rejections are retried with the next unused prompt in the same
  category.
- **Greedy decoding** with a system prompt that asks for one to three helpful
  sentences and no refusal.

Each output line contains exactly:

```json
{"user": "...", "assistant": "...", "family": "identity", "source": "teacher_generated", "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct", "style": "benign_chat"}
```

`family` is one of `identity`, `greeting`, `explanation`, `cooking`, `travel`,
`coding`, `planning`, `general_knowledge`. `source` is always
`teacher_generated` and `style` is always `benign_chat`.

## Validation rules

The builder first cleans the raw reply: surrounding whitespace, Markdown
fences, an `Assistant:`/`Answer:` role prefix, a leading bullet, matching outer
quotes and internal line breaks are removed. It then requires **all** of the
following for an accepted row:

1. **Shape:** the cleaned answer is non-empty and single-line.
2. **Length:** 20-300 characters inclusive.
3. **Words:** at least five whitespace-separated words.
4. **No refusal:** none of `I can't`, `I cannot`, `I won't`, `I'm sorry`,
   `unable to`, `not able to`, `I must decline`, `I'd rather not`,
   `as an AI language model`, `cannot assist`, `can't assist`.
5. **No credential-like fragments:** none of `password=`, `api key`,
   `secret key`, `private key`, `bearer `, `sk-`, `ghp_`.
6. **No personal data:** no email address, URL, phone-like sequence, run of
   seven or more digits, or explicit personal claim such as `my address`,
   `your email`, `social security number` or `credit card number`.
7. **No operational harm:** no narrow harmful phrase (for example
   `make a bomb`, `steal a car`, `hack into`, `hurt someone`,
   `launder money`), and `atomzero.safety.SafetyFilter().check(answer)` must
   return `None`.
8. **Safe prompt:** the authored prompt must also be short, single-line,
   non-personal and pass `SafetyFilter`.
9. **No reuse:** the normalized prompt and the normalized
   `(user, assistant)` pair each appear at most once.
10. **Exact schema:** the row has exactly the six fields shown above.

The system prompt asks for one to three sentences, but the hard validator
enforces the character and word bounds rather than a sentence count. That
avoids rejecting a good two-sentence answer because of an abbreviation or a
decimal point.

A failed candidate is counted and retried with another prompt. Each category
has 96 prompts for a default quota of 50, so there are 46 spare prompts per
category. If the teacher is so unreliable that a category exhausts its pool,
the file is smaller than `--rows`; the manifest reports the `shortfall` and the
affected category.

## Exact GPU command

Run from the repository root on a host with a cached teacher. `data/` is
gitignored, so the generated files stay local and the builder is the committed
source of truth.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/build_teacher_benign_chat.py `
  --model Qwen/Qwen2.5-0.5B-Instruct `
  --device cuda `
  --rows 400 `
  --batch-size 8 `
  --max-new-tokens 96 `
  --seed 41 `
  --out data/teacher_benign_v1/train.jsonl `
  --manifest data/teacher_benign_v1/manifest.json `
  --local-files-only
```

`--local-files-only` is the default and prevents a download. If the model is
not cached, the command fails instead of silently fetching weights. Download it
once explicitly with `--no-local-files-only`, or pass a local model directory
with `--model`.

| flag | default | meaning |
|---|---|---|
| `--model` | `Qwen/Qwen2.5-0.5B-Instruct` | teacher model id or local path |
| `--device` | `cuda` | torch device; use `cpu` only for a slow smoke test |
| `--rows` | `400` | target accepted rows |
| `--batch-size` | `8` | prompts per teacher call |
| `--max-new-tokens` | `96` | generation budget per answer |
| `--seed` | `41` | prompt selection and ordering |
| `--out` | `data/teacher_benign_v1/train.jsonl` | training JSONL |
| `--manifest` | `data/teacher_benign_v1/manifest.json` | content-free manifest |
| `--local-files-only` | on | never download the teacher |
| `--force` | off | overwrite existing output and manifest |

The builder refuses to overwrite either file unless `--force` is passed.

The seed controls which authored prompts are selected and their batch order.
With the same setup and a deterministic generator, the same seed produces the
same accepted rows. A real GPU teacher is not guaranteed to be byte-identical
across different model revisions, CUDA versions, batch sizes or devices.

## Manifest

The manifest is content-free. It contains only configuration, aggregate counts
and a SHA-256 of the exact JSONL bytes. Prompt and answer text is never copied
into it.

```json
{
  "schema": "teacher_benign_chat_v1",
  "source": "teacher_generated",
  "style": "benign_chat",
  "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "seed": 41,
  "requested": 400,
  "accepted": 400,
  "rejected": 12,
  "duplicates": 0,
  "attempted": 412,
  "shortfall": 0,
  "categories": ["identity", "..."],
  "by_category": {"identity": 50, "...": 50},
  "requested_by_category": {"identity": 50, "...": 50},
  "rejected_by_category": {"identity": 1, "...": 1},
  "batch_size": 8,
  "max_new_tokens": 96,
  "sha256": "<64 lowercase hex characters>",
  "bytes": 0,
  "content_free": true
}
```

Check the generated manifest instead of copying the example counts and hashes;
they depend on the teacher, seed, prompt selection and model revision.
`rejected + accepted` does not have to equal `requested`, because a rejected
candidate is retried with another prompt. `accepted + shortfall == requested`
always holds.

## Distinction from the authored templates

| | authored templates (1.5.0) | teacher-generated |
|---|---|---|
| builder | `build_benign_chat.py` | `build_teacher_benign_chat.py` |
| `source` | `synthetic_template` | `teacher_generated` |
| answers | authored variants | model-generated, filtered |
| prompt pool | 72 per category | 96 per category |
| default rows | 480 (60 per category) | 400 (50 per category) |
| wording diversity | combinations of fixed text | free-form paraphrase |
| determinism | byte-identical for a seed | seed controls prompts; answers depend on the teacher |
| main risk | canned, memorizable wording | hallucination, repetition, style drift |

The prompts are still authored, so this does **not** add new topic coverage. It
adds answer wording inside the same eight everyday categories. It is also not a
multi-turn transcript corpus; the prompts are single turns that read like
ordinary conversation.

## Pre-training gate (hard rule)

**This output must pass the product suite before training.**

A valid row is not automatically a good training row, and a content-free
manifest is not a safety certificate. Concretely, a pilot checkpoint trained on
the proposed mix must pass the same product screen used for release candidates
(`research/product_suite.py` with the benign prompt set); only then may that mix
be used for a full training run.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/product_suite.py `
  --candidate checkpoints/pilot_with_teacher_benign `
  --baseline "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --benign data/benign_v1/prompts.jsonl `
  --safety-eval data/safety_replay_v1/eval.jsonl `
  --copy-eval data/multi_short_v1/stage_4short/eval.jsonl `
  --content-eval data/eval_tmp/city_all.jsonl `
  --device cuda `
  --max-new 64 `
  --out research/results/product_suite_pilot_teacher_benign.json `
  --json
```

The suite must report `product_ok: true`, with the benign useful rate at or
above 0.8 and the safety pass rate at or above 0.9, before the mix is used for a
full training run. A benign gain with a safety regression is a failed gate.

After full training, the resulting checkpoint still has to pass the product
suite, retention and the checkpoint gate before any human promotion decision.
Adding these rows does not replace safety replay, and it is not evidence that
the model has improved.

## Caveats

- **No factuality check.** A general-knowledge answer can be wrong. The filters
  reject refusals, personal data and operational harm; they do not verify
  truth. Do not treat the rows as a knowledge benchmark.
- **Lexical, narrow filters.** `SafetyFilter` is an operator guardrail, not a
  moderation system. The extra regexes are deliberately narrow. Review the
  manifest and a sample of rows before use.
- **Small teacher.** Qwen2.5-0.5B can repeat itself, produce bland answers or
  miss the requested style. Rejections are expected; use the accepted rate to
  judge the run.
- **Exact-duplicate detection only.** Dedupe uses case-folded, whitespace-
  collapsed text. Near-duplicate paraphrases are not detected.
- **Same topics, new wording.** The authored prompt inventory, not the teacher,
  defines coverage. Growing beyond 96 prompts per category requires authoring
  more prompts.
- **Determinism is setup-dependent.** A fake generator is deterministic in
  tests. Real decoding is greedy but still depends on the model revision,
  tokenizer, device and library versions.
- **No training here.** The builder only writes JSONL and a manifest. It does
  not download, load or train in tests, and this document makes no capability
  claim.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_teacher_benign_chat.py -q
```

The tests inject a fake generator and never load torch, transformers or a
checkpoint. They cover prompt-pool size and uniqueness, answer cleaning and
generator-output parsing, length and word bounds, refusal and unsafe-term
rejection, personal-data rejection, prompt and pair dedupe, deterministic seeds
with a fake generator, the canonical row schema, a content-free manifest with
the output hash, the CLI with an injected teacher, `--force`, the zero-row path
and the missing-local-model failure path.
