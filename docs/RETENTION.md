# Content retention evaluation

## Why this exists

The 0.80.0 research checkpoint (`checkpoints/fullft_upcycle8l_v1`, 63.4M
parameters) was fully fine-tuned on copy data. It is much better at copying keys
than the product 49.2M `0.8.0` base, but copy skill is not the only behaviour a
product base needs.

Exact match alone is too weak to decide whether the candidate still behaves like
the product. On the city/QA sets both models can score 0 exact, which does not
mean they model the answers equally well. Token loss on the assistant's content
tokens is a denser signal: it measures whether the candidate still assigns
probability to the answer text, even when greedy generation fails.

`research/retention_eval.py` compares a candidate with the product baseline on
the same rows and reports:

| metric | meaning |
|---|---|
| `loss` | assistant-turn cross-entropy; user turns and padding are masked |
| `content_loss` | cross-entropy on non-syntax tokens whose decoded text contains an alphanumeric character |
| `exact_match` | greedy generation, normalized (JSON/KV/whitespace) exact match |
| `key_copy` | generated key-name set equals the expected key-name set; only rows whose answer has keys are counted |
| `key_recall` | mean fraction of expected keys present in the generation |
| `retention_ratio` | `candidate_content_loss / baseline_content_loss` |
| `retention_ok` | true when `retention_ratio <= --threshold` (default `1.10`) |

The JSON report is content-free: counts, losses, rates and family names only.
It never contains prompts, assistant answers, keys or values.

## Exact command: 49M product base vs 63M candidate

Run from the repository root on Windows PowerShell. The product base is the
`0.8.0` cache, and the candidate is the 0.80.0 63M research checkpoint:

```powershell
python research/retention_eval.py --baseline "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" --candidate checkpoints/fullft_upcycle8l_v1 --data data/eval_frozen/heldout_city_yaml.jsonl data/eval_frozen/heldout_city_river.jsonl data/eval_frozen/heldout_arithmetic.jsonl --max-rows 200 --device auto --per-family --out research/results/retention_49m_vs_63m.json
```

On Linux or macOS the same command uses `$HOME/.cache/atomzero/atom-zero-49m`.
If the candidate lives elsewhere, pass its absolute path. Both directories must
contain `model.pt`; the script reads the tokenizer from the baseline and uses it
for both models so token losses stay comparable.

To see the copy side of the same pair, run a second report over a copy eval set:

```powershell
python research/retention_eval.py --baseline "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" --candidate checkpoints/fullft_upcycle8l_v1 --data data/staged_copy_v1/mixed_eval.jsonl --max-rows 200 --device auto --per-family --out research/results/retention_copy_49m_vs_63m.json
```

`--data` accepts one or more JSONL files. Rows are read in the order given and
the combined stream is truncated to `--max-rows` (default 200), so put the most
important file first. Optional `family` labels drive `--per-family`; rows without
a family still count in the overall totals. The frozen eval files live under
`data/` (gitignored); any local JSONL with `user`, `assistant` and an optional
`family` field works.

## How to read `retention_ratio`

`retention_ratio = candidate_content_loss / baseline_content_loss`

- `1.00`: the candidate models content tokens exactly as well as the baseline.
- `< 1.00`: the candidate is better on content loss.
- `> 1.00`: the candidate is worse.
- `<= 1.10`: passes the default gate, i.e. no worse than 10% on content loss.
- `> 1.10`: fails the default gate.

Both losses are average negative log-likelihood in nats per content token, so
the ratio is unit-free. For example, a baseline content loss of `1.20` and a
candidate loss of `1.44` gives a ratio of `1.20`, which fails the default gate.
A candidate loss of `1.26` gives `1.05`, which passes. Change the ceiling with
`--threshold`; `1.00` means "no regression at all" and `1.25` allows a 25%
regression.

`retention_ok` is `null` when the ratio cannot be computed, for example when
either model has no content tokens or the baseline content loss is zero. A
`null` result is not a pass.

`--per-family` reports the same ratio inside each family. Read the per-family
block, not only the overall number: a small QA family can regress sharply while
the overall ratio stays near 1.0 because a large copy family dominates. The
threshold applies to every family; a family with no keyed rows reports
`key_copy: null` rather than a misleading zero.

## Retention is necessary, not sufficient, for promotion

A candidate may be promoted only if retention passes, but passing retention does
not promote it. Promotion requires all of the following:

1. **Retention:** `retention_ok` is true overall and for every product-relevant
   family (or an explicitly documented, reviewed exception).
2. **Capability gain:** copy exact match and key-copy improve enough to justify
   a new base, measured on held-out rows.
3. **Answerability and QA:** source-cited QA, answerability policy and refusal
   behaviour do not regress.
4. **Safety and product gates:** the benchmark, safety, citation and full test
   suites pass.

A candidate that fails retention must not be promoted just because its copy
score is higher. The current product decision is to keep `0.8.0` as the base
until a candidate passes every gate as a complete assistant, not only the copy
task.

## Caveats

- Token loss measures likelihood, not factuality. A model can assign high
  probability to a fluent wrong answer.
- Exact match is brittle. Zero exact match on both models is not evidence of
  equal content behaviour; use `content_loss` and per-family results.
- Key-copy applies only to rows whose expected answer has parseable keys. Rows
  without keys are skipped and reported in `skipped`.
- The candidate is loaded with the baseline tokenizer. Do not compare models
  with different tokenizers or vocabularies; the token losses are not comparable.
- The 6-layer 49M base and 8-layer 63M candidate have different capacity. The
  ratio is a fair same-data, same-tokenizer comparison, not proof that the
  architectures have identical behaviour.
- Sample size is small by design. Do not tune on the eval files, and keep a
  frozen holdout for promotion decisions.
- The report is content-free, so it is safe to commit. Raw rows and generations
  are not written anywhere by this script.
- The 63M checkpoint is a copy-specialized research artifact. A large content
  regression is expected; this tool quantifies it instead of hiding it behind
  copy exact match.
