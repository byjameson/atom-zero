# Adapter task-family separation gate

The cross-distribution adapter candidate must prove that the held-out task
family never appears in training. A different file name or a random tail split
is not proof: the same template can leak through both sides.

This gate is metadata-only. It never trains an adapter, never applies one and
never changes weights.

## Row schema

A verified teacher row may carry an optional, content-free `family` label:

```json
{"user": "What is the launch code?",
 "assistant": "The launch code is ORBIT-42.",
 "family": "person_json",
 "verification": {"verified": true, "promotable": true,
                  "status": "verified"}}
```

Example families are `person_json`, `product_kv` and `city_json`. The label is
a short identifier, not a prompt, an answer or a note.

Validation rules:

- a missing or `null` family becomes `unspecified`;
- the value must be a string;
- surrounding whitespace is stripped and the result must not be blank;
- control characters (including tabs, newlines and DEL) are rejected;
- labels longer than 64 characters are rejected.

Invalid labels raise `ValueError` during export. The exporter does not guess a
family from `user` or `assistant` text.

## Hard cross-distribution check

`build_cross_candidate` (and `export-cross`) computes sorted unique
`train_families` and `eval_families`. If a specified eval family also appears
in the training families, the build fails with a clear error:

```text
task-family leakage: eval families also appear in training: city_json
```

This is a hard failure, not a warning, and no candidate files are written.

`unspecified` is exempt from the conflict check so existing datasets without
family metadata keep working. It is **not** evidence of separation: a candidate
is marked `family_separation: true` only when both sides are fully labeled and
the sets are disjoint. If either side contains `unspecified` or the random
split path was used, separation is not claimed.

Metadata fields:

```json
{"split_method": "explicit_cross_distribution",
 "family_separation": true,
 "train_families": ["person_json"],
 "eval_families": ["product_kv"],
 "train_family_counts": {"person_json": 94},
 "eval_family_counts": {"product_kv": 23}}
```

The random `export` path reports `family_separation: "not_applicable"` because
a random split may mix families. It still reports the observed family lists and
counts, but never claims a cross-family separation.

## CLI

```powershell
atomzero-adapter-candidate export-cross `
  --train-jsonl data/teacher_json.jsonl `
  --eval-jsonl data/teacher_keyvalue.jsonl `
  --out-dir data/teacher_cross_candidate --json
```

The JSON payload includes content-free family counts:

```json
{"candidate": {"family_separation": true, "...": "..."},
 "family_counts": {"train": {"person_json": 94},
                   "eval": {"product_kv": 23}},
 "paths": {"candidate_json": "...", "train_jsonl": "...",
           "eval_jsonl": "..."},
 "auto_apply": false,
 "weights_changed": false}
```

`candidate.json` contains the family metadata. Prompt and answer text stays in
the local `train.jsonl` and `eval.jsonl` files, as before.

## Helper

`family_summary(rows)` returns counts by validated family, sorted by label:

```python
family_summary([
    {"user": "q", "assistant": "a", "family": "person_json"},
    {"user": "q2", "assistant": "a2"},
])
# {"person_json": 1, "unspecified": 1}
```

## Honest limits

- Family labels are self-declared. A wrong or too-broad label cannot be
  detected from metadata alone.
- `unspecified` rows are allowed for compatibility. They may hide leakage, so
  they block the `family_separation: true` claim.
- The gate proves only that the declared eval family did not appear in the
  declared training families. It does not prove semantic novelty or quality.
- No model weights change here; training and promotion remain separate offline
  steps.
