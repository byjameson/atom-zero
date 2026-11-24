# Content audit - exact-match and per-family

Date: 2026-11-24

## Why

The full-loss gate averages cross-entropy over every scored assistant token.
JSON/key-value scaffolding is 56-68% of the base NLL, so a format adapter can
pass the full-loss and content-token gates without learning to copy the
requested keys or values. This audit adds two behavior-facing measurements:

- `content_loss`: cross-entropy over alphanumeric answer tokens only;
- `exact_match`: greedy generation, parsed JSON dict equality for JSON rows and
  whitespace-normalized string equality for KV/YAML/plain rows.

`research/compare_adapters.py` reports both per family; the adapter gate can
require exact match with `--min-exact-match`.

## Active adapter audit

Candidate `ad-e396366ef725656c`, content-weighted replay training
(`--content-weight 5`, 600 steps), base `0.8.0`.

### Held-out city family (`data/eval_frozen/cross_v1.jsonl`, 48 rows)

```text
                 base      adapter
full loss        5.2549    2.8489
content loss     4.4327    3.4892
city_json        5.3507    4.0576   (content)
city_kv          3.4247    2.8651   (content)
exact match      0/48      0/48
```

### In-distribution holdout (`data/eval_frozen/indist_v1.jsonl`, 28 rows)

```text
                 base      adapter
full loss        5.3949    1.3255
content loss     4.5178    0.5708
exact match      0/28      5/28
```

The adapter memorizes some in-distribution answers (5/28) but has **zero**
exact matches on the held-out city family. The content-token loss improves,
so it has learned something about numbers and value-shaped tokens, but it has
not learned to copy the requested key names or whole answers.

A direct generation smoke on a held-out city prompt produced:

```text
prompt:  City: Ankara; population: 5747325. Return only compact JSON ...
answer:  "name": "product"
```

That is the training key vocabulary, not the prompt keys. The adapter is
therefore a **format / content-token adapter**, not a task-copy adapter.

## Gate result with behavior requirement

```text
full loss:      pass
content loss:   pass (-0.9435)
exact match:    fail (0.0, required 0.25)
overall gate:   fail / not promotable
```

The earlier local promotion used the content gate only. It remains an opt-in
`--adapter auto` registry entry; the default chat path is the base `0.8.0`
checkpoint. Before the `1.0.0` boundary, promotion must require the
exact-match gate as well.

## Reproduction

```powershell
python research/compare_adapters.py `
  --base checkpoints/base_0.8.0 `
  --data data/eval_frozen/cross_v1.jsonl `
  --adapters checkpoints/adapters_content/v0001 `
  --per-family --exact-match `
  --out research/results/content_adapter_cross_audit.json

python -m atomzero.adapter_candidates evaluate `
  --candidate-json data/diverse_autonomous_content/candidate.json `
  --comparison-json research/results/content_adapter_cross_audit.json `
  --require-content --min-exact-match 0.25 --json
```

The comparison JSON contains local prompt text and must not be committed.
The content-free aggregate is in
`research/results/content_adapter_audit_summary.json`.
