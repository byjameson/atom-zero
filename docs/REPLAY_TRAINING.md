# Replay-aware adapter training

The cross-distribution adapter run exposed catastrophic forgetting: training on
the JSON-format teacher split moved in-distribution loss `6.5393 -> 1.2733`,
but the same adapter regressed the held-out key-value split `3.3621 -> 9.9921`.
The held-out gate blocked promotion, as it should. This branch adds a bounded
replay mix so the trainer can keep some older-distribution rows in the update.

The mechanism is a mitigation, not a retention guarantee. It does not change
base weights, does not apply an adapter to the product and does not
auto-promote anything.

## What changed

- `research/replay_sampler.py` provides `sample_steps` and `summarize_mix`.
- `research/train_adapter_candidate.py` accepts `--replay-jsonl` and
  `--replay-weight` (default `0.25`).
- The adapter `config.json` and result JSON record `replay_examples` and
  `replay_weight`, plus the actual primary/replay step counts in `train_mix`.

With no `--replay-jsonl`, replay rows are empty and the trainer cycles primary
rows exactly as before, so existing commands keep their behaviour.

## Reproduce

```powershell
python research/train_adapter_candidate.py `
  --base "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --train data/adapter_candidates/train.jsonl `
  --eval data/adapter_candidates/eval.jsonl `
  --replay-jsonl data/adapter_candidates/replay.jsonl `
  --replay-weight 0.25 `
  --out checkpoints/adapters_candidate/replay_v0001 `
  --steps 300 --lr 0.001 --rank 8 --alpha 16 `
  --result research/results/adapter_candidate_replay.json `
  --comparison research/results/adapter_candidate_replay_comparison.json

atomzero-adapter-candidate evaluate `
  --candidate-json data/adapter_candidates/candidate.json `
  --comparison-json research/results/adapter_candidate_replay_comparison.json --json
```

`--eval` is still a single holdout. Run the trainer twice with different
`--eval` files (target distribution and replay distribution) when retention
matters; the trainer itself does not claim a cross-distribution result.

## Sampler API

```python
from research.replay_sampler import sample_steps, summarize_mix

sequence = sample_steps(
    primary_rows,
    replay_rows,
    steps=300,
    replay_weight=0.25,
    seed=13,
)
mix = summarize_mix(sequence, primary_rows, replay_rows)
# {"primary": 225, "replay": 75}
```

Rules:

- `primary_rows` must be a non-empty sequence of mappings.
- `replay_rows` may be empty or `None`; then only primary rows are used.
- `replay_weight` must be a finite number in `0.0..1.0`. `steps` must be a
  non-negative integer and `seed` an integer.
- The response is a new list of row references. Input lists and rows are not
  mutated.
- With replay enabled, `steps * replay_weight` rounded to the nearest step
  (half up) come from replay rows and the rest from primary rows. The exact
  positions are shuffled with `random.Random(seed)`, so the same inputs and
  seed always produce the same sequence. Very small runs can be off by one step
  because the count is rounded; for example 10 steps at weight `0.25` yields 3
  replay steps (30%).
- With replay disabled, primary rows are cycled in input order. This preserves
  the legacy trainer behaviour.

`summarize_mix` matches sampled rows to their source by object identity, which
is valid because `sample_steps` returns references rather than copies. A row
present in both sources is counted as primary; a row from neither source raises
`ValueError`.

## Recommended mix

- Start with `--replay-weight 0.25`; it keeps the target distribution dominant.
- Use a replay file from a distribution that previously passed a gate, not
  freshly generated or unverified rows.
- Keep the replay rows out of `--eval` so the holdout still measures unseen
  behaviour.
- Compare loss on both the target holdout and the replay holdout. A lower
  target loss alone does not show that forgetting was reduced.
- Raise the weight only if the replay holdout still regresses, and remember
  that a higher weight can slow the target skill.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_replay_sampler.py -q
python -m pytest tests -q
python benchmarks/self_improvement_regression.py
```

The tests cover determinism, zero weight, empty replay, weight bounds, invalid
types, input immutability, mix summaries, adapter config/result fields and a
tiny-model trainer run. They do not measure retention on the real base
checkpoint; that needs a real replay corpus and a held-out comparison.

## Limitations

- Replay reduces forgetting only if the replay rows represent the behaviour to
  keep. A small or narrow replay set can still forget.
- The weight is a fraction of optimizer steps, not tokens. Long rows therefore
  contribute more tokens per step than short rows.
- The trainer keeps one primary stream and one replay stream. It has no
  per-distribution weights, dedupe across streams or curriculum schedule.
- Rows are shared by reference and are not copied, so callers must not mutate
  them while training is running.
- This remains an external research harness. No product adapter is activated,
  base `0.8.0` weights are unchanged, and promotion requires the existing
  held-out gate plus a separate decision.
