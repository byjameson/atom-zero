# Multi-key rehearsal and adapter-soup cycle

Date: 2026-11-24

## Goal

The staged curriculum reached 58% exact on echo and 72% exact on one-key JSON
after rehearsal, but multi-key copying stayed at 8% exact / 18% key copy. This
cycle tried three ways to improve it:

1. a pure multi-key fine-tune from the best rehearsal adapter;
2. a multi-focused rehearsal mix;
3. adapter soup (weighted LoRA A/B averaging) to merge complementary adapters.

All measurements use 50 held-out rows per stage and the 250-row mixed held-out
set. Keys and entities are disjoint from training.

## Results

### Mixed held-out set (250 rows)

| adapter | exact | key copy | key recall | loss |
|---|---:|---:|---:|---:|
| rehearsal_focus | **40.0%** | **54.8%** | 62.8% | 0.974 |
| adapter soup (focus 0.5 + multi 0.3 + echo_json 0.2) | 34.4% | 46.4% | 54.7% | 1.028 |

### Per-stage (50 rows, exact / key copy)

| stage | rehearsal_focus | pure multi | adapter soup |
|---|---:|---:|---:|
| echo_kv | 58% / 68% | 44% / 52% | 52% / 62% |
| echo_json | 72% / 82% | 10% / 10% | 68% / 74% |
| two_kv | 34% / 56% | 30% / 42% | 26% / 42% |
| two_json | 28% / 46% | 20% / 34% | 20% / 36% |
| multi | 8% / 22% | **12% / 20%** | 6% / 18% |

## Interpretation

- **Pure multi training** improves multi exact match from 8% to 12%, but it
  catastrophically forgets echo_json (72% -> 10%). It is not a viable
  standalone adapter.
- **Adapter soup** does not help: weighted A/B averaging is worse than
  rehearsal_focus on every stage. A/B averaging is not task arithmetic; the
  low-rank updates interfere.
- **Rehearsal_focus remains the best overall adapter.** It passes the behavior
  gate on the mixed held-out set:

```text
baseline loss:  4.6015 -> 0.9738
content loss:   4.0449 -> 1.0011
exact match:    0.0% -> 40.0%   (threshold 25%)
key copy:       0.0% -> 54.8%   (threshold 40%)
family separation: true
gate: pass / promotable
```

It was promoted to the separate curriculum registry
`data/curriculum_active_adapter.json`, not the product registry. Base `0.8.0`
remains the default.

## Short-value multi-key ladder

The mixed `multi` stage confounds three difficulties: how many keys must be
copied (2-4), how long the values are (single token vs multiword), and which
syntax is requested (JSON or KV). An 8-12% exact-match score does not show
which one is blocking the model. `research/build_multi_short_queue.py` isolates
key count from value length with four short-value stages:

| # | stage dir | pairs | values |
|---|---|---|---|
| 1 | `stage_2short` | 2 | single-token values only |
| 2 | `stage_2mixed` | 2 | one single token + one multiword value |
| 3 | `stage_3short` | 3 | single-token values only |
| 4 | `stage_4short` | 4 | single-token values only |

Every stage alternates compact JSON and one-line KV. The prompt states the
exact instruction and lists the pairs, for example:

```text
Copy these pairs exactly into compact JSON: [niaq=spuckhap; dlkh2=clouded window].
Return only the JSON object and keep exactly these keys and values. ...
```

Keys are 3-5 characters in one of three styles (`short`: lowercase
alphanumeric, `cv`: five-letter consonant-vowel alternation, `word`: 3-5
lowercase letters). Keys and values are drawn from global registries, so
`train.jsonl` and `eval.jsonl` share no key names and no values, per stage and
across stages. Every canonical answer is parsed back and compared key by key
before the row is kept; a failing row is regenerated.

The ladder answers three separate questions:

- `2short` vs `2mixed` isolates the cost of one multiword value;
- `2short` vs `3short` vs `4short` isolates the cost of copying one more pair;
- `2short` vs the earlier `multi` stage isolates short values from the full
  mixed distribution.

Generate the queue (default `data/multi_short_v1`, gitignored):

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python research/build_multi_short_queue.py `
  --out-dir data/multi_short_v1 `
  --rows 2000 --eval-rows 200 `
  --difficulty all --key-style short --seed 23 --shuffle
```

The default build produces 2,000 train and 200 eval rows per stage (8,800 rows
total), 24,200 unique keys and 24,200 unique values, with zero train/eval key
or value overlap. Each stage directory has its own `train.jsonl`,
`eval.jsonl` and `manifest.json`, and the root manifest records the ladder
order and aggregate overlap. Use `--difficulty 2short` (or a comma list) for a
single rung. The generator uses only the standard library and does not load a
model or change the trainer.

```powershell
python -m pytest tests/test_multi_short_queue.py -q
```

This ladder is a data-isolation tool, not a training result. It says nothing
about model quality until each stage is trained and measured on its own
held-out split.

## Conclusion

Multi-key copying is still the bottleneck. The next experiments should target
it directly:

- a larger model or full attention fine-tune, since LoRA out_proj + MLP is
  likely capacity-limited for 2-4 arbitrary keys;
- a copy-focused objective that explicitly weights key tokens more than value
  tokens;
- a pointer/copy mechanism or RAG-style exact transcription at the runtime
  boundary, rather than expecting a 49M model to learn arbitrary copying from
  a small adapter.

Do not repeat A/B soup or pure multi training and expect retention.

## Evidence

- `research/results/mixed_eval_rehearsal_focus.json` (local, contains prompts)
- `research/results/staged_mp50_*.json` (local)
- `research/results/staged_soup50_*.json` (local)
- `research/results/curriculum_gate.json` (content-free gate output)
- `data/curriculum_active_adapter.json` (VM local)
- `checkpoints/staged_curriculum/rehearsal_focus` (VM local)
- `checkpoints/adapter_soup/focus_multi_echo` (VM local)
