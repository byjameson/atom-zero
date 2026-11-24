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
