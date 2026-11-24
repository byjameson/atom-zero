# 0.30.0 batch plan - multi-key copy and the autonomous loop

Date: 2026-11-24

## Why 0.30.0

The release cadence is one 0.1 boundary: 0.20.0 -> **0.30.0** -> 0.40.0.
0.21.0 and 0.22.0 were intermediate patches, not 0.1 boundaries.

The 0.21.0 behavior-passing adapter is the best measured copy adapter:

```text
mixed held-out: 40.0% exact / 54.8% key copy / 62.8% key recall
content loss:   4.0449 -> 1.0011
gate:           pass / promoted to curriculum registry
base:           0.8.0 (product default)
```

The remaining wall is multi-key copying: 8-12% exact on 2-4 arbitrary keys.
This batch targets it directly.

## Experiments

1. **Key/value-aware loss weighting** (`feat/key-weighted-loss`)
   Weight key-name tokens much more than value tokens so the gradient rewards
   copying prompt keys instead of memorizing a key vocabulary.

2. **Multi-key short-value curriculum** (`feat/multi-short-curriculum`)
   A difficulty ladder: 2 keys/short values -> 2 mixed -> 3 short -> 4 short.
   Isolates key copying from tokenization and multiword-value difficulty.

3. **Specialize then rehearse**
   Start from `multi_pure` (12% multi exact, but echo_json falls to 10%) and
   train on the balanced rehearsal mix at low LR. Goal: keep the multi gain and
   recover the earlier stages.

4. **Real self-improve curriculum run**
   Run `atomzero-self-improve` with `--adapter-curriculum-command` on the VM,
   record the gate/promotion block and verify fail-closed behavior.

## First result - specialize then rehearse

`multi_pure` -> balanced rehearsal at low LR reached **48% exact / 60.4%
key copy** on the mixed held-out set (previous best 40% / 54.8%), with content
loss `4.6015 -> 0.7789`. It passed the behavior gate and was promoted to the
curriculum registry. Per-stage: echo_kv 62/74, echo_json 76/80, two_kv 44/54,
two_json 46/72, multi 12/22.

## Release gate

- All tests on the VM (8 vCPU) and benchmark 20/20.
- Behavior gate on the mixed held-out set: exact >= 25%, key copy >= 40%,
  content loss improved, family separation true.
- No model weights changed; base remains 0.8.0.
- Build + twine check + verify_release 0.30.0 with 0 failures.

## Publication surfaces

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
Medium/dev.to/LinkedIn/X get new posts, not edits of the previous ones.
