# Adapter growth cycle 1 - candidate export and held-out gate

This is the first execution of the verified-teacher adapter-candidate branch.
It proves the data pipeline and gate wiring. **It is not a trained adapter and
does not claim a real model improvement.**

## Input

Six synthetic verified teacher examples were used as a fixture:

```text
train examples: 5
held-out examples: 1
candidate_id: ad-10fd65622a80d1ab
```

The candidate metadata and local files:

```text
candidate.json   content-free metadata
train.jsonl      text-bearing local training rows
eval.jsonl       text-bearing local held-out rows
```

## Held-out gate

A local evaluator fixture returned:

```text
baseline: loss 1.20, ppl 3.30
adapter:  loss 1.00, ppl 3.00
```

The gate result:

```text
status: pass
promotable: true
loss_delta: -0.20
ppl_ratio: 0.909091
eval_examples: 1
auto_apply: false
requires_training: true
weights_changed: false
reasons: loss_improved, ppl_not_worse
```

A mismatch in held-out count, a loss that does not improve by the margin, or a
perplexity regression fails the gate.

## Autonomous connection

The same pipeline was executed through the autonomous self-improvement action:

```text
atomzero-self-improve --opt-in \
  --adapter-promotable 6 --adapter-candidates 0 \
  --adapter-examples-jsonl verified_teacher.jsonl \
  --adapter-out candidate --adapter-evaluator-command "python evaluator.py" \
  --adapter-ledger adapter_ledger.jsonl --json
```

Result:

```text
status: completed
action: adapter_candidate_ready
candidate: ad-10fd65622a80d1ab
gate: pass / promotable
weights_changed: false
```

The planner signal is `adapter_candidate_ready` (priority 68): promotable
teacher notes exist but no adapter candidate has been exported.

## Honest boundary

- The evaluator fixture is not a trained adapter. The numbers demonstrate the
  gate mechanics, not a real quality gain.
- No adapter was trained, no checkpoint was changed and no weights were
  applied.
- A real run must use `research/adapter_check.py` or `research/compare_adapters.py`
  against an actual external trainer output.
- The held-out gate is only as good as the supplied comparison and the
  candidate's held-out split.
- `promotable` is permission to consider training/promotion, not a deployment
  decision.
