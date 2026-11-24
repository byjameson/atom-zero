# Verified-teacher adapter candidates

This branch is the **capability-growth** step after teacher verification. It
turns verified teacher traces into a local adapter candidate and measures it on
a held-out split. It does not train or apply an adapter.

```text
verified teacher examples
  -> deterministic train/held-out split
  -> candidate.json + train.jsonl + eval.jsonl
  -> external trainer (separate project)
  -> external adapter evaluator
  -> held-out loss/perplexity gate
  -> content-free candidate event
  -> adapter promotable / not promotable
```

## Export

Example row:

```json
{"user": "What is the launch code?",
 "assistant": "The launch code is ORBIT-42.",
 "verification": {"verified": true, "promotable": true,
                  "status": "verified"}}
```

```powershell
atomzero-adapter-candidate export `
  --examples-jsonl data/verified_teacher.jsonl `
  --out-dir data/adapter_candidates --json
```

The export:

- accepts only examples whose `verification.verified` is `true`;
- rejects `promotable: false`;
- deduplicates identical user/assistant pairs;
- sorts by a seeded hash and splits into train/held-out;
- writes text-bearing `train.jsonl` and `eval.jsonl` locally;
- writes `candidate.json` with no prompt/answer text;
- reports `auto_apply: false` and `weights_changed: false`.

## Cross-distribution split

For a distribution-shift check, use an explicit train/eval split instead of a
random tail split:

```powershell
atomzero-adapter-candidate export-cross `
  --train-jsonl data/teacher_json.jsonl `
  --eval-jsonl data/teacher_keyvalue.jsonl `
  --out-dir data/teacher_cross_candidate --json
```

`export-cross`:

- keeps the two distributions separate;
- rejects an identical row that appears in both sides;
- marks the candidate metadata with `split_method:
  explicit_cross_distribution`;
- still requires every row to be verified.

A pass on this candidate means the adapter improved on a held-out distribution,
not just on a random tail of the training distribution.

## Held-out gate

```powershell
atomzero-adapter-candidate evaluate `
  --candidate-json data/adapter_candidates/candidate.json `
  --comparison-json data/adapter_eval.json --json
```

Accepted comparison shapes:

```json
{"examples": 20,
 "baseline": {"loss": 1.20, "ppl": 3.30},
 "adapter":  {"loss": 1.10, "ppl": 3.10}}
```

or the `research/compare_adapters.py` shape with a `models` list containing
`base` and an adapter.

The gate passes only when:

- the comparison's `examples` equals the candidate's held-out count;
- adapter loss is lower by at least `--min-loss-delta` (default `0.01`);
- adapter perplexity is not worse than `--max-ppl-ratio` (default `1.0`).

A pass sets `promotable: true`, but **does not train or apply anything**.
`auto_apply` and `weights_changed` are always false.

## Autonomous loop

```powershell
atomzero-adapter-candidate loop `
  --examples-jsonl data/verified_teacher.jsonl `
  --out-dir data/adapter_candidates `
  --trainer-command "python research/train_adapter_candidate.py --base {base} --train {train_jsonl} --eval {eval_jsonl} --out {adapter_dir} --comparison {comparison_path}" `
  --base "C:\Users\you\.cache\atomzero\atom-zero-49m" `
  --ledger data/adapter_candidates.jsonl --json
```

The trainer runs through the bounded `atomzero.sandbox` command runner with a
timeout and `PYTHONPATH` rooted at the repository `src`. It must write the
comparison JSON to `{comparison_path}`; the loop then applies the held-out gate.

If no trainer is supplied, an evaluator command can be used:

```powershell
atomzero-adapter-candidate loop `
  --examples-jsonl data/verified_teacher.jsonl `
  --out-dir data/adapter_candidates `
  --evaluator-command "python research/compare_adapters.py --data {eval_jsonl} --json" `
  --ledger data/adapter_candidates.jsonl --json
```

The evaluator runs through the bounded `atomzero.sandbox` command runner with a
timeout. It may be a real adapter comparison job or a local script that emits
the comparison JSON. No human approval is needed for export/evaluation; the
gate is the decision.

The loop records a content-free event:

```json
{"candidate_id": "ad-...", "train_examples": 80, "eval_examples": 20,
 "status": "pass", "promotable": true, "loss_delta": -0.1,
 "ppl_ratio": 0.94, "auto_apply": false, "weights_changed": false}
```

## Connection to the autonomous loop

`atomzero-improve` emits `adapter_candidate_ready` when promotable teacher notes
exist but no candidate has been exported. `atomzero-self-improve` executes that
action autonomously when adapter inputs are supplied:

```powershell
atomzero-self-improve --opt-in `
  --adapter-promotable 20 --adapter-candidates 0 `
  --adapter-examples-jsonl data/verified_teacher.jsonl `
  --adapter-out data/adapter_candidates `
  --adapter-comparison-json data/adapter_eval.json `
  --adapter-ledger data/adapter_candidates.jsonl --json
```

The adapter branch is part of the same **no-per-change-human-approval** policy:
the held-out gate, content-free ledger and explicit `promotable` flag replace
the approval. Training and weight promotion remain a separate offline step.

## Honest limits

- A candidate is not a trained adapter; it is a reviewed data artifact.
- The gate measures only the supplied comparison. Bad evaluation data or a
  mismatched held-out set fails the gate; it cannot prove general quality.
- The evaluator command runs in the worktree sandbox, which is not an OS-level
  VM on this machine.
- `promotable` means "this candidate passed the declared held-out gate", not
  "this adapter is safe or better in every domain".
- No model weights change in this branch.
