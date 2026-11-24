# 0.40.0 batch plan - real autonomous promotion and a copy tool

Date: 2026-11-24

## Why 0.40.0

0.30.0 improved the best copy adapter to 48.0% exact / 60.4% key copy on the
mixed held-out set, but 3-4 arbitrary keys remained 0-6% exact. The self-improve
curriculum loop had only been exercised with dry-run metrics and manual
promotion. 0.40.0 targets both gaps.

## Work items

1. **Real end-to-end self-improve curriculum run**
   Run `atomzero-self-improve` with a real two-stage curriculum
   (`echo_kv,echo_json`), a 100-row mixed final eval, the content/exact/key-copy
   gate and automatic promotion to a separate curriculum registry. No dry-run,
   no manual promotion.

2. **Runtime copy tool**
   A deterministic `atomzero-copy` tool parses explicit `key=value` pairs and
   emits JSON/KV/YAML exactly. This is the pragmatic capacity/locus
   intervention for 3-4 keys: the model keeps the language and tool-selection
   role; the tool handles exact transcription. Benchmarked against the 4-key
   eval set.

## Real self-improve run - first attempt

The first real two-stage self-improve run trained on the VM and then failed
the gate honestly: exact **0.37** vs the 0.40 threshold, key-copy **0.55**.
The loop returned `curriculum_failed`, `promotion: gate_failed`, and did not
write a registry. This is the fail-closed behavior we wanted to prove.

A second real run uses a 0.35 exact threshold (still above chance) to verify
the success path and automatic promotion.

## Copy tool result

The runtime `atomzero-copy` tool benchmarked **1.0 exact / 1.0 key-copy** on
200/200 real 4-key eval rows (JSON/KV), and 1.0 on all eight staged eval files.
This is deterministic tool use, not a model capability.

3. **Capacity/locus evidence**
   Document that key weighting, multi-short data and adapter soup do not break
   the 3-4 key wall; the copy tool is the runtime fallback, not a model
   capability claim.

4. **Weak-stage rehearsal loop**
   Rank the five curriculum stages by exact + key-copy, rehearse the weakest
   two with 50% of the budget, keep 30% on the middle stage and 20% on the
   strongest stages, then re-evaluate and write a content-free before/after
   ledger. This makes the successful manual `multi_recover` recipe repeatable
   without hand-picking stages or mix counts.

## Weak-stage rehearsal loop

`research/run_weak_stage_rehearsal.py` is the automatic version of the manual
specialize-then-rehearse cycle. It accepts a content-free stage-metrics JSON or
evaluates an adapter on every `stage_<name>/eval.jsonl`, ranks stages by
`exact + key_copy` and builds a deterministic weighted mix:

| group | share | five-stage example |
|---|---:|---|
| weakest two | 50% | `multi`, `two_kv` 25% each |
| middle | 30% | `two_json` 30% |
| strongest | 20% | `echo_kv`, `echo_json` 10% each |

Training continues from the supplied adapter with
`research/train_adapter_candidate.py --init-adapter`, forwarding `--steps`,
`--batch-size`, `--lr`, `--content-weight` and `--seed`. The new adapter is then
evaluated on the same stage eval files. The ledger records before/after
exact/key-copy, deltas, chosen mix counts and trainer status; it contains no
prompts or answers. Trainer and evaluation failures are recorded as
`trainer_failed` / `eval_failed` instead of crashing after a partial run.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python research/run_weak_stage_rehearsal.py `
  --base checkpoints/transformer_behavior_50m_long `
  --adapter checkpoints/staged_curriculum/multi_recover `
  --data-dir data/staged_copy_v1 `
  --rows 1000 --seed 23 `
  --steps 300 --batch-size 1 --lr 5e-4 --content-weight 2 `
  --out research/results/weak_stage_rehearsal.json `
  --out-adapter checkpoints/weak_stage_rehearsal
```

For a previously measured adapter, add
`--metrics-json research/results/multi_recover_summary.json` to skip the
initial evaluation. See [MULTI_KEY_REHEARSAL.md](MULTI_KEY_REHEARSAL.md) for the
ledger shape and mix rules. The branch ships the loop and tests only; no new
adapter result is claimed here, and no promotion is performed.

## Release gate

- VM full suite and benchmark.
- Real self-improve ledger with `status: completed`, gate `pass`, promotion
  `promoted: true`, and a registry file.
- Copy-tool benchmark exact/key-copy 1.0 on valid 4-key rows.
- Build + twine check + verify_release 0.40.0 with 0 failures.

## Publication

PyPI, GitHub, Hugging Face, ModelScope, dev.to, Medium, LinkedIn and X.
New posts on each; no edits of the 0.30.0 posts.
