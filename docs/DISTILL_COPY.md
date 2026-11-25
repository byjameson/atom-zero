# Teacher distillation for copy behavior

**Status:** research pipeline. It produces a candidate adapter and a content-free
ledger. It does not change the product weights or promote an adapter.

The 49M student is weak at copying three or four arbitrary key/value pairs. A
larger local teacher (Qwen2.5-0.5B-Instruct by default) can copy them much more
often, but its raw output is not automatically training data. The distillation
cycle turns staged copy prompts into **verified** copy rows, trains the 49M
student on them, and evaluates the student on the original disjoint held-out
sets.

```text
staged train prompts (2short / 3short / 4short)
  -> Qwen2.5-0.5B teacher (host GPU)
  -> tolerant parse + exact key/value verification
  -> canonical flat JSON / KV / YAML
  -> 49M student training (host GPU)
  -> held-out 2/3/4-key evaluation
  -> behavior gate in the VMware VM (CPU)
```

Eval prompts and held-out keys are **never** sent to the teacher. The teacher
reads only `train.jsonl` source rows; the student is evaluated on the original
disjoint `eval.jsonl` files.

## Two scripts

| script | role |
|---|---|
| `research/build_distill_copy.py` | asks the teacher, verifies exact pairs, writes `train_distill.jsonl` and a content-free `manifest.json` |
| `research/run_distill_cycle.py` | runs teacher build -> student trainer -> held-out evaluator and writes one content-free cycle ledger |

The cycle runner is a harness, not a trainer. It uses only the standard
library, never imports `torch` and never loads a model. The real work happens
in subprocesses, and tests inject fake commands.

## Data contract

The source data is the short-value multi-key ladder:

- `data/multi_short_v1/stage_2short/train.jsonl`
- `data/multi_short_v1/stage_3short/train.jsonl`
- `data/multi_short_v1/stage_4short/train.jsonl`

The teacher output is:

- `data/distill_copy_v1/train_distill.jsonl` - rows the student trains on
- `data/distill_copy_v1/manifest.json` - aggregate counts only

The held-out files evaluated by the student are the original
`data/multi_short_v1/stage_*/eval.jsonl` files. They are disjoint from every
train source by key and value.

The teacher builder refuses a source whose name or path looks like an eval
split, and refuses a row whose `partition` is not `train`. This is a guard
against accidentally leaking a held-out set into distillation.

### Verification rule

For each source row the builder:

1. sends the row's own `user` prompt to the teacher with greedy decoding;
2. parses the answer tolerantly (flat JSON, `pairs` lists, source/target
   objects, code fences, KV or YAML);
3. requires the parsed pairs to match the source pairs **exactly**, ignoring
   order;
4. renders the exact requested format (JSON, KV or YAML) as the canonical
   answer.

An answer that renames a key, drops a pair, adds a pair or changes a value is
rejected. A malformed answer is rejected; it is not salvaged from a prefix.

`--fallback-oracle` writes the verified source assistant instead of dropping a
row whose teacher answer failed. These rows are marked
`teacher_verified: false`, `status: oracle_fallback` in the generated JSONL and
counted separately in the manifest. Fallback rows preserve coverage; they are
not teacher-quality evidence.

This rule checks **copy exactness**, not semantic equivalence or reasoning. A
row that copies the pairs in a different but accepted structural shape is
canonicalised; a row with a different meaning but the same literal pairs would
still pass. The manifest counts what happened; it does not prove that the
teacher is generally correct.

### Manifest counts

`build_distill_copy.py` writes a content-free manifest with fields such as:

```json
{
  "dataset": "distill_copy_v1",
  "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
  "verification_method": "teacher_exact_pair_match_v1",
  "requested": 2000,
  "generated": 1900,
  "teacher_valid": 1800,
  "fallback_oracle": 100,
  "rejected": 100,
  "by_format": {"json": 1200, "kv": 600, "yaml": 100},
  "by_stage": {"2short": 700, "3short": 650, "4short": 550},
  "note": "Aggregate counts only: ..."
}
```

The cycle ledger copies only the numeric counts from this file. It never
copies `source_files`, `note`, `teacher_model` or any other text field.

## Exact host-GPU cycle

Run from the repository root on the Windows host with the GPU. The first
command concatenates the three train sources without a byte-order mark. (The
builder reads UTF-8 without BOM; PowerShell `Set-Content -Encoding utf8` is not
used here for that reason.)

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
New-Item -ItemType Directory -Force data\distill_copy_v1 | Out-Null
cmd /c copy /y /b data\multi_short_v1\stage_2short\train.jsonl+data\multi_short_v1\stage_3short\train.jsonl+data\multi_short_v1\stage_4short\train.jsonl data\distill_copy_v1\source_train.jsonl

python research/run_distill_cycle.py `
  --backend gpu `
  --train data/distill_copy_v1/source_train.jsonl `
  --teacher-command "python research/build_distill_copy.py --source {source} --out-dir {out_dir} --rows {steps} --seed {seed} --shuffle --fallback-oracle" `
  --teacher-data data/distill_copy_v1/train_distill.jsonl `
  --base checkpoints/transformer_behavior_50m_long `
  --eval data/multi_short_v1/stage_2short/eval.jsonl `
  --eval data/multi_short_v1/stage_3short/eval.jsonl `
  --eval data/multi_short_v1/stage_4short/eval.jsonl `
  --out checkpoints/distill_cycle `
  --steps 2000 --batch-size 4 --lr 3e-4 `
  --content-weight 2 --key-weight 2 --value-weight 2 --seed 23 `
  --max-eval-rows 600 `
  --result research/results/distill_cycle_run.json
```

The runner expands the teacher template with `{source}`, `{out_dir}`,
`{result}` and the numeric settings. This command uses `{source}`, `{out_dir}`,
`{steps}` and `{seed}`; `{result}` remains available for builders that accept
an explicit manifest path. The canonical manifest location is
`{out_dir}/manifest.json`, which is what `build_distill_copy.py` writes and
what the runner reads.

`{eval}` is deliberately **not** available to the teacher template. A teacher
command that references it is rejected as an invalid template, so held-out
prompts cannot be routed to the teacher by accident.

When a teacher command is supplied, the runner removes the expected teacher
JSONL and manifest before the rebuild, so a command that silently writes
nothing cannot reuse data from an earlier cycle. The teacher output directory
is therefore treated as a generated artifact directory, not a shared store.

The default trainer template is equivalent to:

```text
python research/train_adapter_candidate.py
  --base {base} --train {train} --eval {eval} --out {out}
  --steps {steps} --batch-size {batch_size} --lr {lr}
  --content-weight {content_weight} --key-weight {key_weight}
  --value-weight {value_weight} --seed {seed}
  --result {result} --comparison {comparison}
```

`{train}` is the verified teacher JSONL, not the raw source. `--train` is the
source split the teacher reads; `--teacher-data` is the verified output the
student trains on. The two roles are deliberately separate.

The default evaluator is `research/compare_adapters.py` with `--per-family`,
`--exact-match` and `--key-copy`. It compares the base checkpoint and the new
adapter on every `--eval` file. When more than one `--eval` is supplied, the
runner merges them into a temporary JSONL for the evaluator and trainer; the
temporary file is removed afterwards.

`--max-eval-rows` caps the **final evaluator** input. The trainer's own
`--eval` still uses the full held-out set. Use a smaller number for a quick
smoke run; leave it out for the real cycle.

### Dry run

The pipeline can be validated without a model, data file or GPU:

```powershell
python research/run_distill_cycle.py `
  --dry-run `
  --teacher-command "python research/build_distill_copy.py --source {source} --out-dir {out_dir}" `
  --train data/distill_copy_v1/source_train.jsonl `
  --teacher-data data/distill_copy_v1/train_distill.jsonl `
  --base checkpoints/transformer_behavior_50m_long `
  --eval data/multi_short_v1/stage_3short/eval.jsonl `
  --out checkpoints/distill_cycle_dry `
  --result research/results/distill_cycle_dry.json
```

A dry run runs no subprocess, reads no teacher/eval file and writes only the
ledger. Its metrics are marked `"synthetic": true`.

## Ledger

The ledger is written to `--result` (default
`research/results/distill_cycle_run.json`). It is content-free: no prompts,
answers, key names, values or scored rows. Raw comparison reports contain
scored rows, so the runner writes them only inside a temporary directory and
deletes them before returning. The teacher JSONL is never parsed.

Representative shape:

```json
{
  "schema": "distill_cycle_ledger_v1",
  "created": "2026-11-25T12:00:00",
  "content_free": true,
  "status": "completed",
  "dry_run": false,
  "backend": "gpu",
  "paths": {
    "base": "checkpoints/transformer_behavior_50m_long",
    "source_train": "data/distill_copy_v1/source_train.jsonl",
    "teacher_data": "data/distill_copy_v1/train_distill.jsonl",
    "teacher_manifest": "data/distill_copy_v1/manifest.json",
    "eval": "data/multi_short_v1/stage_2short/eval.jsonl",
    "evals": [
      "data/multi_short_v1/stage_2short/eval.jsonl",
      "data/multi_short_v1/stage_3short/eval.jsonl",
      "data/multi_short_v1/stage_4short/eval.jsonl"
    ],
    "out": "checkpoints/distill_cycle",
    "result": "research/results/distill_cycle_run.json"
  },
  "settings": {
    "steps": 2000, "batch_size": 4, "lr": 0.0003,
    "content_weight": 2.0, "key_weight": 2.0, "value_weight": 2.0,
    "seed": 23, "backend": "gpu", "max_eval_rows": 600, "dry_run": false
  },
  "teacher": {
    "requested": true,
    "status": "completed",
    "returncode": 0,
    "elapsed_seconds": 412.5,
    "command": ["python", "research/build_distill_copy.py", "..."],
    "source": "data/distill_copy_v1/source_train.jsonl",
    "data": "data/distill_copy_v1/train_distill.jsonl",
    "data_bytes": 1234567,
    "manifest": {
      "path": "data/distill_copy_v1/manifest.json",
      "found": true,
      "counts": {
        "requested": 2000,
        "generated": 1900,
        "teacher_valid": 1800,
        "fallback_oracle": 100,
        "rejected": 100,
        "by_format.json": 1200,
        "by_format.kv": 600,
        "by_format.yaml": 100,
        "by_stage.2short": 700,
        "by_stage.3short": 650,
        "by_stage.4short": 550
      },
      "error": null
    }
  },
  "trainer": {
    "status": "completed",
    "returncode": 0,
    "elapsed_seconds": 638.1,
    "command": ["python", "research/train_adapter_candidate.py", "..."],
    "adapter": "checkpoints/distill_cycle",
    "adapter_exists": true,
    "result_path": "checkpoints/distill_cycle/trainer_result.json",
    "result_found": true,
    "result": {
      "train_loss": 0.42, "train_steps": 2000, "optimizer_steps": 500,
      "rows_seen": 2000, "device": "cuda", "weights_changed": true
    },
    "adapter_config": {"r_mlp": 8, "alpha_mlp": 16, "method": "..."}
  },
  "evaluation": {
    "status": "completed",
    "returncode": 0,
    "elapsed_seconds": 84.0,
    "command": ["python", "research/compare_adapters.py", "..."],
    "source": "data/multi_short_v1/stage_2short/eval.jsonl",
    "sources": [
      "data/multi_short_v1/stage_2short/eval.jsonl",
      "data/multi_short_v1/stage_3short/eval.jsonl",
      "data/multi_short_v1/stage_4short/eval.jsonl"
    ],
    "data": "<temporary combined eval>",
    "max_eval_rows": 600,
    "rows_available": 600,
    "rows_used": 600,
    "examples": 600,
    "base": {
      "loss": 4.10, "content_loss": 4.04,
      "exact_match_rate": 0.0, "key_copy_rate": 0.0
    },
    "adapter": {
      "loss": 1.10, "content_loss": 1.20,
      "key_loss": 1.30, "value_loss": 1.40,
      "exact_match_rate": 0.42, "key_copy_rate": 0.58,
      "key_recall": 0.64
    },
    "delta": {
      "loss": -3.0, "content_loss": -2.84,
      "exact_match_rate": 0.42, "key_copy_rate": 0.58
    },
    "per_family": {
      "distill_copy_2short": {
        "base": {"loss": 4.0, "exact_match_rate": 0.0},
        "adapter": {"loss": 0.8, "exact_match_rate": 0.55},
        "exact_match": {"examples": 200, "matches": 110, "rate": 0.55},
        "key_copy": {"examples": 200, "matches": 130, "rate": 0.65}
      }
    },
    "exact_match_rate": 0.42,
    "key_copy_rate": 0.58,
    "loss": 1.10,
    "content_loss": 1.20,
    "synthetic": false
  },
  "error": null,
  "elapsed_seconds": 1135.0
}
```

Only whitelisted numeric metrics and known scalar trainer fields are copied.
`evaluation.per_family` is built from aggregate family metrics, never from
`metrics.rows`. Family labels are task names, not prompt text.

### Status values

| status | meaning |
|---|---|
| `completed` | teacher (if requested), trainer and evaluator all succeeded |
| `teacher_failed` | source missing, teacher command non-zero, or teacher output missing/empty; trainer and evaluator did not run |
| `trainer_failed` | trainer non-zero, could not start, or wrote no result/adapter; evaluator did not run |
| `eval_failed` | eval input missing/empty, evaluator non-zero, or the comparison is invalid/incomplete |

A failed stage is recorded in the ledger and the CLI exits with code 1. The
CLI exits with code 2 for invalid arguments or an unwritable ledger. No failure
path stores child stdout/stderr; rerun the failing command directly when you
need its logs.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_distill_cycle.py -q
```

The tests use fake teacher/trainer/evaluator commands. They never load a model
and never touch a real checkpoint.

## VM behavior gate handoff

The host cycle is **not** the final gate. It produces:
`checkpoints/distill_cycle` (adapter),
`research/results/distill_cycle_run.json` (content-free ledger), and the
held-out metrics above. The final behavior gate runs in the Ubuntu VMware VM,
which has no GPU. This is intentional: the VM is the disposable, separate
validation environment.

Sync the repository and copy the adapter and base checkpoint into the VM:

```powershell
powershell -File scripts/vm_test.ps1 -Sync
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -r checkpoints\distill_cycle byjameson@192.168.121.128:~/atom-zero/checkpoints/
scp -i "$env:USERPROFILE\.ssh\id_ed25519" -r checkpoints\transformer_behavior_50m_long byjameson@192.168.121.128:~/atom-zero/checkpoints/
```

If the eval files are not already in the VM, copy the three
`data/multi_short_v1/stage_*/eval.jsonl` files to the matching paths.

In the VM, re-run the held-out comparison for each key count:

```bash
cd ~/atom-zero
for stage in 2short 3short 4short; do
  PYTHONPATH=src .venv/bin/python research/compare_adapters.py \
    --base checkpoints/transformer_behavior_50m_long \
    --data data/multi_short_v1/stage_${stage}/eval.jsonl \
    --adapters checkpoints/distill_cycle \
    --per-family --exact-match --key-copy \
    --out research/results/distill_vm_gate_${stage}.json
done
```

Then run the full VM suite and benchmark:

```bash
cd ~/atom-zero
PYTHONPATH=src .venv/bin/python -m pytest tests -q
PYTHONPATH=src .venv/bin/python benchmarks/self_improvement_regression.py
```

Or run both from the host with `powershell -File scripts/vm_test.ps1` (use
`-Sync` first if the code changed).

The human decision compares the VM results with the 0.70 plan:

- at least one held-out 3-4 key stage improves against the base;
- no retained stage loses more than 5 points of exact match;
- exact match and key copy meet the current promotion thresholds in
  [`docs/ADAPTER_PROMOTION.md`](ADAPTER_PROMOTION.md) when the gate is run;
- the full VM suite and benchmark pass.

The cycle runner never promotes and never writes an active-adapter registry. A
VM pass is evidence for a separate, explicit promotion decision, not an
automatic one.

## Honest limits

- Teacher verification is literal pair matching, not reasoning or semantic
  correctness.
- Fallback-oracle rows are verified copy rows but are not teacher behavior.
  Read `fallback_oracle` before claiming teacher quality.
- The 49M student can still fail to generalise to unseen keys. Held-out
  exact-match and key-copy rates, not training loss, decide the cycle.
- A host-side comparison is a smoke test. The VM comparison and full suite are
  the final gate.
- The ledger proves what happened in the harness; it does not prove that the
  adapter is safe to ship or that the teacher model is appropriate for other
  tasks.
