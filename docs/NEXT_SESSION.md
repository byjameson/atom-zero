# Next session - 0.90.0 work in progress

Snapshot: 2026-11-25, after the 0.80.0 release.
Main HEAD: `22e1315` (4 commits ahead of `origin/main`, not pushed).
Working tree: clean. No training or evaluation process is running.

## Released

`0.80.0` is published and verified on PyPI, GitHub, Hugging Face
(`atom-zero-49m` and the new `atom-zero-63m` research repo), ModelScope,
dev.to, Medium, LinkedIn and X. `verify_release --version 0.80.0` reported
0 failures. Product base remains `0.8.0`.

Headline evidence: the 63M depth-upcycled + fully fine-tuned checkpoint
reaches **89.5% / 77.0% / 65.5% exact** on held-out 2/3/4-key copy (200 rows
each), versus 49M at 32/12/0. Data-only control (656 new teacher tasks) left
4short at 0% / 5%. Out-of-distribution key styles score 0/30.

## Merged locally for 0.90.0 (not pushed)

- `feat/content-retention` (`76fe4af`): `research/retention_eval.py`,
  `tests/test_retention_eval.py`, `docs/RETENTION.md`. Compares a candidate
  checkpoint with the product base on content loss; `retention_ratio =
  candidate_content_loss / baseline_content_loss`, default pass `<= 1.10`.
- `feat/checkpoint-gate` (`f0ee02a`): `research/checkpoint_gate.py`,
  `tests/test_checkpoint_gate.py`, `docs/CHECKPOINT_GATE.md`. Full-checkpoint
  gate: content loss, exact match, key-copy, optional safety; missing checks
  block promotion unless `--allow-missing`.
- Both merged with `--no-ff`; their 20 tests passed in 7.45s.

## Experiment status

- 8L upcycled base: `checkpoints/upcycle/atom-zero-49m-8l` (63,389,184 params).
- 8L full fine-tune: `checkpoints/fullft_upcycle8l_v1`. Works: 2short 89.5%,
  3short 77.0%, 4short 65.5% exact (200 rows each). Trained on copy data only;
  content retention not yet measured.
- 8L **LoRA** (`checkpoints/upcycle8l_lora_v1`, 3000 steps, 491,520 params,
  mixed loss `6.2225 -> 1.3372`): 2short only **2% exact / 6% key-copy**.
  Frozen interpolated weights do not recover the copy skill; full fine-tuning
  is the working path. A LoRA-only product path is not supported by evidence.
- 8L LoRA 3short/4short evaluation was stopped when the PC shut down.

## Next steps, in order

1. Measure retention of the 63M full fine-tune against the 49M base:
   ```powershell
   $env:PYTHONPATH="$PWD;$PWD\src"; $env:PYTHONIOENCODING="utf-8"
   python research/retention_eval.py `
     --baseline "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
     --candidate checkpoints/fullft_upcycle8l_v1 `
     --data data/eval_frozen/heldout_city_yaml.jsonl data/eval_frozen/heldout_city_river.jsonl `
     --max-rows 200 --per-family `
     --out research/results/retention_49m_vs_63m.json
   ```
2. Run the new checkpoint gate on the 63M candidate with content + copy evals
   and review the explicit reasons.
3. If retention fails (expected, copy-only full FT), run the 0.90.0 core
   experiment: **full fine-tune the 8L base on a copy + content replay mix**
   (not copy only), then repeat steps 1-2. Locate the original content corpus
   under the desktop workspace `ai gelistir\data` or `data/learning_queue*.jsonl`
   before training; do not use eval files.
4. Optional scale branch: upcycle to 12 layers
   (`python research/upcycle_model.py --base ... --layers 12`) and repeat the
   copy evaluation. 91.7M params, ~367 MB fp32.
5. Finish the 0.90.0 boundary only with: held-out copy numbers, retention
   ratio, checkpoint gate report, full VM suite + benchmark, then the usual
   release surfaces and fresh posts.

## Environment reminders

- Training / distillation / large-model work: host GPU (RTX 3050 6GB).
- Tests, gates, safety verification: VM Ubuntu `192.168.121.128`
  (`byjameson` / `arslan123`, repo `~/atom-zero`). Suspend or shut down VMware
  before closing the PC.
- Do not publish 0.90.0 before its evidence is complete. Release only at the
  10-minor boundary.
- Raw comparison JSONs under `research/results/` are gitignored; commit only
  content-free summaries.
