# atomzero 0.90.0 - mixed replay makes the 63M copy base pass content and safety gates

Status: **release candidate**. Product base weights remain `0.8.0`; the 63M
mixed-replay candidate is a promotable research base.

## Highlights

- **Depth is not the lever.** A 24-layer, 176.8M-parameter upcycle improved
  4-key copy only from 65.5% to 69.5% exact (two seeds) while content retention
  fell to **1.66x** the product base's content loss. Both seeds failed
  retention. The run cost about $0.42 on a rented RTX 3090 and the instance was
  destroyed.
- **Mixed replay fixes the content regression.** The 8-layer 63M model,
  fine-tuned on **6000 copy + 492 content + 40 safety rows**, kept the copy
  gain and improved content loss over the product base.
- **Two seeds pass all four checkpoint gates:**

| seed | 2short | 3short | 4short | retention | safety |
|---|---:|---:|---:|---:|---:|
| 71 | 84.0% / 86.0% | 77.0% / 84.5% | **57.0% / 67.0%** | **0.722** | **40/40** |
| 72 | 86.0% / 86.5% | 76.5% / 80.5% | **58.0% / 63.5%** | **0.809** | **40/40** |

  Copy numbers are exact / key-copy on 200 held-out rows per stage. Retention
  is candidate content loss divided by the 49M baseline; below 1.0 is better.
  The safety probes are held out and disjoint from the training templates; the
  baseline scored 12/40 (30%) on the same set.
- **The trade is explicit.** Safety replay costs about **9 points** of 4-key
  exact match versus the copy-only 8L model (66.5% -> 57.5%), and restores
  refusal behavior from 0/40 to 40/40.
- **Verification:** 1598 tests passed on the VM in 55.84s; benchmark 20/20.

## New tooling

- `research/retention_eval.py`: content-loss retention ratio with per-family
  reporting.
- `research/checkpoint_gate.py`: full-checkpoint gate for content, exact match,
  key-copy and safety, with explicit missing-check handling.
- `research/mix_training_data.py`: deduplicated copy + content mixer with an
  eval-path guard.
- `research/build_content_rows.py`: deterministic QA/summary/citation rows from
  local fact files.
- `research/build_safety_probes.py` and `research/build_safety_replay.py`:
  disjoint safety probe and refusal-replay splits.
- `research/pretokenize.py` plus `--cache-dir` in the trainer: optional
  pretokenized batches for rented GPUs.
- `scripts/vastai/`: preflight, bundle, detached job runner, launcher that
  destroys the instance in `finally`, and a one-command wrapper.

## What stays the same

- The product base remains `0.8.0` (49.2M). The 63M candidate is published as a
  research base, not the default assistant.
- Donation addresses are unchanged.
- The 4-key runtime copy tool remains the exact path for arbitrary keys.
- No adapter is promoted.

## Install

```powershell
pip install atomzero==0.90.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.90.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
