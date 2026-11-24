# atomzero 0.20.0 - gated self-improvement and copy-curriculum evidence

Status: **release candidate**. Base weights remain `0.8.0`; no model checkpoint
was changed by this release.

## Highlights

- **Behavior-gated adapter promotion.** A candidate now needs full-loss,
  content-loss, exact-match and key-copy evidence before `promote` accepts it.
  The replay-weighted adapter that passed the old loss/content gate had 0/48
  held-out exact matches, so the active registry was removed. Base `0.8.0` is
  the only active model.
- **Content and behavior audit.** `compare_adapters.py` reports per-family
  loss, content loss, key loss, value loss, syntax loss, exact match, key-copy
  rate and key recall. `evaluate_candidate` can require exact match and key
  copy.
- **Copy-curriculum research.** Added a randomized-key generator and a staged
  curriculum:
  `echo_kv -> echo_json -> two_kv -> two_json -> multi`.
  Echo training reached **58% exact / 68% key copy** on its held-out stage.
  Two-key KV reached **29% exact / 41% key copy**, but sequential training
  forgot the earlier JSON skill (50% -> 3%). The next batch targets rehearsal
  and adapter merging.
- **Resumable adapter training.** `--init-adapter` continues from a saved
  adapter with strict rank/alpha checks.
- **Gradient accumulation.** `--batch-size` and `--lr-scale` support stable
  multi-row updates without changing the default single-row behavior.
- **Curriculum runner.** `run_staged_curriculum.py` chains stages, evaluates
  each stage plus the final multi stage and writes a content-free ledger.
- **Full fine-tune research control.** Research-only; never a product
  checkpoint and never written to the active registry.
- **VM test environment.** Resized from 2 vCPU / 6 GB to **8 vCPU / 16 GB**.
  Full suite: **1248 passed in 49.02s**; benchmark **20/20**.

## Adapter evidence

```text
stage        exact match   key copy   note
echo_kv          58%          68%     held-out echo stage
echo_json        50%          56%     held-out one-key JSON
two_kv           29%          41%     held-out two-key KV
two_kv -> echo_json retention: 50% -> 3% exact (forgetting)
```

Rehearsal results on 50-row-per-stage held-out slices (exact / key copy):

| stage | mix_all | focus rehearsal | multi rehearsal |
|---|---:|---:|---:|
| echo_kv | 58% / 70% | 58% / 68% | 58% / 68% |
| echo_json | 18% / 18% | **72% / 82%** | 64% / 70% |
| two_kv | 32% / 48% | 34% / 56% | 32% / 48% |
| two_json | 8% / 12% | 28% / 46% | **32% / 56%** |
| multi | 2% / 14% | 8% / 22% | 8% / 18% |

The focus rehearsal recovered the earlier JSON skill and roughly tripled
two-key JSON performance. The multi stage remains hard: 2-4 arbitrary keys with
mixed JSON/KV output still reaches only 8% exact. That is the next research
target, not a product claim.

The held-out city/river adapters from the previous batch stayed at **0/48
exact** despite passing the old full-loss and content gates. That is why the
behavior gate is now mandatory for promotion.

## Install

```powershell
pip install atomzero==0.20.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.20.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>

## Honest limits

- This is not general model self-improvement yet. The adapters learn formats
  and some value copying; arbitrary key copying is still unreliable.
- No product weights changed; base remains `0.8.0`.
- The 0.20.0 release is a tooling and measurement milestone, not an ASI claim.
- The next release should target multi-key copying with a larger model or a
  stronger copy mechanism, a larger held-out copy set and a behavior-passing
  adapter before any product promotion.
