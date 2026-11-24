# Atom Zero 0.21.0: first behavior-passing adapter and the multi-key wall

Atom Zero 0.21.0 is out. This release has a real milestone and a real limit.

## The milestone: a behavior-passing adapter

The copy curriculum now has an adapter that clears the behavior gate on a
250-row mixed held-out set:

```text
exact match:   40.0%   (threshold 25%)
key copy:      54.8%   (threshold 40%)
key recall:    62.8%
content loss:  4.0449 -> 1.0011
full loss:     4.6015 -> 0.9738
gate:          pass / promotable
```

It was promoted to a separate curriculum registry, not the product default.
Base `0.8.0` remains the only active product model.

The chain is now reproducible: staged training -> mixed final eval -> content,
exact-match and key-copy gate -> optional promotion -> content-free ledger.

## The limit: multi-key copying

The best adapter is still weak when the prompt contains 2-4 arbitrary keys:

| stage | rehearsal adapter | pure multi | adapter soup |
|---|---:|---:|---:|
| echo_kv | 58% / 68% | 44% / 52% | 52% / 62% |
| echo_json | 72% / 82% | 10% / 10% | 68% / 74% |
| two_kv | 34% / 56% | 30% / 42% | 26% / 42% |
| two_json | 28% / 46% | 20% / 34% | 20% / 36% |
| multi | 8% / 22% | **12% / 20%** | 6% / 18% |

Values are exact match / key-copy rate.

Two negative results matter:

- **Pure multi training** improves multi exact from 8% to 12%, but destroys
  one-key JSON (72% -> 10%). It is not a viable standalone adapter.
- **Adapter soup** (weighted LoRA A/B averaging) is worse than the rehearsal
  adapter on every stage. Averaging low-rank updates is not task arithmetic.

## What else is new

- Curriculum gate/promotion loop with a safe registry separate from the product.
- Adapter soup CLI with strict geometry validation.
- 1264 tests pass on the 8 vCPU VM in 50.35s; benchmark 20/20.
- CI now installs scikit-learn for the calibration tests.

## Install

```powershell
pip install atomzero==0.21.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.21.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The honest summary: the system can now measure, gate and promote a narrow
capability without human approval. It still cannot reliably copy arbitrary
multi-key structures, and it is not ASI. The next target is the multi-key wall,
not a bigger claim.
