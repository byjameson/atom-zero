# Atom Zero 0.90.0: depth was the wrong answer, data was the right one

Atom Zero 0.90.0 started as a scaling experiment and ended as a data experiment.
The result is a 63M copy-capable base that passes content, copy and safety
gates for the first time.

## The depth experiment

At 0.80.0 an 8-layer 63M checkpoint copied four arbitrary keys at 65.5% exact.
The obvious next question: does more depth help?

A 24-layer, 176.8M-parameter upcycle was fully fine-tuned on the same copy data
on a rented RTX 3090. Two seeds:

| stage | 8L (63M) | 24L (177M) |
|---|---:|---:|
| 2short | 89.5% | 92.3% |
| 3short | 77.0% | 83.5% |
| 4short | 65.5% | **69.5%** |
| retention | 1.089 | **1.664** |

Depth added about **four points** on 4-key copy and made content loss **66%
worse** than the product base. Both seeds failed retention. The run cost $0.42
and the instance was destroyed automatically.

The lesson is simple: more layers scale the copy skill slowly and forget
everything else faster.

## The data experiment

The next run used the same 8-layer base and the same copy data, but mixed in:

- **492 content replay rows** built from local facts (QA, summary, citation),
- **40 safety refusal rows**, repeated to survive a copy-heavy mix.

Two seeds, 200 held-out rows per stage:

| seed | 2short | 3short | 4short | retention | safety |
|---|---:|---:|---:|---:|---:|
| 71 | 84.0% / 86.0% | 77.0% / 84.5% | **57.0% / 67.0%** | **0.722** | **40/40** |
| 72 | 86.0% / 86.5% | 76.5% / 80.5% | **58.0% / 63.5%** | **0.809** | **40/40** |

Copy numbers are exact / key-copy. Retention is candidate content loss divided
by the 49M product base; **below 1.0 is better**. The safety probes are disjoint
from the training templates; the product base scored 12/40 (30%) on the same
set.

Both seeds pass all four checkpoint gates: content not worse, exact match, key
copy, safety.

## The trade is real

Safety replay costs about **nine points** of 4-key exact match compared with the
copy-only 8L model (66.5% -> 57.5%). Mixed replay without safety scored 66.5%
and retention 0.863, but safety was not measured there.

This is not a free win. It is an explicit trade: 57.5% four-key copy, better
content loss than the product base, and refusal behavior that the copy-only
model had completely lost.

## What the gate checks

```text
content:   candidate content loss must not exceed the baseline
exact:     candidate exact match floor 0.25
key-copy:  candidate key-name copy floor 0.25
safety:    held-out refusal probes, explicit missing-check failure
```

Missing checks block promotion. A pass is necessary but not sufficient for
product replacement; packaging, general QA behavior, latency and human review
remain.

## What stays the same

The product base remains **0.8.0** (49.2M). The 63M candidate is published as a
research base because its general assistant behavior has not been validated.
The 4-key runtime copy tool remains the exact path for arbitrary keys. No
adapter is promoted.

## Verification

- VM Ubuntu: **1609 tests passed in 50.45s**.
- Benchmark: 20/20.
- 24L run: two seeds, 2.2 hours, $0.42, instance destroyed.
- Candidate: two seeds, all four gates pass.

## Install

```powershell
pip install atomzero==0.90.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.90.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

Depth was the wrong answer. Data was the right one. The wall moved because the
mix changed, not because the model got taller.
