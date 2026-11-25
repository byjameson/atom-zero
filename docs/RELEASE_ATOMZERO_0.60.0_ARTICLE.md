# Atom Zero 0.60.0: a scorecard that refuses to hide failures

Atom Zero 0.60.0 is the 0.6 boundary. The headline is not a bigger benchmark
number; it is a scorecard that makes the failures explicit.

## Six criteria, six verdicts

| Criterion | Status | Evidence |
|---|---|---|
| Discovers its own knowledge gaps | partial | ECE 0.064 passes, Brier 0.233 misses; 2 specs FAIL |
| Decides what, when and how to learn | partial | real autonomous curriculum promoted at 0.53/0.60; still human-gated |
| Retains old knowledge while learning | partial | best rehearsal retains 2/5 stages; weak-stage rehearsal negative |
| Transfers to different problems | **not_met** | arithmetic, city-YAML, city/river all 0/30 exact |
| Improves its own learning algorithm | **not_met** | weak-stage, key weighting and first QKV result negative |
| Low hardware cost | **met** | 8 vCPU / 16 GB / 40 GB; 49.2M base |

Status counts: **met 1 ? partial 3 ? not_met 2**.

## Capacity evidence

A Qwen2.5-0.5B-Instruct probe, with tolerant parsing for its pairs-list and
code-fence outputs:

```text
2short: 65% baseline / 85% flat prompt
3short: 55% / 65%
4short: 55% / 85%
```

The 49M adapter is 0-6% on 3-4 keys. This confirms capacity is the copy
bottleneck. It is not a product result: the model is pretrained, not
from-scratch, and it scored 0/10 exact on arithmetic and city-YAML zero-shot.
The 49M `0.8.0` remains the low-cost default; larger models are teachers and
research backends.

## Controlled QKV ablation

| arm | exact mean | key-copy mean | verdict |
|---|---:|---:|---|
| qkv4 (QKV r4/a8) | 3.33% | 8.89% | inconclusive |
| mlp (QKV off) | 3.33% | 5.56% | inconclusive |

Three seeds, 2000 sampled rows per run, 30-row eval subset. Exact match is
identical; the key-copy difference has overlapping 95% intervals. QKV LoRA
does not establish a win and does not break the 3-4 key wall.

## Held-out transfer

Arithmetic, city-YAML and city/river: base and `multi_recover` both **0/30
exact**. The runtime copy tool is 1.0 on explicit transcription, but it is tool
use, not model transfer.

## Verification

- Host: 1463 tests passed.
- VM: 1463 passed in 46.72s.
- Benchmark: 20/20.
- No model weights changed; base remains `0.8.0`.

## Install

```powershell
pip install atomzero==0.60.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.60.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The project now measures what it cannot do. That is the precondition for
improving it.
