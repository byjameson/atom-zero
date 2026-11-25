# atomzero 0.60.0 - capability scorecard and controlled attention capacity

Status: **published on 2026-11-25**. Base weights remain `0.8.0`.

## Six-criterion capability scorecard

The project now has a content-free scorecard that refuses to hide failures in
an aggregate loss:

| Criterion | Status | Evidence |
|---|---|---|
| Discovers its own knowledge gaps | partial | answerability ECE 0.064 passes, Brier 0.233 misses; 2 specs FAIL |
| Decides what, when and how to learn | partial | real autonomous curriculum run promoted at 0.53/0.60; still human-gated |
| Retains old knowledge while learning | partial | multi_recover retains 2/5 stages within 0.02; weak-stage rehearsal negative |
| Transfers to different problems | not_met | arithmetic, city-YAML, city/river all 0/30 exact |
| Improves its own learning algorithm | not_met | weak-stage, key weighting and first QKV result negative |
| Low hardware cost | met | 8 vCPU / 16 GB / 40 GB; 49.2M base |

Status counts: **met 1 ? partial 3 ? not_met 2**.

## Capacity evidence (research only)

A Qwen2.5-0.5B-Instruct probe, using a tolerant parser for its `pairs` list /
code-fence outputs, measured:

```text
2short: 65% baseline / 85% flat prompt
3short: 55% / 65%
4short: 55% / 85%
```

The 49M adapter is 0-6% on 3-4 keys. This is strong evidence that capacity is
the copy bottleneck. It is **not** a product result: the model is pretrained,
not from-scratch, and it scored 0/10 exact on arithmetic and city-YAML
zero-shot. Decision: keep the 49M `0.8.0` default; use larger models as
teachers/research backends, not as a replacement without a full gate.

## Controlled QKV vs MLP ablation

| arm | exact mean | key-copy mean | verdict |
|---|---:|---:|---|
| qkv4 (QKV r4/a8) | 3.33% | 8.89% | inconclusive |
| mlp (QKV off) | 3.33% | 5.56% | inconclusive |

Three seeds, 2000 sampled rows per run, 30-row eval subset. Exact match is
identical; the key-copy difference has overlapping 95% intervals. QKV LoRA
does not establish a win and does not break the 3-4 key wall.

## Held-out task evidence

Arithmetic, city-YAML and city/river: base and `multi_recover` both **0/30
exact**. The runtime copy tool is 1.0 on explicit transcription but is excluded
from transfer because it is tool use.

## Verification

- Host: 1463 tests passed.
- VM: 1463 passed in 47.20s.
- Benchmark: 20/20.
- No model weights changed.

## Install

```powershell
pip install atomzero==0.60.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.60.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-0600-a-scorecard-that-refuses-to-hide-failures-15h6>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-0-60-0-a-scorecard-that-refuses-to-hide-failures-3b1f7a748cbf>
