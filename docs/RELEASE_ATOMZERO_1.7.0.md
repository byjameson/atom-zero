# atomzero 1.7.0 - a 7B teacher lifts conversation, the joint profile still decides

Status: **published on 2026-11-27.** No default change: the 63M
balanced base (weights `0.9.0`) remains the default; the 49M `0.8.0` base
remains the fallback. The A anchor arm is parked after seed 72 failed self-harm
14/16.

## The result

1.7.0 tests whether a larger teacher (Qwen2.5-7B-Instruct) can lift the 63M
conversation profile without giving back copy, retention or safety. The 7B
teacher produced 260 accepted answers; the internal fallback mix is 8980 rows
(518 7B, 2 base fallback), 854,500 pretokenized tokens.

| arm | single | multi final/both | 4-key copy | retention | safety | product |
|---|---:|---:|---:|---:|---:|---|
| J 1.6.0 (default) | 95.8% | 62.5 / 39.6 | 48.5% | 0.703 | 40/40 | ok |
| T 7B s71 (no anchor) | 95.8% | **79.2 / 47.9** | **38.5%** | **0.868** | 40/40 | ok |
| TA 7B s71 (J anchor) | 97.5% | 77.1 / 45.8 | **38.5%** | 0.844 | 40/40 | ok |

T is the conversation win: **79.2% final-turn / 47.9% both-turn** multi-turn
and retention **0.868**. It also loses **10 pp 4-key copy** (48.5% -> 38.5%),
beyond the 3-5 pp budget, so T alone is not a promotion. TA adds the J anchor
(beta 0.3) to recover copy while keeping the multi-turn and retention gain, but
the anchor leaves copy at **38.5%** and multi-turn at **77.1 / 45.8%**. The copy
gap is unchanged, so TA is rejected too.

## What changed

- 7B teacher pipeline (internal, not part of the public package): fixed prompt
  set, 260 accepted answers, content-free manifest, one substituted explanation
  prompt recorded.
- T and TA training arms over the 8980-row teacher mix; full training and
  product evidence is recorded in the internal batch plan.
- Scoped ModelScope Studio demo: **Atom Zero 63M — document QA demo**, with an
  explicit scope message for out-of-corpus questions and a cited answer path
  for the synthetic launch/warranty/travel/visitor corpus.
- Reddit is no longer a release surface (its filters remove posts); the README,
  release checklist and release body were updated.
- PyPI metadata now carries the live demo URL (the 1.6.0 description was
  immutable). 1.7.0 is a tooling and documentation boundary.

## Why the default stays J

At 63M the binding constraint is the joint profile. Conversation and retention
can move together, but the copy budget (3-5 pp around J's 48.5%) and the safety
gates still decide. Two independent arms (T, TA) land at 38.5% copy, so the 7B
result is a negative/limited result, not a promotion. Seeds 72/73 are not
trained: the same recipe cannot close a 10 pp gap.

## Verification

- VM Ubuntu: **1764 tests passed in 57.97s**; benchmark **20/20** (before the
  1.7.0 version bump; the final run is repeated after the bump).
- TA training: train loss 0.8335, task loss 0.8014, anchor loss 0.1071, staged
  exact 0.236.
- TA product suite: benign single 97.5%, multi 77.1 / 45.8, copy 38.5%, content
  ratio 0.844, safety 40/40, RAG 100%, product `ok`.
- A s73 main gate was promotable, but the arm is parked by s72 self-harm 14/16.
- No Vast.ai instance is running; the 7B teacher run was destroyed after
  download.

## Install

```powershell
pip install atomzero==1.7.0
atomzero-quickstart
```

The quickstart still downloads `byjameson/atom-zero-63m-balanced` (~279 MB) into
`~/.cache/atomzero/atom-zero-63m-balanced`.

Fallback:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-49m `
  --cache ~/.cache/atomzero/atom-zero-49m
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.7.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-63m-balanced>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-63m-balanced>
- Live demo: <https://modelscope.cn/studios/byjameson/atomzerodemo>
- Sponsor: <https://github.com/sponsors/byjameson>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
