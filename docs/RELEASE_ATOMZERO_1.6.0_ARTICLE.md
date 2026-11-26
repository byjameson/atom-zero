# Atom Zero 1.6.0: teacher data raised conversation, but the profile still belongs to the 63M default

The 1.6.0 question was simple: can teacher-generated benign conversation beat the
model that already passed every gate?

The answer is **not yet**. Teacher data did exactly what it was supposed to do to
conversation. It also moved the safety and copy numbers in the wrong direction,
and the combined profile still favors the current default.

## The numbers

| variant | benign | safety | self-harm | 4-key copy | retention | multi-turn (final / both) |
|---|---:|---:|---:|---:|---:|---:|
| J (default) | 93.4% | 40/40 | 16/16 | 48.5% | 0.703 | 75% / 50% |
| K | **98.5%** | 39/40 | 14/16 | 48.5% | 0.808 | **93.8% / 75%** |
| K2 | 89.0% | **40/40** | **16/16** | **53.5%** | 0.816 | 62.5% / 56.3% |
| K3 | 95.6% | **40/40** | **16/16** | 45.0% | 0.828 | 68.8% / 37.5% |

K is the conversation result: 98.5% benign useful and 93.8% multi-turn. It fails
the safety gate (39/40) and the self-harm gate (14/16), so it cannot be the
default.

K2 restores both gates and even lifts four-key copy to 53.5%. The cost is
conversation: benign drops to 89.0% and multi-turn to 62.5%.

K3 is the midpoint. It passes safety, self-harm, the checkpoint gate and the
product gate, but copy falls to 45.0%, retention to 0.828, and multi-turn to
68.8% / 37.5%. Every one of those is below J.

## Why the trade exists

At 63M parameters, these behaviors share the same small weights:

- More teacher-style conversation makes ordinary prompts more useful.
- Fewer safety and self-harm repeats let the refusal behavior drift.
- More safety repeats restore the gates but crowd out conversation.
- The copy grammar and the content-retention signal are additional constraints,
  not free variables.

The 6x midpoint is the clearest demonstration. It is safe enough and helpful
enough to pass the checkpoint and product gates, but it is not better than J on
the full profile. "Passes" and "beats the default" are different bars.

## What 1.6.0 ships

- A teacher benign chat generator: 400 requested, 260 accepted, 437 rejected
  attempts, zero duplicates. Validation covers length, word count, the local
  SafetyFilter, personal data and duplicate pairs.
- An expanded product suite: 136 prompts, local RAG cases, and separate
  multi-turn final-turn and both-turn rates.
- 35 tests for the new generator and suite.
- No new weights. The default remains the 63M balanced base (`0.9.0`); the 49M
  `0.8.0` base remains the fallback.

The point of the release is the evidence, not a promotion. A negative result
that rules out a mix is still progress: it says the next gain has to come from a
better data objective or a different capacity point, not from adding more of the
same teacher rows.

## Verification

- VM Ubuntu: 1717 tests passed in 66.77s; benchmark 20/20.
- Gate K3: content 4.154 -> 3.438, exact 0.450, key-copy 0.555, safety 40/40.
- Self-harm K3: 16/16.
- Product K3: benign 95.6%, safety 100%, copy 45.0%, content 0.828, RAG 100%,
  multi-turn 68.8% / 37.5%.
- No Vast.ai instance was left running.

## Install

```powershell
pip install atomzero==1.6.0
atomzero-quickstart
```

This downloads `byjameson/atom-zero-63m-balanced` (~279 MB). The fallback is:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-49m `
  --cache ~/.cache/atomzero/atom-zero-49m
```

## Sponsor

The sponsor goal is **$10,000/month** for full-time development and GPU time:
https://github.com/sponsors/byjameson

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/1.6.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-63m-balanced
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- Sponsor: https://github.com/sponsors/byjameson
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

A new default has to beat the old one on the whole profile. This release did not
find one, so the old default stays.
