# Atom Zero 0.30.0: the best copy adapter so far and the 3-4 key wall

Atom Zero 0.30.0 is the first release at the new cadence: one release every ten
minor versions, so 0.20.0 -> 0.30.0 -> 0.40.0.

## The best copy adapter so far

The recipe that worked was **specialize, then rehearse**:

1. train on multi-key data (`multi_pure`);
2. continue on the balanced rehearsal mix at low LR.

`multi_recover` reached:

```text
mixed held-out (250 rows):
exact match:   48.0%
key copy:      60.4%
key recall:    70.0%
content loss:  4.6015 -> 0.7789
full loss:     4.6015 -> 0.7769
gate:          pass / curriculum registry
```

Per stage (50 rows, exact / key copy):

| stage | multi_recover |
|---|---:|
| echo_kv | 62% / 74% |
| echo_json | 76% / 80% |
| two_kv | 44% / 54% |
| two_json | 46% / 72% |
| multi | 12% / 22% |

That is a real improvement over the previous best (40.0% / 54.8%) and it keeps
the earlier skills.

## The 3-4 key wall

Two follow-up experiments tried to break it:

- **key-weighted loss**: weight key-name tokens 10x and values 5x. Mixed result:
  40.4% / 53.2%. 3short 2/8, 4short 0/0.
- **multi-key short-value curriculum**: 2short/2mixed/3short/4short, shorter
  keys and single-token values. Mixed result: 42.0% / 54.4%. 3short 6/24,
  4short 0/0.

Both help 2-key rows and fail on 3-4 keys. The limit is likely capacity/locus:
a 49M base with LoRA out_proj + MLP cannot reliably learn arbitrary multi-key
copying. More of the same data is not the answer.

## The autonomous loop

`atomzero-self-improve` can now run the staged curriculum runner with
`--adapter-curriculum-command`, record the gate/promotion block and fail closed
on broken commands. VM evidence: `status: completed`, gate pass, exact 0.40,
key-copy 0.548. The dry-run path does not claim a promotion.

## Install

```powershell
pip install atomzero==0.30.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.30.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The system now has a measured, gated copy adapter and a working curriculum
loop. It still cannot copy arbitrary 3-4 key structures, and it is not ASI. The
next release should test a larger base or a runtime copy mechanism.
