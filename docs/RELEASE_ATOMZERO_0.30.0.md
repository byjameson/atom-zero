# atomzero 0.30.0 - multi-recover, key weighting and the 3-4 key wall

Status: **published on 2026-11-24**. Base weights remain `0.8.0`.

## Highlights

- **Best copy adapter so far.** Specialize on multi, then rehearse on the
  balanced mix. `multi_recover` reached **48.0% exact / 60.4% key copy /
  70.0% key recall** on the 250-row mixed held-out set, with content loss
  `4.6015 -> 0.7789`. The behavior gate passed and the curriculum registry was
  updated.
- **Key/value-aware token weighting.** `--key-weight` and `--value-weight`
  weight key-name and value tokens separately. It did not break the 3-4 key
  wall: mixed 40.4% / 53.2%, 3short 2/8, 4short 0/0.
- **Multi-key short-value curriculum.** 2short / 2mixed / 3short / 4short.
  It helps 2-key rows (42/54 and 44/58) but 3-4 keys remain 0-6% exact.
- **Self-improve curriculum integration.** The autonomous loop can run the
  staged runner, record the gate/promotion block, and fail closed on broken
  commands. VM evidence: `status: completed`, gate pass, exact 0.40,
  key-copy 0.548.
- 1303 tests passed on the host; VM full-suite evidence is recorded at release.

## Per-stage evidence (50 rows, exact / key copy)

| stage | multi_recover |
|---|---:|
| echo_kv | 62% / 74% |
| echo_json | 76% / 80% |
| two_kv | 44% / 54% |
| two_json | 46% / 72% |
| multi | 12% / 22% |

## The 3-4 key wall

The short-value curriculum and key weighting both fail on 3-4 arbitrary keys.
This is evidence that the limit is capacity/locus, not just data:
a 49M base with LoRA out_proj + MLP cannot reliably learn arbitrary multi-key
copying. The next release should test a larger base or a runtime copy
mechanism, not more of the same data.

## Install

```powershell
pip install atomzero==0.30.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.30.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-0300-the-best-copy-adapter-so-far-and-the-3-4-key-wall-96p>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-0-30-0-the-best-copy-adapter-so-far-and-the-3-4-key-wall-934b23dc6506>

## Honest limits

- 48% exact on a synthetic copy curriculum is not a general assistant
  capability and is not the product default.
- 3-4 key copying is still unreliable.
- No model weights changed; base remains `0.8.0`.
