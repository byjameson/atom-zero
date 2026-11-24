# atomzero 0.21.0 - curriculum gate loop and first behavior-passing adapter

Status: **release candidate**. Base weights remain `0.8.0`; the product default
has no active adapter.

## Highlights

- **First behavior-passing curriculum adapter.** On a 250-row mixed held-out
  set, `rehearsal_focus` reached **40.0% exact match / 54.8% key copy /
  62.8% key recall**, with content loss `4.0449 -> 1.0011`. The behavior gate
  passed and the adapter was promoted to the separate curriculum registry
  `data/curriculum_active_adapter.json`. The product registry was not touched.
- **Curriculum gate/promotion loop.** `research/run_staged_curriculum.py` can
  now chain stages, evaluate a final mixed set, run the content/exact/key-copy
  gate and optionally promote with a content-free ledger.
- **Adapter soup.** Weighted LoRA A/B merging with strict geometry checks and
  per-split evaluation. The experiment was negative: soup scored 34.4% exact /
  46.4% key copy, worse than the rehearsal adapter on every stage.
- **Pure multi-key training.** Multi exact improved from 8% to 12%, but
  echo_json fell from 72% to 10%. Multi-key copying remains the bottleneck.
- **VM evidence.** 1264 tests passed in 50.35s; benchmark 20/20. CI includes
  `scikit-learn` for the calibration tests.

## Multi-key evidence (50 held-out rows per stage)

| stage | rehearsal_focus | pure multi | adapter soup |
|---|---:|---:|---:|
| echo_kv | 58% / 68% | 44% / 52% | 52% / 62% |
| echo_json | 72% / 82% | 10% / 10% | 68% / 74% |
| two_kv | 34% / 56% | 30% / 42% | 26% / 42% |
| two_json | 28% / 46% | 20% / 34% | 20% / 36% |
| multi | 8% / 22% | **12% / 20%** | 6% / 18% |

Values are exact match / key-copy rate. The mixed-set gate result for
`rehearsal_focus` is 40.0% / 54.8%.

## Install

```powershell
pip install atomzero==0.21.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/0.21.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-0200-gated-self-improvement-and-the-copy-curriculum-evidence-13pj>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-0-20-0-gated-self-improvement-and-the-copy-curriculum-evidence-63edc5daa21c>

## Honest limits

- A 40% exact adapter on a synthetic copy curriculum is not a general assistant
  capability and is not the product default.
- Multi-key copying is still unreliable; the 49M base with LoRA may be
  capacity-limited.
- No model weights changed; base remains `0.8.0`.
- The next release should target key-token weighting, a larger base or a
  runtime copy mechanism, and keep the behavior gate mandatory.
