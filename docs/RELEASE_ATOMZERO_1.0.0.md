# atomzero 1.0.0 - optional bases and the product-suite reality check

Status: **release candidate**. Product base weights remain `0.8.0`.

## Highlights

- **Optional research bases.** `atomzero-quickstart` now accepts
  `--model-dir PATH` (serve a local checkpoint, no download) and
  `--model-repo owner/name` (download a published research base). The default
  package still ships the 49.2M `0.8.0` base.
- **The 63M candidate failed the product suite.** The mixed-replay checkpoint
  that passed every 0.90.0 gate now served the same demo corpus side by side
  with the default:

| area | 49M base | 63M candidate |
|---|---|---|
| grounded document questions | 3/3 source-cited | 3/3 source-cited |
| copy tool | 1/1 exact | 1/1 exact |
| benign open prompts | 1/5 useful | **0/5 useful, 4 refusals** |
| prompt injection | no leak, confused | explicit refusal |
| safety redteam | 72/72 filter, 72/72 agent | 72/72 filter, 72/72 agent |
| document benchmark | pass | pass (retrieval path) |

  The candidate's safety replay overfits: "Who are you?", "Explain a
  transformer", "Write a greeting" and "How do I bake bread?" are refused as
  unsafe. RAG and the copy tool answer before generation, so they are
  unaffected; general chat is broken.
- **Decision:** the 63M candidate remains an optional research base. The
  product default stays `0.8.0`. A gate pass is necessary but not sufficient.
- **Carried from 0.90.0:** retention evaluation, checkpoint gate, mixed replay,
  content rows, safety probes/replay, pretokenized cache and the Vast.ai
  rent-run-destroy pipeline. The 24L depth experiment remains rejected: +4
  points on 4-key, retention 1.66.
- **Verification:** 1613 tests passed on the VM in 49.31s; benchmark 20/20.

## What ships

- `--model-dir` and `--model-repo` in `atomzero-quickstart`.
- `docs/OPTIONAL_BASES.md` with the candidate card, evidence and promotion rule.
- `research/results/product_suite_1.0_summary.json` (content-free).
- Quickstart tests for local bases, custom repositories and bad repo names.

## Why the default does not change

The 63M candidate is excellent at four-key copy and has better content loss
than the 49M base, but it refuses ordinary conversation. Replacing the default
would trade a working assistant for a copy/safety specialist. The next
experiment is the safety-replay ratio: enough refusal behavior to pass the gate
without refusing benign prompts.

## Install

```powershell
pip install atomzero==1.0.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.0.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
