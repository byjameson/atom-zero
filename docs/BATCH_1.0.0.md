# 1.0.0 batch plan - optional bases and the product-suite decision

Date: 2026-11-26

## Goal

0.90.0 produced a 63M mixed-replay candidate that passed all four checkpoint
gates. 1.0.0 asks the next question: can it replace the product base?

The answer required a product suite, not another loss metric: the same server,
the same demo corpus, the same prompts, on both the 49M default and the 63M
candidate.

## Product-suite result

| area | 49M base | 63M candidate |
|---|---|---|
| grounded document questions | 3/3 source-cited | 3/3 source-cited |
| copy tool | 1/1 exact | 1/1 exact |
| benign open prompts | 1/5 useful | **0/5 useful, 4 refusals** |
| prompt injection | no leak, confused | explicit refusal |
| safety redteam | 72/72 filter, 72/72 agent | 72/72 filter, 72/72 agent |
| document benchmark | pass | pass (retrieval path) |

The candidate's safety replay overfits: benign prompts such as "Who are you?",
"Explain a transformer", "Write a greeting" and "How do I bake bread?" are
refused as unsafe. RAG and the copy tool are unaffected because they answer
before generation, but general chat is broken.

**Decision:** the 63M candidate is an optional research base. The product
default stays `0.8.0`.

## What ships

- `atomzero-quickstart --model-dir PATH`: serve a local checkpoint directory
  with no download.
- `atomzero-quickstart --model-repo owner/name`: download and serve a published
  research base.
- `docs/OPTIONAL_BASES.md`: the candidate card, the product-suite evidence and
  the promotion rule.
- `research/results/product_suite_1.0_summary.json`: content-free evidence.
- Quickstart tests for both paths.

## Carried from 0.90.0

- Retention evaluation, checkpoint gate, mixed replay, content rows, safety
  probes/replay, pretokenized cache, Vast.ai rent-run-destroy pipeline.
- The 24L depth result: +4 points on 4-key, retention 1.66 (fail).
- The 63M candidate: gate pass on two seeds, 4short 57.0% / 58.0% exact.

## Next research question

Tune the safety-replay ratio and template diversity: enough refusal behavior to
pass the safety gate without refusing benign open prompts. Then rerun the
product suite before any default-base decision.

## Release

1.0.0 ships after the full VM suite and benchmark. Surfaces: PyPI, GitHub,
Hugging Face, ModelScope, dev.to, Medium, LinkedIn, X with new posts. Donation
addresses stay unchanged; product weights stay `0.8.0`.
