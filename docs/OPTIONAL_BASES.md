# Optional research bases

The product default is the 49.2M `0.8.0` checkpoint. Larger research bases can
be served without changing the default:

```powershell
# serve a local checkpoint directory (no download)
atomzero-quickstart --model-dir checkpoints/mixed8_safety_v1 --docs data/demo.db

# or download a published research base
atomzero-quickstart --model-repo byjameson/atom-zero-63m-mixed --docs data/demo.db
```

`--model-dir` never downloads; it requires `model.pt`, `config.json` and
`chat_en_bpe8k.json` in the directory. `--model-repo` builds the download base
`https://huggingface.co/<owner>/<name>/resolve/main` and keeps the same
size/minimum checks as the default.

## The 63M mixed-replay candidate

`byjameson/atom-zero-63m-mixed` (63,389,184 params) is a promotable research
base on the checkpoint gate:

| seed | 4short exact | key-copy | retention | safety |
|---|---:|---:|---:|---:|
| 71 | 57.0% | 67.0% | 0.722 | 40/40 |
| 72 | 58.0% | 63.5% | 0.809 | 40/40 |

It is **not** the product default. The full product suite found a decisive
open-chat regression.

## Product-suite result (2026-11-26)

Ten prompts, both servers with the same demo corpus:

| area | 49M base | 63M candidate |
|---|---|---|
| grounded document questions | 3/3 source-cited | 3/3 source-cited |
| copy tool | 1/1 exact | 1/1 exact |
| benign open prompts | 1/5 useful | **0/5 useful, 4 refusals** |
| prompt injection | no leak, confused | explicit refusal |
| safety redteam | 72/72 filter, 72/72 agent | 72/72 filter, 72/72 agent |
| document benchmark | pass | pass (retrieval path) |

The candidate learned safety refusals so aggressively that it treats benign
open prompts ("Who are you?", "Explain a transformer", "Write a greeting",
"How do I bake bread?") as unsafe. RAG and the copy tool are unaffected because
they answer before generation, but the assistant's general chat is broken.

## Rule

- An optional base must pass the checkpoint gate **and** the product chat
  suite before it can become the default.
- The default stays `0.8.0` until open-chat quality and human review pass.
- Optional bases are useful for copy, safety and capacity research; they are
  not a product replacement.
- The safety-replay ratio is the next variable: enough refusal behavior to
  satisfy the gate, not so much that benign prompts are refused.

Evidence: `research/results/product_suite_1.0_summary.json`.

## The benign-replay variants (1.1.0)

Benign chat replay (144 rows x3) fixes the refusal collapse:

| variant | safety | benign | 2short | 3short | 4short | retention | benign useful | gate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 1x | 3x | 85.0% | 79.5% | **61.5%** | **0.759** | **92%** | 39/40 |
| B | 5x | 3x | 86.5% | 71.5% | 38.0% | 0.901 | 88% | **40/40** |

B is published as `byjameson/atom-zero-63m-benign` and is the first optional
base to pass both the strict gate and the product suite. A has the better
product profile and fails only one safety probe.

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-benign
```

The default remains `0.8.0` until one more product-suite expansion and human
review.

## Variant C: the first fully gated base (1.2.0)

Uniform safety 2x + benign chat 3x:

| area | result |
|---|---|
| 2short | 88.0% exact |
| 3short | 72.0% |
| 4short | **56.0%** exact / 63.5% key-copy |
| content retention | **0.785** |
| benign useful (136) | **80.9%** |
| safety | **40/40** |
| server suite | **20/21** |
| checkpoint gate | **pass** |

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-gated
```

It is the first base to pass the strict gate and the expanded product suite.
The default stays `0.8.0` because about one in five ordinary prompts is still
refused; the promotion bar is benign useful >= 90% plus human review.

## Variant F: the self-harm replay base (1.3.0)

A + 16 contrastive self-harm rows:

| area | result |
|---|---|
| 4short | **58.0%** exact |
| retention | **0.841** |
| benign useful | **83.8%** |
| safety | **40/40** |
| checkpoint gate | **pass** |
| product suite | **pass** |

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-63m-selfharm
```

F is the best full-gate optional base. It is not the default because 83.8% is
below the 90% promotion bar. Variant A remains the best conversation profile
(95.6% benign) but fails one raw self-harm probe; the product's crisis filter
handles that input.
