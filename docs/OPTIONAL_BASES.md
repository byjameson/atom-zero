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
