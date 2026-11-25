# Depth-upcycle report

Schema: `atomzero.upcycle_report.v1`  
Status: research only; no training was run.

## Source and target

- Base: `C:/Users/biris/.cache/atomzero/atom-zero-49m`
- Output: `checkpoints/upcycle/atom-zero-49m-8l`
- Mode: `interpolate`
- Seed: 23
- Layers: 6 -> 8
- Tokenizer: C:/Users/biris/.cache/atomzero/atom-zero-49m/chat_en_bpe8k.json

## Parameter counts

| model | layers | parameters | added | fp32 weights |
|---|---:|---:|---:|---:|
| source | 6 | 49,213,440 | - | 0.197 GB |
| target | 8 | 63,389,184 | 14,175,744 (28.8%) | 0.254 GB |

The LM head is tied to `tok.weight`; counts exclude the duplicate state-dict entry. A checkpoint file is roughly the unique parameter count times four bytes, plus small container overhead.

## Layer mapping

Target block -> source block:

`[0, 0, 1, 2, 3, 3, 4, 5]`

New blocks use a blend of the source block and its neighbor:

| target | source | neighbor | alpha |
|---:|---:|---:|---:|
| 1 | 0 | 1 | 0.500 |
| 5 | 3 | 4 | 0.500 |

## Compute and data (RTX 3050 6 GB)

| item | estimate |
|---|---:|
| Full fine-tune state (weights + AdamW) | 1.01 GB |
| Activation lower bound | 0.13 GB |
| Practical training memory range | 1.5-2.6 GB |
| Fit on 6 GB | likely |
| Estimated throughput | 11,357 tokens/s |
| Chinchilla-style token target (20/param) | 1,267,783,680 tokens |
| Time for that token target | 31.0 hours |
| Time for the existing 214.8M replay mix | 5.3 hours |

Rough planning estimate. The lower activation bound omits allocator, attention and CUDA-context overhead; measure the actual fit before starting a long run. On 6 GB, reduce batch size or use gradient accumulation/checkpointing if CUDA runs out of memory.

The data budget is usually the binding constraint: the existing replay mix is about 215M tokens, while a 20-tokens-per-parameter budget is larger than 1B tokens for both 8 and 12 layers. More clean data is needed before copied depth can be judged.

## Caveats

- This is a research base, not the product 0.8.0 checkpoint and not a capability claim.
- No training was run. New layers are copied or interpolated from the base; copied depth alone does not add capability.
- The checkpoint needs continued research pretraining or fine-tuning on cached data before it can be evaluated fairly.
- Parameter counts exclude the tied LM head, which shares storage with tok.weight.
- Memory, throughput and token figures are planning estimates for an RTX 3050 6 GB, not benchmarks.
- Keep the product 0.8.0 checkpoint unchanged. Do not promote this checkpoint without the full capability, behavior and safety gate.

## Next step

Continue research pretraining or fine-tuning on cached data, then evaluate held-out loss, behavior, safety and the product benchmark. Do not write this checkpoint into `data/active_adapter.json`, the promotion path or product defaults.

No prompt or answer text is stored in this report.
