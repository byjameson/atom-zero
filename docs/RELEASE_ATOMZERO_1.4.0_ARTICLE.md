# Atom Zero 1.4.0: 24 layers bought copy, not conversation

Atom Zero 1.4.0 ran the capacity experiment. The answer is unusually clean:
depth is a copy lever, not an alignment lever.

## The run

A 24-layer / 176,795,136-parameter upcycle was trained on the F recipe — copy,
content, safety, benign chat and contrastive self-harm replay — on a rented
RTX 3090. The full product suite ran on the same instance.

| area | 8L F | 24L F |
|---|---:|---:|
| 2short copy | 88.0% | **95.5%** |
| 3short copy | 72.0% | **86.5%** |
| 4short copy | 58.0% | **77.5%** |
| 4short key-copy | 67.0% | **81.0%** |
| retention | 0.841 | **0.790** |
| benign useful | 83.8% | 83.8% |
| safety (40) | 40/40 | **40/40** |
| server suite | 21/21 | **21/21** |
| self-harm gate | **16/16** | **9/16** |

Capacity added **19.5 points** of four-key copy and kept content retention
better than the 49M base. It changed benign conversation by exactly zero
points. It made the self-harm boundary worse.

## What this means

The 63M model could not separate a crisis prompt from ordinary conversation.
The 177M model does not either — and with 24 layers it is worse at the held-out
self-harm pairs (9/16 vs 16/16). More parameters make the copy circuit stronger,
not the judgment circuit.

That is not a failure of the experiment. It is the result: the bottleneck for
default-assistant quality is the benign corpus and the boundary data, not depth.

## The decision

No default promotion. The product default stays `0.8.0`.

The checkpoint download was interrupted at 31 MB and the instance was
destroyed, so no 24L artifact is published. The measured result is the release
evidence. The 8L F and 63M gated optional bases remain available.

The next branch expands the benign corpus and the contrastive boundary data,
and keeps the product crisis filter from 1.3.0.

## Why this is progress

A clean negative result saves GPU time. Before this run, depth looked like the
obvious next step; after it, the next money goes to data, not layers.

The 24L copy result is still the best copy profile in the project: 77.5%
four-key exact, 95.5% two-key, retention 0.790. If a later alignment dataset
fixes the conversation and safety axes, the capacity is already proven.

## Verification

- VM Ubuntu: **1682 tests passed**.
- Benchmark: 20/20.
- Gate 24L: content 4.154 -> 3.129, exact pass, key-copy pass, safety 40/40.
- Product 24L: benign 83.8%, safety 100%, copy 77.5%, content 0.900.
- Server 24L: 21/21, 0 errors, 5 source citations.
- Vast.ai: 0.88 h, about $0.16; instance destroyed; no leftovers.

## Sponsor

The sponsor goal is **$10,000/month** for full-time development and GPU time:
https://github.com/sponsors/byjameson

## Install

```powershell
pip install atomzero==1.4.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/1.4.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- Sponsor: https://github.com/sponsors/byjameson
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

Depth bought copy. It did not buy judgment. The next dollar goes to data.
