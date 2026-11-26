# atomzero 1.4.0 - more capacity buys copy, not conversation

Status: **published on 2026-11-26**. Product base weights remain `0.8.0`.

## The capacity experiment

A 24-layer / 176,795,136-param upcycle was trained on the F recipe (copy +
content + safety + benign + contrastive self-harm replay) on a rented RTX 3090.
The full evaluation ran on the same instance.

| area | 24L result |
|---|---|
| 2short copy | **95.5%** exact |
| 3short copy | **86.5%** |
| 4short copy | **77.5%** exact / 81.0% key-copy |
| content retention | **0.790** |
| benign useful (136) | 83.8% |
| safety (40) | **40/40** |
| checkpoint gate | **pass, promotable** |
| product suite | **pass** |
| server suite | **21/21**, 5 source citations |
| self-harm gate | **9/16 (fail)** |

The 8L F model scores 58.0% on 4short but **16/16** on the same self-harm set.

## What it means

Capacity added **19.5 points** of four-key copy and left benign conversation
unchanged at 83.8%. It made the self-harm boundary worse: 24L refuses 9 of 16
held-out crisis/benign pairs, while 8L handles all 16.

This is the cleanest result of the batch: depth is a copy lever, not an
alignment or conversation lever. The bottleneck is the benign corpus and the
boundary data, not parameter count.

## Decision

No default promotion. The product default stays `0.8.0`. The 24L checkpoint
download was interrupted at 31 MB and the instance was destroyed, so no 24L
artifact is published; the measured result is the release evidence.

The next branch expands the benign corpus and the contrastive boundary data,
and keeps the product crisis filter from 1.3.0.

## Verification

- VM: 1682 tests passed in 46.59s.
- Benchmark: 20/20.
- Gate 24L: content 4.154 -> 3.129, exact match pass, key-copy pass, safety 40/40.
- Product 24L: benign 83.8%, safety 100%, copy 77.5%, content 0.900.
- Server 24L: 21/21, 0 errors, 5 source citations.
- Vast.ai: 0.88 h, about $0.16; instance destroyed; no leftovers.

## Install

```powershell
pip install atomzero==1.4.0
```

## Links

- GitHub: <https://github.com/byjameson/atom-zero>
- PyPI: <https://pypi.org/project/atomzero/1.4.0/>
- Hugging Face: <https://huggingface.co/byjameson/atom-zero-49m>
- ModelScope: <https://modelscope.cn/models/byjameson/atom-zero-49m>
- Sponsor: <https://github.com/sponsors/byjameson>
- X: <https://x.com/aigokhan>
- Instagram: <https://instagram.com/quantsinner>
- dev.to: <https://dev.to/mycute_97b39c4e34027570f9/atom-zero-140-24-layers-bought-copy-not-conversation-37c7>
- Medium: <https://medium.com/@myredditacc9393/atom-zero-1-4-0-24-layers-bought-copy-not-conversation-e4a2ce73d9f3>
