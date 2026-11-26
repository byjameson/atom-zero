# Atom Zero 1.5.0: the first model good enough to be the default

For the first time, a new base passes every gate Atom Zero has built:

- **93.4%** benign useful on the 136-prompt product suite.
- **40/40** held-out safety probes.
- **16/16** held-out self-harm / benign-support pairs.
- **48.5%** four-key copy exact, **0.703** content retention.
- Checkpoint gate, product suite and server suite all pass.

The default moves from the 49.2M `0.8.0` base to the 63M balanced base
(`0.9.0`). The 49M model stays as the fallback.

## How it got here

1.0.0 shipped the 63M candidate and failed ordinary conversation: 4% benign
useful, 96% refusals. 1.1.0 found that benign chat replay fixes the collapse.
1.2.0 widened the product suite and exposed the trade: A had 95.6% benign but
failed one self-harm probe; B and C were safer but refused too much. 1.3.0
tried contrastive self-harm replay and measured the cost. 1.4.0 added capacity
and learned that depth buys copy, not judgment.

1.5.0 changed the balance:

```text
6000 copy + 492 content
+ 480 benign chat rows x3
+ 40 safety rows x5
+ 16 self-harm rows x5
+ 96 contrastive boundary pairs
```

The benign corpus fixes conversation. The safety and self-harm repeats keep
refusal behavior from collapsing. The boundary pairs teach near-neighbor
requests without turning every prompt into a refusal.

## A gate bug mattered

The self-harm gate was scoring benign support rows as failures: those eval rows
had no `expect` field, so the reader defaulted them to `"refusal"`. The gate now
infers `expect` from the row `kind` (`crisis` -> refusal, `benign` -> safe), and
the builder writes it explicitly.

With the fix, J goes from an apparent 8/16 to the correct **16/16**. The 1.4.0
24L self-harm result (9/16) also needs re-reading with the fixed data; the
"capacity hurt the boundary" conclusion is now provisional. That is the kind of
mistake a product suite exists to catch.

## What is still bounded

- Copy is inside the trained key grammar; out-of-distribution key styles fail.
- The benign and safety suites are authored screens, not public benchmarks.
- Self-harm templates are narrow; crisis responses require human review.
- The 4-key runtime copy tool remains the exact path for arbitrary keys.
- The 49M base remains available as a fallback.

## The default

```powershell
pip install atomzero==1.5.0
atomzero-quickstart
```

This downloads `byjameson/atom-zero-63m-balanced` (~279 MB) into
`~/.cache/atomzero/atom-zero-63m-balanced`.

Fallback:

```powershell
atomzero-quickstart --model-repo byjameson/atom-zero-49m `
  --cache ~/.cache/atomzero/atom-zero-49m
```

## Verification

- VM Ubuntu: **1693 tests passed in 46.87s**.
- Benchmark: 20/20.
- Gate J: content 4.154 -> 2.920, exact 0.485, key-copy 0.585, safety 40/40.
- Self-harm J: 16/16.
- Product J: benign 93.4%, safety 100%, copy 48.5%, content 0.703.
- Server J: 20/21, 0 errors, 5 source citations.

## Sponsor

The sponsor goal is **$10,000/month** for full-time development and GPU time:
https://github.com/sponsors/byjameson

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/1.5.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-63m-balanced
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- Sponsor: https://github.com/sponsors/byjameson
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The default is no longer the model that was easiest to train. It is the model
that passed the whole suite.
