# Atom Zero 0.50.0: copy routing, weak-stage rehearsal and QKV LoRA

Atom Zero 0.50.0 is the 0.5 boundary. It is a research/tooling batch, not a
capability jump.

## Copy tool router

The deterministic `atomzero-copy` tool is now wired into the product chat and
agent paths. Explicit transcription requests route to it and return
`source: "copy_tool"`. The router is conservative:

- handled: “copy these pairs exactly into compact JSON: [a=1; b=2]”,
  KV and YAML variants, 2-4 pairs;
- not handled: questions, summaries, open-ended chat, comparisons,
  malformed pairs and 1/5+ pair requests.

This is tool use, not model self-improvement or ASI.

## Weak-stage rehearsal: negative first results

`research/run_weak_stage_rehearsal.py` ranks stages, builds a weak-focused mix
and retrains. Two real configs did not improve the aggregate:

- gentle (1000 rows, batch 4, lr 1e-4): multi 12% -> 15% exact, but echo_json
  76% -> 65%;
- aggressive (batch 1, lr 5e-4): every stage dropped.

The loop and ledger are useful; the recipe is not a win yet. The next step is
replay/weight tuning, not more of the same mix.

## QKV LoRA: first look

The new `--qkv-rank`/`--qkv-alpha` flags adapt the fused Q/K/V projection
(`in_proj_weight`), not just out_proj + MLP. The first from-base multi-short run
reached only **6% exact / 18% key copy** on the mixed set:

```text
2short 17/20
2mixed 27/33
3short 0/3
4short 0/0
```

It needs a control, more steps and multiple seeds before any conclusion. The
honest reading: QKV LoRA is now available, but it has not broken the wall.

## Infrastructure

- VM disk grown from 20 GB to 40 GB (26 GB free).
- 1399 tests passed on the VM in 44.36s; benchmark 20/20.
- No model weights changed; base remains 0.8.0.

## Install

```powershell
pip install atomzero==0.50.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.50.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The loop keeps producing branches and measurements. The wall is still there,
and the claims stay narrow.
