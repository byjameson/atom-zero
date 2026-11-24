# Atom Zero 0.40.0: the loop promotes on its own, and a copy tool for the wall

Atom Zero 0.40.0 is the 0.4 release boundary (0.20.0 -> 0.30.0 -> 0.40.0).

## The autonomous loop actually promoted

We ran `atomzero-self-improve` on the VM with a real two-stage curriculum:

- train `echo_kv`;
- continue with `echo_json`;
- evaluate a 100-row mixed held-out set;
- run the content + exact-match + key-copy gate;
- promote automatically on pass.

Result:

```text
status:      completed
gate:        pass
exact:       0.53
key copy:    0.60
promotion:   promoted: true
registry:    data/self_improve_curriculum_active.json
```

The first attempt used a 0.40 exact threshold and failed closed:
**0.37 exact / 0.55 key copy**, content loss not improved. It did not write a
registry. That is the behavior we want from an autonomous loop.

The promoted adapter is an echo-stage curriculum adapter, not a product
assistant. Base `0.8.0` remains the product default.

## The runtime copy tool

The 3-4 arbitrary key wall is real: key weighting, multi-short data and adapter
soup all fail at 0-6% exact. So 0.40.0 adds a deterministic runtime tool:

```powershell
python -m atomzero.copy_tool --text "a=1;b=hello world" --format json
{"a":"1","b":"hello world"}
```

Benchmark on the real 4-key eval set:

```text
200/200 exact
key-copy 1.0
value-copy 1.0
```

This is tool use, not a model capability. The model keeps language and
tool-selection; the tool handles exact transcription.

## Evidence

- 1337 tests passed on the VM in 48.82s.
- Benchmark 20/20.
- No model weights changed; base remains `0.8.0`.

## Install

```powershell
pip install atomzero==0.40.0
atomzero-quickstart
```

## Links

- GitHub: https://github.com/byjameson/atom-zero
- PyPI: https://pypi.org/project/atomzero/0.40.0/
- Hugging Face: https://huggingface.co/byjameson/atom-zero-49m
- ModelScope: https://modelscope.cn/models/byjameson/atom-zero-49m
- X: https://x.com/aigokhan
- Instagram: https://instagram.com/quantsinner

The loop can now measure, gate and promote a narrow capability on its own. The
runtime tool covers the copy wall. It is not ASI, and the product still says
“I don’t know” when it doesn’t.
