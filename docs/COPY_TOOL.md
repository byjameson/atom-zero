# Deterministic runtime copy tool

Date: 2026-11-24

## What it solves

The 49.2M base model does not reliably transcribe three or four arbitrary keys
and values. In the randomized-key and staged copy evaluations, exact match on
held-out keys has stayed in the low single digits (0-6%), even after LoRA
training. The failure is transcription, not knowledge: the prompt already
contains the answer.

`atomzero-copy` performs that transcription in ordinary Python. It extracts the
explicit `key=value` or `key: value` pairs from the request and renders them as
compact JSON, one-line KV or YAML. The tool uses only the standard library, does
not import `torch`, does not load a checkpoint and does not touch model weights.

This is a capacity/locus intervention: a deterministic runtime tool handles a
task the small model cannot do reliably. It is **not** evidence that the model
learned to copy, and it is **not** a training result.

## What it is not

- **Not a model capability.** The model's own copy accuracy is unchanged.
- **Not training or self-improvement.** No weights, optimizer state, adapter or
  checkpoint are modified.
- **Not ASI or general intelligence.** It is a flat string transcription
  helper for explicit pairs.
- **Not a fact source.** It copies what the prompt says; it does not verify that
  a value is true.
- **Not automatic.** The runtime does not route model replies through it unless
  a caller explicitly invokes the tool or its functions.
- **Not a parser for arbitrary prose.** It expects pair-shaped input, optionally
  inside `[...]` blocks.

## Commands

Installed entry point:

```powershell
atomzero-copy --text "a=1;b=hello world" --format json
atomzero-copy --text "a=1;b=hello world" --format kv
atomzero-copy --text "a=1;b=hello world" --format yaml
atomzero-copy --text-file pairs.txt --format json
```

Without installation, from the repository root:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m atomzero.copy_tool --text "a=1;b=hello world" --format json
```

Output:

```text
{"a":"1","b":"hello world"}
```

KV is one line (`a=1;b=hello world`). YAML is one pair per line. Values are
treated as strings; YAML quotes values that would otherwise be read as numbers,
booleans or nulls so the copied text keeps its string type.

Piped input works too:

```powershell
"a=1;b=hello world" | atomzero-copy --format json
```

Exit codes: `0` on success, `2` for invalid pairs, an unknown format, unreadable
input or missing input. Errors are written to stderr.

## Supported pair syntax

```text
a=1; b=hello world
a: 1
b: two words
[first="hello world"; second='a; b: c']
key=value
key2=value2
```

- `=` or `:` separates a key from its value.
- `;` or a newline separates pairs; both may be mixed.
- Pairs may sit directly in the text or inside one or more `[...]` blocks.
- Values may be quoted with single or double quotes. Quoted values may contain
  spaces, semicolons, colons, brackets and escaped quotes.
- Keys must match `[A-Za-z0-9_][A-Za-z0-9_.-]*`.
- Missing keys, missing values, duplicate keys, unbalanced quotes, unclosed
  brackets and malformed segments raise `ValueError`; the CLI reports them and
  exits non-zero.

The output is flat. Nested objects, lists as values, YAML documents with
anchors, and inferred types are out of scope.

## Python API

```python
from atomzero.copy_tool import CopyTool, copy_pairs, extract_pairs

pairs = extract_pairs("[a=1; b=hello world]")
# [('a', '1'), ('b', 'hello world')]

text = copy_pairs("[a=1; b=hello world]", fmt="json")
# '{"a":"1","b":"hello world"}'

tool = CopyTool("yaml")
tool("[a=1; b=hello world]")
```

`fmt` accepts `json`, `kv`, `yaml` (and `yml` as an alias). `parse_rendered`
parses the tool's JSON/KV/YAML output back to a mapping and rejects duplicate
JSON keys.

## Benchmark

The benchmark measures the tool, not the model. It reads a staged eval JSONL,
extracts the pairs from each prompt, renders the requested format, parses the
result and compares keys and values with the row's `pairs` metadata. Rows with a
malformed prompt, unsupported format or prompt/metadata mismatch are counted
separately and excluded from the rates. The summary contains counts and rates
only; no prompt, key or value text is included.

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m research.benchmark_copy_tool
python -m research.benchmark_copy_tool --dataset data/multi_short_v1/stage_4short/eval.jsonl
python -m research.benchmark_copy_tool --dataset path/to/eval.jsonl --out copy_summary.json
```

The default dataset is `data/multi_short_v1/stage_4short/eval.jsonl`. The
`data/` tree is not tracked by git; build it with
`research/build_multi_short_queue.py` first, or pass another staged eval file.

### Result

On the real 200-row `stage_4short` eval file (100 JSON and 100 KV rows):

| metric | result |
|---|---:|
| valid rows | 200 / 200 |
| exact rows | 200 / 200 |
| exact rate | **1.000** |
| key-copy rate | **1.000** |
| value-copy rate | **1.000** |
| invalid rows | 0 |

This 1.0 is the deterministic tool's transcription rate on valid rows. It must
not be compared with the model's generation accuracy or reported as a model
capability. The model and its copy curricula are unchanged.

## Tests

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
python -m pytest tests/test_copy_tool.py -q
```

The tests cover 2, 3 and 4 pairs in JSON/KV/YAML, multiword and quoted values,
colon/newline syntax, malformed input, missing/duplicate keys, CLI output and
exit codes, and the benchmark's 1.0 exact rate on a small fixture.
