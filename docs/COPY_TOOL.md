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
- **Not automatic for arbitrary prompts.** The chat server routes only
  explicit transcription requests (see below). Every other prompt keeps the
  normal RAG/model path.
- **Not a parser for arbitrary prose.** It expects pair-shaped input, optionally
  inside `[...]` blocks.

## Chat router

`atomzero.copy_router` adds a small, conservative intent check in front of the
normal chat answer path. When a prompt explicitly asks to transcribe pairs, the
server returns the deterministic copy output and flags provenance with
`"source": "copy_tool"`. Otherwise the router returns `None` and the existing
RAG/model flow runs exactly as before.

The boundary is **explicit transcription only**:

- The prompt must contain a clear instruction such as `copy these pairs`,
  `transcribe`, `return only`, or `exactly these keys and values`.
- It must contain pair-shaped content (`key=value` or `key: value`) with
  **2-4 pairs**, preferably inside one `[...]` block.
- Questions, summaries, comparisons, explanations, paraphrases and open-ended
  chat are never routed, even if they mention keys or values.
- Requests outside the 2-4 pair range fall back to the normal chat path.
- Malformed pairs fall back too: `route_copy` returns `handled=False` with a
  reason instead of raising.

Handled examples:

```text
Copy these pairs exactly into compact JSON: [alpha=one; beta=two].
Return only the JSON object.
  -> {"alpha":"one","beta":"two"}            source: copy_tool

Return only one line of semicolon-separated key=value pairs:
[alpha=one; beta=two]. Copy every key and value exactly.
  -> alpha=one;beta=two                      source: copy_tool

Return only YAML with exactly these keys and values:
[alpha=one; beta=two]. Do not add prose.
  -> alpha: one
     beta: two                               source: copy_tool
```

Not handled (normal RAG/model path):

```text
What is the launch code?
Summarize the document about Apollo.
How do I copy a file in PowerShell?
Copy the launch code from the document.
Explain this config: host=localhost; port=8080.
Return only JSON.
```

Python API:

```python
from atomzero.copy_router import detect_copy_request, route_copy

detect_copy_request(prompt)   # "json" | "kv" | "yaml" | None
route_copy(prompt)            # {handled, format, answer, pairs_count, reason}
```

The router changes no weights and no copy-tool output format. It is a routing
convenience for a deterministic tool, not model self-improvement, training,
general intelligence or ASI. When the router does not handle a prompt, the
model's own behaviour is unchanged.

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
python -m pytest tests/test_copy_tool.py tests/test_copy_router.py -q
```

The copy-tool tests cover 2, 3 and 4 pairs in JSON/KV/YAML, multiword and quoted
values, colon/newline syntax, malformed input, missing/duplicate keys, CLI
output and exit codes, and the benchmark's 1.0 exact rate on a small fixture.
The router tests cover format detection, the 2-4 pair bound, malformed-pair
fallback, non-copy questions/summaries/RAG prompts and the chat helper's
`copy_tool` provenance marker.
