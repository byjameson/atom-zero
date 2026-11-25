# Large-base capacity probe (research only)

Date: 2026-11-25

Status: research evidence about a capacity ceiling. This document does **not**
announce a new product model, a training result or a replacement for the 49.2M
Atom Zero default.

## Question

The 49.2M base model cannot reliably copy three or four arbitrary key/value
pairs. Held-out exact match has stayed in the low single digits (0-6%), before
and after LoRA training. The prompt already contains the answer, so the failure
is transcription, not missing knowledge.

Is that failure specific to the 49.2M model, or does a much larger pretrained
model handle the same staged task? This probe runs an external Hugging Face
causal LM on the staged `multi_short_v1` eval files with the model's chat
template and greedy decoding.

## What was compared

| Property | Atom Zero 49.2M | Qwen2.5-0.5B-Instruct |
| --- | --- | --- |
| Parameters | 49.2M | about 0.5B (reported) |
| Initialization | trained in this project | pretrained by a third party |
| Tokenizer | project tokenizer | Qwen BPE tokenizer |
| Product status | low-cost default (0.8.0) | research backend only |
| Weights changed by this probe | no | no |

This is not a controlled scaling experiment. Pretraining data, compute,
tokenizer, context handling and instruction tuning all differ. The comparison
can show that the copy wall is not universal at this task; it cannot attribute
the difference to parameter count alone.

## Measured probe

Model: `Qwen/Qwen2.5-0.5B-Instruct`, loaded from the local cache
(`local_files_only=True`). Data: 20 rows from each staged eval file. Decoding:
greedy, at most 64 new tokens. Exact match is after tolerant parsing, so a
correct answer wrapped in a `pairs` list or a code fence still counts.

| Stage | Pairs per row | Baseline exact | Flat-prompt exact | Rows |
| --- | ---: | ---: | ---: | ---: |
| `2short` | 2 | 65% (13/20) | 85% (17/20) | 20 |
| `3short` | 3 | 55% (11/20) | 65% (13/20) | 20 |
| `4short` | 4 | 55% (11/20) | 85% (17/20) | 20 |

The same 20-row subsets were used for the baseline and flat-prompt runs in each
stage. "Flat prompt" appends one instruction: return only a flat compact JSON
object, with no `pairs` list, code fences or prose.

The model frequently emitted `{"pairs": [["key", "value"], ...]}` or markdown
fences. A strict flat-JSON parser therefore undercounts the copy content. The
harness in `research/eval_large_base.py` accepts:

- a flat JSON object;
- `{"pairs": [[key, value], ...]}`;
- `{"pairs": [{"source": key, "target": value}, ...]}`;
- the same objects inside code fences or surrounding prose;
- one-line KV and YAML, via `atomzero.copy_tool.extract_pairs`.

The summary also reports flat-format compliance separately, so content accuracy
and instruction-format following are not conflated. The manual probe numbers
above are tolerance-aware exact matches; strict format compliance was not
recorded.

## Why this is capacity evidence

- The task is pure copying. No world knowledge or reasoning is required.
- The 49.2M model stays at 0-6% exact match on held-out keys even with LoRA.
- A pretrained model roughly ten times larger reaches 55-85% exact match on the
  same staged data and prompt style.
- The larger model's failures are often formatting failures rather than copy
  failures, which is consistent with a capacity/instruction-following limit in
  the small model rather than a data-labeling problem.

This is evidence that model capacity is a plausible bottleneck. It is not proof
that capacity is the only bottleneck, and it is not a controlled ablation.

## Why this is not a product claim

- **Different pretraining.** Qwen2.5-0.5B-Instruct was pretrained on a large
  external corpus. The 49.2M model was not. The comparison mixes scale with
  data, compute and tokenizer differences.
- **Small samples.** Twenty rows per stage give wide uncertainty; a 55% result
  at n=20 has a 95% interval of roughly 32-77%. Differences of 10-30 points on
  this sample are directional, not precise.
- **Tolerant parsing.** Exact match measures copied content after unwrapping.
  It does not mean the model followed the requested output format.
- **No integration evidence.** The probe does not run the chat server, RAG,
  citations, answerability calibration, safety checks, adapters, packaging or
  latency/memory budgets.
- **Not a like-for-like replacement.** A 0.5B pretrained model is a different
  product trade-off, not an upgrade of the 49.2M weights. It is also not a
  from-scratch result for this project.

The deterministic `atomzero-copy` tool remains the reliable path for explicit
transcription requests (100% exact on the staged copy-tool benchmark). The
model probe measures the neural path only.

## Decision

1. **Keep the 49.2M 0.8.0 model as the low-cost default.** No base weights,
   trainer behavior, gate thresholds or product defaults change.
2. **Evaluate a larger research backend separately.** Any larger pretrained
   model runs behind a research-only backend/interface and is reported as an
   external baseline, not as Atom Zero.
3. **Do not replace base weights without a full integration and gate.** A
   replacement would need the capability scorecard, answerability calibration,
   citation and content audits, safety checks, latency/memory measurements,
   packaging and license review, plus a documented rollback.

This keeps the capacity finding honest: it explains the wall and points to a
possible research direction without turning a 20-row probe into a release.

## Reproduction

From the repository root, with the model already in the Hugging Face cache:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

# Default dataset: data/multi_short_v1/stage_4short/eval.jsonl
python -m research.eval_large_base --rows 20 --device cpu `
    --out research/results/large_base_4short_baseline.json

python -m research.eval_large_base --rows 20 --device cpu --flat-prompt `
    --out research/results/large_base_4short_flat.json

# Other stages
python -m research.eval_large_base `
    --data data/multi_short_v1/stage_2short/eval.jsonl `
    --rows 20 --device cpu --out research/results/large_base_2short.json
```

`--local-files-only` is true by default, so a missing cache entry fails instead
of downloading. Pass `--no-local-files-only` only when a download is intended.
The dataset files are gitignored; generate them with the existing
`multi_short_v1` builder before running the probe.

The committed summary is `research/results/large_base_probe.json`. It contains
only counts, rates and settings, never prompts, keys, values or generations.

## Metrics

- `exact_match`: fraction of evaluated rows whose parsed mapping equals the
  expected mapping exactly (same keys and values, no extras).
- `key_recall`: mean fraction of expected keys present.
- `value_recall`: mean fraction of expected key/value pairs copied exactly.
- `parse_failures`: generations from which no non-empty mapping could be
  recovered.
- `flat_format_rate`: fraction of raw generations that are exactly one flat
  JSON object with no code fences or `pairs` wrapper.
- `rows_invalid`: malformed eval rows, excluded from the rates.


## Transfer probe (negative)

The same Qwen2.5-0.5B-Instruct probe was run zero-shot on 10 arithmetic
rows and 10 city-YAML rows. Both scored **0/10 exact** and all outputs were
parser-invalid. The larger model solves explicit copy, but it does not
automatically solve a different task family. Capacity evidence is
task-specific; the 49M default and the runtime copy tool remain unchanged.
