# Server product suite

`research/server_product_suite.py` checks the path a user actually talks to.
It starts `python -m atomzero.quickstart`, seeds the quickstart demo corpus,
sends authored HTTP requests and reports four content-free areas:

| area | requests | pass rule |
|---|---:|---|
| `rag` | 5 | expected answer substring **and** the expected source citation |
| `copy` | 3 | exact deterministic copy answer in compact JSON or one-line KV |
| `multi_turn` | 3 conversations / 6 requests | second answer is non-empty, non-refusal and at least 4 words |
| `benign` | 10 | ordinary answer is non-empty, non-refusal and at least 4 words |

`server_ok` requires `rag >= 0.8`, `copy >= 0.8`, `multi_turn >= 0.66` and
`benign >= 0.8`. The thresholds are inclusive. The command exits `0` only when
`server_ok` is true.

## Why this is separate from the product suite

`research/product_suite.py` loads a checkpoint directly and measures benign
answers, safety probes, copy tensors and content loss. That is the right screen
for weights, but it never starts the HTTP server and never sees these paths:

- **RAG is a server path.** The quickstart inserts four demo documents into
  `DocMemory`; `/v1/documents/answer` performs retrieval and returns an
  extractive answer with `sources`. A direct model load bypasses the corpus,
  the retrieval index, the citation gate and the answer template entirely.
- **The copy tool is a server path.** `atomzero.copy_router` recognises an
  explicit transcription request in the chat handler and returns a
  deterministic `copy_tool` answer without calling the model. Loading the model
  directly cannot test that routing, tool metadata or exact rendering.
- **Multi-turn chat is a server path.** A messages array is formatted by
  `chat_prompt` and sent through `/v1/chat/completions`. The request shape,
  role ordering and server-side handling are part of the product surface.
- **The quickstart corpus is a server path.** The suite uses the same demo
  launch, warranty, security and travel notes that a new user sees, so a
  passing RAG result is evidence that the shipped demo works end to end.

This runner is a wiring and product screen, not a model benchmark. A failure
means the served product path is broken or the checkpoint cannot answer after
retrieval; it does not estimate general model quality.

## Run it

Run from the repository root in PowerShell:

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"

python research/server_product_suite.py `
  --model-dir checkpoints/mixed8_benign_b `
  --port 8031 `
  --timeout 120 `
  --out research/results/server_product_suite_mixed8_benign_b.json `
  --json
```

The same command against the 49M cache:

```powershell
python research/server_product_suite.py `
  --model-dir "$env:USERPROFILE\.cache\atomzero\atom-zero-49m" `
  --port 8032 `
  --timeout 120 `
  --out research/results/server_product_suite_49m.json `
  --json
```

Both commands use `--model-dir`, so the quickstart never downloads weights.
The checkpoint directory must already contain `model.pt`, `config.json` and
`chat_en_bpe8k.json`.

| flag | default | meaning |
|---|---|---|
| `--model-dir` | required | existing checkpoint directory for the quickstart |
| `--port` | `8031` | local quickstart port |
| `--timeout` | `120` | startup and per-request timeout in seconds |
| `--out` | empty | write the content-free JSON report |
| `--json` | off | print the JSON report |
| `--keep-server` | off | leave the quickstart process running after the suite |

The runner starts exactly this subprocess:

```text
python -m atomzero.quickstart --no-browser --model-dir PATH --port PORT --docs TEMP_DB
```

The temporary document DB and server log live in a temporary directory. Both
are removed after the run unless `--keep-server` is used. The server is stopped
in a `finally` block, including when a request or startup check fails.

## Metric definitions

### RAG

Five authored questions use the quickstart demo corpus:

| case | expected answer | expected source |
|---|---|---|
| launch code | `L-7781` | `launch_policy.txt` |
| warranty window | `30 days` | `warranty.txt` |
| visitor badge | `blue badge` | `security.txt` |
| travel portal code | `TR-42` | `travel.txt` |
| demo policy | `06:30 UTC` | `launch_policy.txt` |

Each request is `POST /v1/documents/answer` with `{"query": ...}`. A case passes
only when the normalized answer contains the expected substring **and** the
response cites the expected source. A citation is a matching `sources` record,
a path that contains the expected file name, or a rendered marker such as
`(source: launch_policy.txt)`.

`source_citations` counts cases with the expected citation even when the answer
substring is missing, so retrieval and answer extraction remain distinguishable
in the report. `refusals` counts answers matching the lexical refusal screen.

### Copy

Three explicit prompts ask for two to four key/value pairs in compact JSON or
one-line KV, for example:

```text
Copy these pairs exactly as JSON: [project=atlas; owner=mina; tier=gold]
```

The request goes through `/v1/chat/completions`. The server-side copy router
returns a deterministic answer without model generation. A case passes only
when the returned text is exactly the expected rendering after trimming outer
whitespace. This is case-sensitive, so key and value case errors fail.

### Multi-turn

Three two-turn conversations use `/v1/chat/completions` with a growing
`messages` array. The second request includes the first user turn and the first
assistant reply. A conversation passes only when:

1. both requests complete without an HTTP/parse error, and
2. the second answer is non-empty, not a refusal, and has at least
   `MIN_USEFUL_WORDS` (4) words.

`refusals` counts second-turn lexical refusals. `errors` counts failed requests,
so one conversation can contribute up to two errors.

### Benign

Ten ordinary single-turn prompts come from
`research/build_benign_prompts.py` when it imports successfully. The selection
is deterministic (seed `17`) and spread across the builder's category order.
If the builder is unavailable, a fixed local list of ten ordinary prompts is
used.

Each answer is generated through `/v1/chat/completions` and classified with the
same rule as `research/product_suite.py`:

- `useful` / pass: non-empty, at least 4 words, no refusal marker.
- `refusal`: contains a marker such as `I can't help`, `cannot assist`,
  `I'm sorry`, or a phrase from `atomzero.safety.REFUSALS`.
- `short`: non-empty, non-refusal, fewer than 4 words.
- `empty`: whitespace only.

The report keeps only counts and rates. It never stores the prompt or answer.

### Report and decision

Every area reports `examples`, `passed`, `pass_rate`, `refusals`,
`refusal_rate`, `source_citations` and `errors`. `totals` sums those counts.
`checks` records `pass`, `fail` or `skipped`, the value and the threshold.

```json
{
  "method": "server_product_suite_v1",
  "status": "pass",
  "server_ok": true,
  "decision": "server_ok",
  "areas": {
    "rag": {"examples": 5, "passed": 5, "pass_rate": 1.0,
            "refusals": 0, "source_citations": 5, "errors": 0},
    "copy": {"examples": 3, "passed": 3, "pass_rate": 1.0,
             "refusals": 0, "source_citations": 0, "errors": 0},
    "multi_turn": {"examples": 3, "passed": 2, "pass_rate": 0.666667,
                   "refusals": 0, "source_citations": 0, "errors": 0},
    "benign": {"examples": 10, "passed": 8, "pass_rate": 0.8,
               "refusals": 1, "source_citations": 0, "errors": 0}
  },
  "checks": {
    "rag_ok": {"status": "pass", "value": 1.0, "threshold": 0.8},
    "copy_ok": {"status": "pass", "value": 1.0, "threshold": 0.8},
    "multi_turn_ok": {"status": "pass", "value": 0.666667,
                      "threshold": 0.66},
    "benign_ok": {"status": "pass", "value": 0.8, "threshold": 0.8}
  },
  "content_free": true
}
```

Values above are illustrative, not a new measurement. A missing or empty area
has a `None` rate, a `skipped` check and forces `server_ok` to false.

## Caveats

- **Small authored screen.** Five RAG questions, three copy prompts, three
  conversations and ten benign prompts catch wiring failures and large product
  regressions; they do not estimate general quality or a confidence interval.
- **RAG matching is lexical.** The expected substring must appear in the
  extractive answer and the expected source must be cited. A correct paraphrase
  that omits the code or file name fails. This is deliberate for a deterministic
  server check.
- **Copy is a tool check.** The deterministic copy tool does not measure model
  transcription ability. A pass proves routing and exact rendering; a fail
  points at the server handler, router or copy tool.
- **The refusal screen has false positives.** Phrases such as "I'm sorry to
  hear that" count as refusals. Read the live replies before changing data.
- **Usefulness is length plus non-refusal.** There is no semantic on-topic
  grader. A fluent four-word wrong answer counts as useful.
- **The demo corpus is tiny.** RAG covers only the quickstart launch, warranty,
  security and travel notes. It is not a document-retrieval benchmark.
- **CPU runs are slow.** The runner does not require a GPU, but model startup
  and benign/multi-turn generation can exceed the default timeout on a slow
  CPU. Raise `--timeout` rather than lowering the bars.
- **No weights change.** The runner starts a read-only quickstart server and
  never trains, downloads or promotes a checkpoint.

## Verification

```powershell
$env:PYTHONPATH="$PWD;$PWD\src"
$env:PYTHONIOENCODING="utf-8"
python -m pytest tests/test_server_product_suite.py -q
```

The tests use a fake process, a recording stopper and a fake HTTP function.
They cover case parsing, useful/refusal classification, source-citation
detection, exact copy, threshold boundaries, content-free reports and cleanup.
No real model, port or GPU is used.
