# Consensus algebra

A retrieved chunk can support a claim; independent chunks can support it more
than once. `atomzero.consensus` clusters sentence-level claims from local
retrieval hits by lexical overlap and reports which clusters have more than one
source. It is deterministic, local and uses no model calls.

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.consensus --file hits.json --json
python -m atomzero.consensus --min-sources 2
```

```python
from atomzero.consensus import audit_consensus

report = audit_consensus([
    {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
    {"source": "b.txt", "chunk": "The battery lasts 8 hours."},
], min_sources=2)
# {"clusters": [...], "conflicts": [...], "summary": {...},
#  "weights_changed": False, "method": "deterministic_cross_source_consensus_v1"}
```

## What it computes

- Claims are extracted from each hit with `atomzero.claims.extract_claims`.
- Claims are clustered when their token overlap is at least `min_overlap`
  (default `0.55`) and their numbers do not conflict.
- A cluster is `consensus` when it has at least `min_sources` distinct sources
  (default `2`); otherwise it is `single_source`.
- Numeric, date, identifier and explicit negation mismatches are reported by
  `atomzero.claims.compare_claims`. They are not averaged into the cluster.
- The representative text is deterministic: longest claim, then claim ID.

## HTTP

`POST /v1/consensus` accepts `hits` (maximum 50) plus optional `min_overlap`
and `min_sources`. Only `source`, `title`, `section`, `page`, `chunk` and
`snippet` are passed to the audit. It does not persist the request.

```powershell
$body = @{
  hits = @(
    @{ source = "a.txt"; chunk = "The battery lasts 8 hours." },
    @{ source = "b.txt"; chunk = "The battery lasts 8 hours." }
  )
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/consensus `
  -ContentType "application/json" -Body $body
```

## Honest limits

- Clustering is lexical overlap, not entailment. Paraphrases with little shared
  vocabulary can stay separate; copied text can cluster even when the source is
  weak.
- `consensus` means "two local sources say the same thing lexically", not
  "true". Source independence and reliability are not established.
- Numeric compatibility is a guard against merging obviously conflicting
  measures; it is not a full unit/date reasoning engine.
- A cluster can contain two hits from the same original document. The report
  counts distinct `source` labels, so callers should keep source identifiers
  meaningful.

## Verification

```powershell
$env:PYTHONPATH="$PWD\src"
python -m pytest tests/test_consensus.py tests/test_chat_server_docs.py::test_http_consensus_audit -q
python -m aisc check docs/aisc/atomzero-consensus.toml --strict
```

The AISC spec declares four verified components, no irreversible steps and
`min_assurance = 0.90`.
