# Memory lifecycle

This branch covers what happens when local memory is learned, replaced or
forgotten. The first shipped slice is deliberately small and verifiable:

- direct HTTP document/session deletion requires an explicit boolean
  `confirm: true`;
- a successful delete appends one **content-free** event to a local JSONL
  ledger;
- the ledger can be listed and verified without exposing document text,
  queries, answers or session content.

## Confirmation

```powershell
POST /v1/documents/forget
{"source": "note.txt", "confirm": true}

POST /v1/memory/forget
{"session_id": "web-abc", "text": "alice@example.test", "confirm": true}
```

Missing/false confirmation returns **409** (`confirmation_required`); a
non-boolean value returns **400**. The old direct-HTTP bypass is closed. The CLI
delete path is already explicit.

## Ledger

```powershell
GET /v1/memory/ledger?limit=20
POST /v1/memory/ledger/verify
python -m atomzero.memory_ledger --list
python -m atomzero.memory_ledger --verify
```

Example event:

```json
{
  "id": "m_4f0c...",
  "ts": "2026-11-23T20:31:02",
  "op": "document_forget",
  "scope": "document",
  "target": "note.txt",
  "before": {"chunks": 2},
  "after": {"chunks": 0},
  "actor": "local",
  "request_id": "",
  "status": "committed",
  "meta": {}
}
```

The ledger validator rejects content fields such as `text`, `query`, `answer`,
`chunk`, `snippet`, `content`, `document`, `prompt`, `message`, `fact` and
`episode`, recursively. Counts such as `chunks` are allowed.

## AISC

[`docs/aisc/atomzero-memory-lifecycle.toml`](aisc/atomzero-memory-lifecycle.toml)
models `MemoryRequestParser -> ScopeResolver -> HumanConfirmation ->
MemoryStore -> LedgerWriter`. The irreversible store is after the human gate;
the ledger append is reversible and has no side effect outside local files.

## What is not covered yet

- Deleting a source does not cascade into feedback/training queues, session
  transcripts that quote it, or an active adapter trained from that text.
- Replace-on-learn and directory-prune are still implicit deletions without a
  ticket.
- The ledger is per-process append-only JSONL, not a multi-writer database.
- Full lifecycle reconciliation and adapter revocation remain future slices.

Do not claim full erasure or a complete Memory Lifecycle Algebra from this
slice.
