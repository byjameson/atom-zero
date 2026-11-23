# Improvement Loop

The improvement loop is the **planning layer** of a self-improving Atom Zero.
It turns weakness signals that already exist locally into a prioritized,
reviewable list of improvement actions. It is deliberately not an autonomous
weight updater: it does not train, mutate retrieval weights, rewrite documents,
promote feedback or execute any action on its own.

```
local signals -> prioritized plan -> human approval -> evaluation -> decision ledger
```

The implementation lives in `src/atomzero/improvement.py` and is described by
the `AtomZeroImprovementLoop` AISC spec in
`docs/aisc/atomzero-improvement.toml`.

## Why this layer exists

Atom Zero already produces several local quality summaries:

- `proactive.note_claim_audit` stores content-free claim-audit counts
  (`unsupported`, `conflict_count`, `single_source`);
- `knowledge_gaps` tracks questions that lack a local source;
- `feedback` tracks corrections that still need review;
- `freshness` reports stale and undated retrieval hits;
- `reliability` summarizes source audit outcomes.

Each summary is useful on its own, but a human needs one place to see which
weakness is most urgent and what "done" would mean. The improvement loop
provides that place without adding any hidden automation.

## The loop

### 1. Collect signals

`collect_signals(...)` accepts optional local inputs and returns deterministic
signals:

| Kind | Source | Meaning |
| --- | --- | --- |
| `unsupported_claims` | claim audit | claims without a supporting local source |
| `conflicts` | claim audit / consensus | conflicting claim pairs |
| `knowledge_gaps` | gap store | gaps whose status is not `done`, `dismissed` or `closed` |
| `single_source` | claim audit / consensus | claim clusters resting on one source |
| `stale_sources` | freshness summary | hits older than the freshness window |
| `undated_sources` | freshness summary | hits without a publication date |
| `weak_sources` | reliability summary | sources marked `weak` or seen fewer than two times |
| `pending_feedback` | feedback store | items not `approved` or `exported` |

Every signal has the shape:

```json
{
  "kind": "unsupported_claims",
  "priority": 80,
  "count": 3,
  "title": "Unsupported claims",
  "detail": "3 claim(s) lack a supporting local source.",
  "refs": ["claim_audit"]
}
```

Signals with a zero count are omitted. The list is sorted by priority
(descending), then by kind. A claim audit and a consensus summary can report
the same underlying counts; the larger view is used so one audit is not counted
twice. The knowledge-gap signal includes the top pending gap id and query as a
local reference only.

### 2. Build a plan

`build_plan(signals, limit=5)` sorts signals by priority, keeps at most `limit`
actions (1..20) and attaches a concrete acceptance gate to each one:

```json
{
  "action_id": "imp-01-unsupported_claims",
  "kind": "unsupported_claims",
  "title": "Unsupported claims",
  "detail": "3 claim(s) lack a supporting local source.",
  "priority": 80,
  "requires_human": true,
  "acceptance": "add a supporting source and rerun claim audit; run the relevant pytest/AISC regression before accepting"
}
```

The plan always reports:

```json
{
  "auto_apply": false,
  "weights_changed": false,
  "method": "deterministic_human_gated_improvement_v1"
}
```

`plan_from_inputs(**inputs, limit=5)` is a convenience wrapper that composes
signal collection and planning.

### 3. Require human approval

`requires_human` is always `true`. The plan is a proposal. A person may approve,
reject or defer each action with `record_decision(...)`. The planning layer
never decides that an action is safe to apply, and it never applies one.

A typical follow-up is:

1. a human approves an action;
2. the approved change (document ingest, source addition, feedback review,
   regression fix) is made outside this module;
3. the relevant pytest or AISC check is run;
4. the next audit/freshness/reliability summary is collected.

### 4. Evaluate before accepting

Acceptance strings are concrete gates, for example:

- add a supporting source and rerun claim audit;
- add an independent second source and rerun claim audit;
- refresh or replace stale sources and rerun freshness audit;
- add more observations or stronger sources and rerun reliability summary;
- run the relevant pytest/AISC regression before accepting.

An improvement is "accepted" only after its gate is satisfied. The improvement
loop does not mark an action accepted by itself.

### 5. Record decisions without content

`record_decision(ledger, action, decision, note="", now=None)` appends a
content-free record to a caller-owned list:

```json
{
  "version": 1,
  "at": "2026-01-01T12:00:00Z",
  "action_id": "imp-01-unsupported_claims",
  "kind": "unsupported_claims",
  "decision": "approve",
  "note_id": "6c7d8e0f1a2b3c4d"
}
```

`decision` must be `approve`, `reject` or `defer`. `note_id` is a 16-hex
SHA-256 prefix of the trimmed note (or `""` for an empty note). The note text is
never stored.

`verify_ledger(ledger)` returns `{"ok", "entries", "errors"}` and rejects
records that contain content keys (`text`, `query`, `answer`, `chunk`,
`content`, `note`, `session`, `source`), unknown keys, invalid decisions or
malformed fields.

## CLI

```powershell
$env:PYTHONPATH="$PWD\src"
python -m atomzero.improvement --claims-json claims.json --gaps-json gaps.json --json
python -m atomzero.improvement --freshness-json freshness.json --limit 3
python -m atomzero.improvement < payload.json
```

Flags:

- `--claims-json`: content-free claim audit (the dictionary written by
  `proactive.note_claim_audit`);
- `--gaps-json`: list of knowledge gaps;
- `--feedback-json`: list of feedback items;
- `--freshness-json`: freshness or consensus summary;
- `--reliability-json`: reliability summary;
- `--limit`: maximum number of actions (1..20);
- `--json`: machine-readable plan.

With no file flags, stdin must contain one JSON object with any of
`claim_audit` (alias `claims`), `gaps`, `feedback`, `freshness` or
`reliability`. Invalid JSON, an empty stdin payload or an invalid limit exits
with code 2. A valid payload exits 0, including a clean payload that produces
zero actions.

## Integration example

```python
from atomzero.improvement import plan_from_inputs, record_decision

plan = plan_from_inputs(
    claim_audit={"unsupported": 2, "conflict_count": 1,
                 "single_source": 3},
    gaps=[{"id": "g0001", "query": "local-only question",
           "status": "pending", "count": 4}],
    freshness={"status_counts": {"stale": 1, "undated": 2}},
    reliability={"sources": [
        {"source_id": "0123456789abcdef", "status": "weak",
         "observations": 1},
    ]},
    limit=3,
)

ledger = []
record_decision(ledger, plan["actions"][0], "defer",
                note="needs a primary source first")
```

## HTTP

`POST /v1/improvement/plan` accepts optional `session_id`, `gaps`,
`feedback`, `freshness`, `reliability` and `limit`. With a `session_id` it reads
only the content-free `claim_audit` counts stored on that local session.

`POST /v1/improvement/decision` accepts `{action, decision, note, gate}` and
appends a decision record to the local JSONL ledger at
`improvement_ledger_path` (default `data/improvement_ledger.jsonl`). An
`approve` decision is rejected unless `gate` is a passing evaluation-gate
report; reject/defer may carry a failed gate. The file stores action id, kind,
decision, timestamp, a 16-hex note hash and the gate counts/method only; the
note text and command output are never written. The existing ledger is verified
before and after the append.

```powershell
$plan = Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/improvement/plan `
  -ContentType "application/json" `
  -Body (@{session_id="web-abc"} | ConvertTo-Json)
$body = @{
  action = $plan.actions[0]
  decision = "approve"
  note = "reviewed locally"
  gate = @{
    ok = $true
    summary = @{ total = 2; passed = 2; failed = 0 }
    method = "allowlisted_evaluation_gate_v1"
  }
} | ConvertTo-Json -Depth 5
Invoke-RestMethod -Method Post `
  -Uri http://127.0.0.1:8000/v1/improvement/decision `
  -ContentType "application/json" -Body $body
```

## Guarantees

- No model call, no training and no weight update.
- No file is written except by the caller; the ledger is an in-memory list
  that the caller may persist deliberately.
- `auto_apply` and `weights_changed` are always `false`.
- Every action requires human approval and has an explicit acceptance gate.
- Every action also carries an allowlisted `next_step`; the gap action points
  at `POST /v1/learning/gaps/propose` and the research queue, and the coverage
  action points at `atomzero-coverage` plus `atomzero-gate`.
- Every action carries `suggested_checks` in `kind:target` form, ready to pass
  to `atomzero-gate --check` (for example `pytest:tests/test_claims.py` or
  `aisc:docs/aisc/atomzero-read.toml`).
- Signal and plan output is deterministic for identical inputs.
- Decision records are content-free and verifiable.

## Limitations

- **Lexical and count-based.** Claim and consensus signals inherit the limits
  of the upstream claim/consensus extractors. They are hints about local
  evidence, not truth verdicts. A "supported" claim can still be wrong.
- **Small samples stay small.** `weak_sources` counts sources whose status is
  `weak` or whose observation count is below two. The reliability score is a
  simple support ratio over audit outcomes; two observations do not establish
  statistical confidence.
- **Caller-maintained statuses.** Closed gaps and approved/exported feedback are
  only as accurate as the local store. A stale status can hide real work.
- **Fixed priorities.** Signal priorities are a documented local heuristic, not
  a learned policy. The limit can hide lower-priority signals, so the full
  signal list should be reviewed when needed.
- **The ledger is not a reasoning log.** It records only the action id, kind,
  decision and note hash. The human's reasoning must be stored separately if it
  is needed.
- **Evaluation is outside this module.** The loop defines gates but does not run
  retrieval benchmarks, training, document updates or any other follow-up. It
  is the planning layer, not the executor.
- **Top gap reference is local.** The knowledge-gap signal includes the top
  pending gap query as a local reference. Do not publish that payload if the
  question itself is sensitive.
