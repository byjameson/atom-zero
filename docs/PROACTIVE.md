# Proactive conversation

Atom Zero can start a short conversation from local state instead of only
answering. The feature is **opt-in**, local and deterministic: it uses tasks,
knowledge gaps, recent sources and content-free claim-audit counts, never a
remote service, and never sends a message while consent is off.

## Policy

- `enabled` is false by default.
- At most one message per `min_interval_minutes` (default 180).
- At most `max_per_day` messages (default 3).
- Quiet hours default to 22:00–08:00 local time.
- The same reason is suppressed until the state changes or the user
  acknowledges it.
- A message is short and offers local command actions, for example
  `list tasks` or `brief me`.

## Claim signals

`POST /v1/claims/audit` and `POST /v1/consensus` accept an optional
`session_id`. When present, `note_claim_audit` stores only counts on the local
session: claim count, supported/weak/unsupported counts, single-source cluster
count, conflict count, method label and timestamp. It never stores the query,
claim text, evidence chunk, source label or answer.

If the last audit found unsupported claims, conflicts or single-source clusters,
the proactive layer can start a local check-in with the actions `show claims`
and `run consensus`. The normal consent, quiet-hours, cooldown, daily-limit and
duplicate-reason rules still apply.

`POST /v1/improvement/plan` with a `session_id` records only the plan shape:
action count, top action kind and whether every action requires human approval.
It never stores titles or details. A pending plan can start a check-in with the
actions `show improvements` and `run evaluation gate`.

## API

```powershell
POST /v1/proactive
{"session_id": "web-abc", "ack": false}

POST /v1/proactive/config
{"session_id": "web-abc", "enabled": true, "min_interval_minutes": 180}

POST /v1/proactive/ack
{"session_id": "web-abc", "proactive_id": "p-...", "action": "shown"}
```

`POST /v1/proactive` returns `due`, `reason`, `message`, `actions`,
`proactive_id` and `state`. The UI calls it on load, when the user enables the
toggle, and on a low-frequency timer. A due message is acknowledged after it is
shown so the same reason is not repeated.

## UI

The top bar has a `Proactive` toggle. When enabled, the page polls the local
endpoint and renders a bot message with action buttons. The toggle is a local
convenience; the server-side `enabled` flag is the authority.

## AISC

[`docs/aisc/atomzero-proactive.toml`](aisc/atomzero-proactive.toml) models the
path as `SignalCollector -> ConsentGate -> ProactivePolicy -> MessageComposer
-> UserChannel`. `ConsentGate` is a human component before any delivery, and
`UserChannel` is reversible. The spec is included in `atomzero-audit`.

## Limits

- This is not a background daemon. Messages are delivered only while the local
  UI is open and polling.
- It does not generate new facts; it routes to existing task/gap/brief data
  and to counts from the last claim audit.
- The claim signal is a count summary, not a truth verdict; it inherits the
  lexical claim audit's limits.
- It does not replace notifications, email or any external channel.
- Rate limits are per session and stored in the local session memory.
