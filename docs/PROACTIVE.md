# Proactive conversation

Atom Zero can start a short conversation from local state instead of only
answering. The feature is **opt-in**, local and deterministic: it uses tasks,
knowledge gaps and recent sources, never a remote service, and never sends a
message while consent is off.

## Policy

- `enabled` is false by default.
- At most one message per `min_interval_minutes` (default 180).
- At most `max_per_day` messages (default 3).
- Quiet hours default to 22:00–08:00 local time.
- The same reason is suppressed until the state changes or the user
  acknowledges it.
- A message is short and offers local command actions, for example
  `list tasks` or `brief me`.

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
- It does not generate new facts; it routes to existing task/gap/brief data.
- It does not replace notifications, email or any external channel.
- Rate limits are per session and stored in the local session memory.
