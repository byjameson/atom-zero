# Escalation ladder

**Status:** opt-in, deterministic router. The router is the first step of the
snowball ladder, not a trainer.

Atom Zero tries the local model first. When local confidence is insufficient,
a caller may follow a deterministic route to a configured teacher or to an
opt-in agent. The router only *decides*; it never calls a network, never starts
an agent, never changes a weight and never applies anything.

```text
task metadata -> deterministic router -> local / teacher / agent / refuse
                                      -> content-free growth ledger
                                      -> human-reviewed next step
```

## The honest ladder

The "snowball" is a sequence of increasingly expensive capabilities. Each step
is separate, explicit and reviewable. Nothing rolls forward automatically.

| Stage | What grows | What actually happens | Automatic? |
|---|---|---|---|
| 1. Retrieval memory | local document index | teacher notes are stored as unverified notes; a human or an existing verification flow may accept them | no |
| 2. Skills / recipes | reviewed procedural memory | a successful procedure is written down and reviewed like any other local source | no |
| 3. Distilled adapters | an adapter candidate set | verified teacher events become *candidates*; a separate external trainer and held-out comparison must be run | no |
| 4. Offline checkpoint upcycling | a new model checkpoint | a training run produces a new checkpoint; it must beat the base on a held-out set and pass the release gates | no |
| 5. Agentic code improvement | the repository | an opt-in external agent works in an isolated worktree; the change passes a human gate before acceptance | no |

The router does not move data between these stages. It only produces a
decision and a content-free event. In particular:

- retrieval memory is not a weight update;
- an adapter candidate is not an adapter, and an adapter is not applied by the
  router;
- an offline checkpoint is a new artifact that a human must evaluate, choose
  and ship;
- an agent handoff is a request for a human-gated handoff, not an agent run.

**Parameter count does not grow automatically.** The base model keeps the same
weights and parameter count unless a human runs an offline, reproducible
training step, evaluates the result and explicitly promotes the new artifact.

## Routing rules

Rules are evaluated in this order. The first match wins.

| # | Condition | Route | Reason |
|---|---|---|---|
| 1 | `privacy == "secret"` | `refuse` | `privacy_secret` |
| 2 | `kind == "code"` and `agent_enabled` and agent budget left | `agent` | `bounded_code_handoff` |
| 3 | `confidence >= min_local_confidence` and not `requires_freshness` | `local` | `local_confident` |
| 4 | teacher enabled, external access allowed, external budget left, `privacy == "public"` | `teacher` | `low_confidence_or_freshness` |
| 5 | otherwise | `refuse` | `no_available_route`, or `privacy_private` when only privacy blocked the teacher |

Route 2 deliberately comes before the confident-local route: a policy that
enables the agent has asked for bounded code handoffs, and that handoff always
carries `requires_human: true`. A secret task is still refused first.

`private` tasks may not go to the teacher. They can still be routed to an
opt-in agent when the policy enables agent handoffs, because that route is
human-gated; keep the agent's sandbox and data access under your own control.
`secret` tasks never leave the router.

## Decision output

`decide(task, *, policy, history=None)` returns only routing metadata:

```json
{
  "route": "teacher",
  "reason_code": "low_confidence_or_freshness",
  "confidence_bucket": "low",
  "requires_human": false,
  "external_call": false,
  "auto_apply": false,
  "weights_changed": false,
  "budget": {
    "external_calls": 0,
    "external_limit": 2,
    "external_remaining": 2,
    "agent_calls": 0,
    "agent_limit": 0,
    "agent_remaining": 0
  },
  "method": "deterministic_escalation_router_v1"
}
```

`external_call` is always false: the router does not perform the teacher call.
`auto_apply` and `weights_changed` are always false.

The output never contains task text, a prompt, a query, an answer, a source or
a session identifier. `confidence_bucket` is `low` below 0.50, `medium` below
0.80 and `high` otherwise. `complexity` is accepted and validated for forward
compatibility, but v1 does not use it to change the route.

## Task input

```json
{
  "kind": "answer",
  "confidence": 0.31,
  "privacy": "public",
  "requires_freshness": false,
  "complexity": "medium"
}
```

| field | values | required |
|---|---|---|
| `kind` | `answer`, `code`, `research` | yes |
| `confidence` | number from 0 to 1 | yes |
| `privacy` | `public`, `private`, `secret` | yes |
| `requires_freshness` | boolean | no, default false |
| `complexity` | `low`, `medium`, `high` | no |

Unknown keys and content-bearing keys are rejected with `ValueError`.

## Policy and budgets

```json
{
  "teacher_enabled": false,
  "agent_enabled": false,
  "allow_external": false,
  "min_local_confidence": 0.55,
  "max_external_calls": 0,
  "max_agent_calls": 0
}
```

All fields are optional. The defaults above are deliberately closed: no
teacher, no agent and no external access.

`history` is an optional list of prior content-free escalation records. The
router counts records whose `route` is `teacher` against
`max_external_calls`, and records whose `route` is `agent` against
`max_agent_calls`. Failed attempts count too; otherwise a failing provider
could be retried without bound. Local and refuse records do not consume a
budget.

Budgets are local and advisory. The caller must record events after execution
for them to affect later decisions.

## Growth ledger

```python
from atomzero.escalation import decide, record_escalation

ledger = []
decision = decide(task, policy=policy, history=ledger)
if decision["route"] == "teacher":
    answer = teacher.ask(prompt)              # caller's explicit action
    record_escalation(ledger, decision, outcome="success", verified=True)
```

`record_escalation` appends a content-free event:

```json
{
  "version": 1,
  "at": "2026-11-24T12:00:00Z",
  "route": "teacher",
  "reason_code": "low_confidence_or_freshness",
  "outcome": "success",
  "verified": true,
  "growth_event": "verified_teacher",
  "method": "deterministic_escalation_growth_v1"
}
```

Only `route == "teacher"`, `outcome == "success"` and `verified is True`
produces the `verified_teacher` growth event. A verified local, agent or
refusal event is still only a normal event. Verification is a human decision;
the router cannot verify an answer.

`summarize_growth(ledger)` reports:

- `routes`: counts for `local`, `teacher`, `agent` and `refuse`;
- `verified_teacher_answers`: verified, successful teacher events;
- `adapter_candidates`: the same verified teacher events, as a *candidate*
  count only;
- `external_calls`: teacher-route events;
- `agent_handoffs`: agent-route events;
- `refusals`: refuse-route events;
- `parameter_growth: 0` and `parameter_growth_state: "unchanged"`;
- `weights_changed: false` and `auto_apply: false`.

Ledgers reject content-bearing keys (`text`, `query`, `answer`, `prompt`,
`content`, `source`, `note`, `session`) and malformed events. The ledger is
local and unsigned; validation checks shape and content-free rules, not
authorship.

## CLI

```powershell
atomzero-escalate --task-json task.json --policy-json policy.json --json
atomzero-escalate --task-json task.json --history-json history.json
atomzero-escalate --task-json - --json
atomzero-escalate --summary-json growth_ledger.json
```

| flag | meaning |
|---|---|
| `--task-json PATH` | task metadata; `-` reads stdin |
| `--policy-json PATH` | optional policy; `-` reads stdin |
| `--history-json PATH` | optional prior records; `-` reads stdin |
| `--summary-json PATH` | print `summarize_growth` for a ledger; `-` reads stdin |
| `--json` | print the decision as JSON |

Only one input may use stdin. Exit code `0` means a valid decision or summary;
`2` means invalid JSON, a missing file, a schema error or conflicting modes.
The CLI never constructs a teacher provider, never invokes an agent and never
touches weights.

The summary mode always prints JSON.

## Relationship to existing components

| Component | Role in the ladder |
|---|---|
| `hungry.py` | grows retrieval memory with bounded teacher questions |
| `teacher_provider.py` | explicit OpenAI-compatible teacher call, never automatic |
| `teacher_notes.py` | stores teacher answers as unverified notes |
| `improvement.py` | human-gated planning and decision ledger |
| `escalation.py` | content-free routing and growth-event accounting |
| `agent_bridge.py` / `agent_runner.py` | external, opt-in agent execution with scope and gate checks |
| `docs/ADAPTERS.md` | external adapter training and local comparison |

A typical reviewed path is:

```text
escalation decision -> caller asks teacher -> answer stored unverified
-> human verifies -> verified_teacher event -> adapter candidate review
-> external trainer -> held-out comparison -> explicit human promotion
```

## Limits

- `confidence` comes from the caller or the local model; it is not a calibrated
  probability in this module.
- The router cannot verify a teacher answer, detect a wrong answer or prove
  that freshness is satisfied. `requires_freshness` is a caller flag.
- Budgets only know what the caller recorded. An unrecorded teacher call is
  invisible to the router.
- `complexity` does not change routing in v1.
- An agent route is a human-gated request, not an execution. A separate
  opt-in runner must enforce worktree scope, checks and the evaluation gate.
- A verified teacher event is an adapter candidate, not training data that is
  automatically safe, licensed or correct. Review before use.
- The ledger is a local audit aid, not a security boundary or a signature.
- Nothing in this module increases parameter count, changes weights or applies
  a change.
