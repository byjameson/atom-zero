# Context budget algebra

AISC checks a declared `max_context_tokens`; Atom Zero now applies a bounded
context plan at runtime as well. The planner is deterministic and local:

```python
from atomzero.context_budget import plan_hits, format_budget

plan = plan_hits(hits, max_chars=4000, max_items=3)
print(format_budget(plan))
```

`plan_hits` keeps the highest-ranked hits that fit the character budget, caps
the item count, and reports every dropped hit with a reason
(`item_limit` or `char_budget`). The first hit is always kept when one exists;
an oversized single chunk is bounded later by `best_snippet`, because returning
nothing is worse than returning a bounded quote.

## CLI

```powershell
atomzero-docs answer "what is the launch code?" --max-context-chars 4000
```

The answer payload then includes:

```json
{
  "context_budget": {
    "max_chars": 4000,
    "max_items": 3,
    "used_chars": 3271,
    "used_tokens_estimate": 818,
    "dropped": [{"index": 3, "reason": "char_budget", "chars": 901}],
    "within_budget": true
  }
}
```

The estimate is four characters per token. It is deliberately approximate: the
product does not assume one tokenizer for every document format. The budget is
a guardrail, not a model context-window promise.

## AISC spec

[`docs/aisc/atomzero-context-budget.toml`](aisc/atomzero-context-budget.toml)
models the path as `ContextPlanner -> AnswerBuilder -> LocalRenderer`. It has
no irreversible component and passes the non-strict AISC architecture check.
Strict calibration evidence is reported separately by `atomzero-audit
--strict --require-calibration`.

## Limits

- The budget is character-based; token counts are approximate.
- It does not change retrieval ranking; it only selects from the ranked hits.
- It does not replace the citation gate or the answerability experiment.
- A first oversized hit can put the plan over budget; the flag
  `within_budget` makes that visible instead of hiding it.
