"""Deterministic context-budget planning for local evidence.

AISC checks a system's declared ``max_context_tokens``; this module applies the
same idea at runtime. It selects the best retrieved hits that fit a character
budget and reports what was dropped. The estimate is approximate (four
characters per token) because Atom Zero does not assume a specific tokenizer
for every document.

    from atomzero.context_budget import plan_hits
    plan = plan_hits(hits, max_chars=4000, max_items=3)
"""
from __future__ import annotations

DEFAULT_MAX_CHARS = 6000
DEFAULT_MAX_ITEMS = 3


def estimate_tokens(text: str) -> int:
    return max(1, (len(str(text or "")) + 3) // 4)


def plan_hits(hits: list[dict], max_chars: int = DEFAULT_MAX_CHARS,
              max_items: int = DEFAULT_MAX_ITEMS) -> dict:
    """Return the highest-ranked hits that fit the budget.

    The first hit is always kept when one exists; a single oversized chunk is
    truncated by the caller's snippet logic rather than dropped, because an
    empty answer is worse than a bounded quote.
    """
    max_chars = max(1, int(max_chars))
    max_items = max(1, int(max_items))
    selected = []
    dropped = []
    used = 0
    for index, hit in enumerate(hits):
        text = str(hit.get("chunk", ""))
        cost = len(text)
        if len(selected) >= max_items:
            dropped.append({"index": index, "reason": "item_limit",
                            "chars": cost})
            continue
        if selected and used + cost > max_chars:
            dropped.append({"index": index, "reason": "char_budget",
                            "chars": cost})
            continue
        selected.append(hit)
        used += cost
    return {
        "hits": selected,
        "max_chars": max_chars,
        "max_items": max_items,
        "used_chars": used,
        "used_tokens_estimate": ((used + 3) // 4) if used else 0,
        "dropped": dropped,
        "within_budget": used <= max_chars,
    }


def format_budget(plan: dict) -> str:
    return (
        f"context: {plan['used_chars']}/{plan['max_chars']} chars "
        f"(~{plan['used_tokens_estimate']} tokens), "
        f"items={len(plan['hits'])}/{plan['max_items']}, "
        f"dropped={len(plan['dropped'])}")
