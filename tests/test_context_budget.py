from atomzero.citation_gate import CitationGate
from atomzero.context_budget import (
    estimate_tokens, format_budget, plan_hits)
from atomzero.doc_memory import grounded_answer


def _hit(index, size):
    return {"source": f"s{index}", "chunk": "x" * size}


def test_plan_hits_keeps_first_hit_even_if_oversized():
    plan = plan_hits([_hit(1, 100), _hit(2, 100)], max_chars=10)
    assert len(plan["hits"]) == 1
    assert plan["used_chars"] == 100
    assert plan["within_budget"] is False
    assert plan["dropped"][0]["reason"] == "char_budget"


def test_plan_hits_respects_item_limit():
    plan = plan_hits([_hit(1, 10), _hit(2, 10), _hit(3, 10)],
                     max_chars=1000, max_items=2)
    assert len(plan["hits"]) == 2
    assert plan["dropped"][0]["reason"] == "item_limit"


def test_plan_hits_reports_usage_and_format():
    plan = plan_hits([_hit(1, 100), _hit(2, 100)],
                     max_chars=250, max_items=3)
    assert len(plan["hits"]) == 2
    assert plan["used_chars"] == 200
    assert plan["used_tokens_estimate"] == 50
    assert "context:" in format_budget(plan)
    assert estimate_tokens("abcd") == 1


def test_grounded_answer_exposes_context_budget():
    hit = {
        "source": "note.txt", "title": "Note", "section": "",
        "chunk": "The launch code is L-7781.", "rerank_score": 0.9,
        "kind": "document", "verified": True,
    }
    result = grounded_answer(
        "what is the launch code?", [hit],
        gate=CitationGate(mode="off"), max_context_chars=5)
    assert result["refused"] is False
    assert result["context_budget"]["used_chars"] == len(hit["chunk"])
    assert result["context_budget"]["within_budget"] is False
