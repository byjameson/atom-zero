from atomzero.plan_audit import audit_plan, format_audit


def test_read_only_plan_is_allowed_and_low_risk():
    result = audit_plan([
        {"tool": "time"},
        {"tool": "calculator", "args": {"expression": "2 + 2"}},
    ])
    assert result["ok"] is True
    assert result["valid"] is True
    assert result["allowed"] is True
    assert result["risk"] == "low"
    assert result["requires_confirmation"] is False
    assert result["side_effects"] == []
    assert "ALLOWED" in format_audit(result)


def test_side_effect_plan_requires_confirmation():
    result = audit_plan([{"tool": "task_add", "args": {"text": "buy milk"}}])
    assert result["ok"] is True
    assert result["allowed"] is False
    assert result["requires_confirmation"] is True
    assert result["risk"] == "high"
    assert [item["tool"] for item in result["side_effects"]] == ["task_add"]
    codes = {issue["code"] for issue in result["issues"]}
    assert "confirmation_required" in codes
    assert "irreversible_unconfirmed" in codes


def test_confirmed_side_effect_plan_is_allowed_but_still_reported():
    result = audit_plan(
        [{"tool": "research", "args": {"query": "Ankara"}}], confirm=True)
    assert result["allowed"] is True
    assert result["requires_confirmation"] is False
    assert result["irreversible"]
    assert result["issues"] == []


def test_invalid_plan_is_blocked():
    result = audit_plan([{"tool": "shell", "args": {"cmd": "dir"}}])
    assert result["ok"] is False
    assert result["allowed"] is False
    assert result["risk"] == "blocked"
    assert result["issues"][0]["code"] == "invalid_plan"
    assert "BLOCKED" in format_audit(result)
