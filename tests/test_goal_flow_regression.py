from benchmarks.goal_flow_regression import run


def test_goal_flow_regression_passes(tmp_path):
    output = tmp_path / "goal-flow.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 6
    assert output.exists()
