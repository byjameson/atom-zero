from benchmarks.skills_flow_regression import run


def test_skills_flow_regression_passes(tmp_path):
    output = tmp_path / "skills-flow.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 9
    assert output.exists()
