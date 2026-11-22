from benchmarks.brief_me_flow_regression import run


def test_brief_me_flow_regression_passes(tmp_path):
    output = tmp_path / "brief-me.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 7
    assert output.exists()
