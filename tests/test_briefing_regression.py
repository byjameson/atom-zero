from benchmarks.briefing_regression import run


def test_briefing_regression_passes(tmp_path):
    output = tmp_path / "briefing.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 9
    assert output.exists()
