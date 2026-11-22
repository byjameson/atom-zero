from benchmarks.contradiction_regression import run


def test_contradiction_regression_passes(tmp_path):
    output = tmp_path / "contradiction.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 5
    assert output.exists()
