from benchmarks.weekly_export_regression import run


def test_weekly_export_regression_passes(tmp_path):
    output = tmp_path / "weekly-export.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 23
    assert output.exists()
