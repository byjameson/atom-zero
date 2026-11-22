from benchmarks.weekly_review_regression import run


def test_weekly_review_regression_passes(tmp_path):
    output = tmp_path / "weekly-review.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 6
    assert output.exists()
