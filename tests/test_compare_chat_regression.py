from benchmarks.compare_chat_regression import run


def test_compare_chat_regression_passes(tmp_path):
    output = tmp_path / "compare-chat.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 6
    assert output.exists()
