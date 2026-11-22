from benchmarks.scoped_regression import run


def test_scoped_regression_passes(tmp_path):
    output = tmp_path / "result.json"
    payload = run(output)
    assert payload["ok"] is True
    assert payload["failed"] == 0
    assert payload["passed"] >= 10
    assert output.exists()
    assert {item["name"] for item in payload["checks"]} >= {
        "source_scope_answer", "prefix_scope_answer", "kind_scope_answer",
        "collection_summary", "exact_code_ranking"}
