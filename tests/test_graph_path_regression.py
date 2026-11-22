from benchmarks.graph_path_regression import run


def test_graph_path_regression_passes(tmp_path):
    output = tmp_path / "graph-path.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 6
    assert output.exists()
