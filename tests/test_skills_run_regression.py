from benchmarks.skills_run_regression import run


def test_skills_run_regression_passes(tmp_path):
    output = tmp_path / "skills-run.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 9
    assert output.exists()
