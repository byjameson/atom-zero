from benchmarks.skills_chain_regression import run


def test_skills_chain_regression_passes(tmp_path):
    output = tmp_path / "skills-chain.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 6
    assert output.exists()
