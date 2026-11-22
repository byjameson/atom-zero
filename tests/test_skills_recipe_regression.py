from benchmarks.skills_recipe_regression import run


def test_skills_recipe_regression_passes(tmp_path):
    output = tmp_path / "skills-recipe.json"
    payload = run(output)
    assert payload["ok"] is True, payload["checks"]
    assert payload["failed"] == 0
    assert payload["passed"] >= 6
    assert output.exists()
