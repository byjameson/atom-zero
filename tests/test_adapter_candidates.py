import json
import sys
from pathlib import Path

import pytest

from atomzero.adapter_candidates import (
    build_candidate,
    build_cross_candidate,
    evaluate_candidate,
    family_summary,
    main,
    promote_candidate,
    run_adapter_loop,
    summarize_adapter_ledger,
    write_candidate,
)


def _example(index, verified=True, family=None):
    row = {
        "user": f"question {index}",
        "assistant": f"answer {index}",
        "verification": {
            "verified": verified,
            "promotable": verified,
            "status": "verified" if verified else "unverified",
        },
    }
    if family is not None:
        row["family"] = family
    return row


def _candidate(eval_ratio=0.25, count=4):
    return build_candidate(
        [_example(index) for index in range(count)], eval_ratio=eval_ratio)


def _comparison(examples=1, base_loss=1.2, adapter_loss=1.1,
                base_ppl=3.3, adapter_ppl=3.1):
    return {
        "examples": examples,
        "baseline": {"loss": base_loss, "ppl": base_ppl},
        "adapter": {"loss": adapter_loss, "ppl": adapter_ppl},
    }


def test_build_candidate_is_deterministic_and_splits():
    first = _candidate()
    second = _candidate()
    assert first["candidate_id"] == second["candidate_id"]
    assert first["metadata"]["total_examples"] == 4
    assert first["metadata"]["train_examples"] == 3
    assert first["metadata"]["eval_examples"] == 1
    assert len(first["train"]) == 3
    assert len(first["eval"]) == 1
    assert first["metadata"]["weights_changed"] is False


def test_build_candidate_skips_duplicates():
    examples = [_example(0), _example(0), _example(1)]
    candidate = build_candidate(examples, eval_ratio=0.5)
    assert candidate["metadata"]["duplicates_skipped"] == 1
    assert candidate["metadata"]["total_examples"] == 2


def test_build_candidate_rejects_unverified():
    with pytest.raises(ValueError):
        build_candidate([_example(0, verified=False)])
    with pytest.raises(ValueError):
        build_candidate([{"user": "q", "assistant": "a"}])


def test_cross_candidate_keeps_distributions_separate(tmp_path):
    train = [_example(index) for index in range(4)]
    eval_rows = [{"user": f"cross {index}", "assistant": f"answer {index}",
                  "verification": {"verified": True, "promotable": True}}
                 for index in range(2)]
    candidate = build_cross_candidate(train, eval_rows, seed=13)
    assert candidate["metadata"]["split_method"] == (
        "explicit_cross_distribution")
    assert candidate["metadata"]["train_examples"] == 4
    assert candidate["metadata"]["eval_examples"] == 2
    assert candidate["metadata"]["candidate_id"].startswith("ad-")
    with pytest.raises(ValueError):
        build_cross_candidate(train, [train[0]])


def test_family_summary_counts_labels_and_defaults():
    rows = [
        _example(0, family="person_json"),
        _example(1, family="person_json"),
        _example(2, family="product_kv"),
        _example(3),
    ]
    assert family_summary(rows) == {
        "person_json": 2, "product_kv": 1, "unspecified": 1}
    with pytest.raises(ValueError):
        family_summary([_example(4, family="bad\nfamily")])
    with pytest.raises(ValueError):
        family_summary([_example(5, family=13)])
    with pytest.raises(ValueError):
        family_summary([_example(6, family="   ")])


def test_cross_candidate_reports_disjoint_families():
    train = [_example(f"train-{index}", family="person_json")
             for index in range(3)]
    eval_rows = [_example(f"eval-{index}", family="product_kv")
                 for index in range(2)]
    candidate = build_cross_candidate(train, eval_rows)
    metadata = candidate["metadata"]
    assert metadata["train_families"] == ["person_json"]
    assert metadata["eval_families"] == ["product_kv"]
    assert metadata["family_separation"] is True
    assert metadata["train_family_counts"] == {"person_json": 3}
    assert metadata["eval_family_counts"] == {"product_kv": 2}
    assert candidate["train"]
    assert candidate["eval"]


def test_cross_candidate_rejects_family_leakage():
    train = [
        _example("train-0", family="city_json"),
        _example("train-1", family="person_json"),
    ]
    eval_rows = [
        _example("eval-0", family="product_kv"),
        _example("eval-1", family="city_json"),
    ]
    with pytest.raises(ValueError, match="task-family leakage"):
        build_cross_candidate(train, eval_rows)


def test_cross_candidate_without_families_still_works():
    train = [_example(f"train-{index}") for index in range(2)]
    eval_rows = [_example(f"eval-{index}") for index in range(2)]
    candidate = build_cross_candidate(train, eval_rows)
    metadata = candidate["metadata"]
    assert metadata["train_families"] == ["unspecified"]
    assert metadata["eval_families"] == ["unspecified"]
    assert metadata["family_separation"] is False
    assert metadata["train_family_counts"] == {"unspecified": 2}
    assert metadata["eval_family_counts"] == {"unspecified": 2}

    mixed = build_cross_candidate(
        [_example("mixed-train")],
        [_example("mixed-eval", family="city_json")])
    assert mixed["metadata"]["eval_families"] == ["city_json"]
    assert mixed["metadata"]["family_separation"] is False


def test_build_candidate_never_claims_family_separation():
    examples = [
        _example(index, family="person_json" if index % 2 else "city_json")
        for index in range(4)
    ]
    metadata = build_candidate(examples, eval_ratio=0.5)["metadata"]
    assert metadata["family_separation"] == "not_applicable"
    assert set(metadata["train_families"]) <= {"person_json", "city_json"}
    assert sum(metadata["train_family_counts"].values()) == (
        metadata["train_examples"])
    assert sum(metadata["eval_family_counts"].values()) == (
        metadata["eval_examples"])


def test_family_labels_are_validated_and_stripped():
    candidate = build_candidate(
        [_example(0, family="  person_json  ")], eval_ratio=0.0)
    assert candidate["train"][0]["family"] == "person_json"
    assert candidate["metadata"]["train_families"] == ["person_json"]
    with pytest.raises(ValueError):
        build_candidate([_example(1, family="bad\x7ffamily")],
                        eval_ratio=0.0)
    with pytest.raises(ValueError):
        build_candidate([_example(2, family="x" * 65)], eval_ratio=0.0)


def test_family_metadata_has_no_prompt_or_answer_text():
    train = [_example("secret-train", family="person_json")]
    eval_rows = [_example("secret-eval", family="product_kv")]
    candidate = build_cross_candidate(train, eval_rows)
    serialized = json.dumps(candidate["metadata"])
    assert "secret-train" not in serialized
    assert "secret-eval" not in serialized
    assert "question" not in serialized
    assert "answer" not in serialized
    assert not ({"user", "assistant"} & set(candidate["metadata"]))


def test_write_candidate_keeps_text_out_of_metadata(tmp_path):
    candidate = _candidate()
    paths = write_candidate(candidate, tmp_path)
    metadata = json.loads(
        (tmp_path / "candidate.json").read_text(encoding="utf-8"))
    assert "question 0" not in json.dumps(metadata)
    assert "answer 0" not in json.dumps(metadata)
    train_rows = (tmp_path / "train.jsonl").read_text(
        encoding="utf-8").splitlines()
    eval_rows = (tmp_path / "eval.jsonl").read_text(
        encoding="utf-8").splitlines()
    assert len(train_rows) == 3
    assert len(eval_rows) == 1
    assert paths["candidate_json"].endswith("candidate.json")


def test_held_out_gate_pass_and_fail():
    candidate = _candidate()
    passed = evaluate_candidate(candidate, _comparison())
    assert passed["status"] == "pass"
    assert passed["ok"] is True
    assert passed["promotable"] is True
    assert passed["loss_delta"] == pytest.approx(-0.1)
    assert passed["auto_apply"] is False
    assert passed["weights_changed"] is False

    loss_fail = evaluate_candidate(
        candidate, _comparison(adapter_loss=1.2))
    assert loss_fail["status"] == "fail"
    assert "loss_not_improved" in loss_fail["reasons"]

    ppl_fail = evaluate_candidate(
        candidate, _comparison(adapter_ppl=3.5))
    assert ppl_fail["status"] == "fail"
    assert "ppl_regression" in ppl_fail["reasons"]

    mismatch = evaluate_candidate(candidate, _comparison(examples=2))
    assert mismatch["status"] == "fail"
    assert any(item.startswith("eval_count_mismatch")
               for item in mismatch["reasons"])


def test_invalid_comparison_is_reported_not_raised():
    result = evaluate_candidate(_candidate(), {"models": []})
    assert result["status"] == "invalid"
    assert result["ok"] is False
    assert result["auto_apply"] is False


def test_run_adapter_loop_writes_content_free_ledger(tmp_path):
    ledger = tmp_path / "adapter_ledger.jsonl"
    result = run_adapter_loop(
        [_example(index) for index in range(4)],
        tmp_path / "candidate",
        comparison=_comparison(),
        eval_ratio=0.25,
        ledger_path=ledger)
    assert result["status"] == "pass"
    assert result["evaluation"]["promotable"] is True
    assert ledger.is_file()
    event = json.loads(ledger.read_text(encoding="utf-8").strip())
    assert event["candidate_id"] == result["candidate"]["candidate_id"]
    assert event["promotable"] is True
    assert not ({"user", "assistant", "text", "content"} & set(event))
    summary = summarize_adapter_ledger([event])
    assert summary["pass"] == 1
    assert summary["promotable"] == 1
    assert summary["parameter_growth"] == 0
    assert summary["weights_changed"] is False


def test_evaluator_command_runs_in_sandbox(tmp_path):
    script = tmp_path / "evaluator.py"
    script.write_text(
        "import json\n"
        "print(json.dumps({'examples': 1, "
        "'baseline': {'loss': 1.2, 'ppl': 3.3}, "
        "'adapter': {'loss': 1.0, 'ppl': 3.0}}))\n",
        encoding="utf-8")
    result = run_adapter_loop(
        [_example(index) for index in range(4)],
        tmp_path / "candidate",
        evaluator_command=[sys.executable, str(script)],
        eval_ratio=0.25,
        timeout=60)
    assert result["status"] == "pass"
    assert result["source"] == "evaluator"
    assert result["evaluation"]["loss_delta"] == pytest.approx(-0.2)


def test_trainer_command_can_feed_the_gate(tmp_path):
    script = tmp_path / "trainer.py"
    script.write_text(
        "import json, pathlib, sys\n"
        "comparison = pathlib.Path(sys.argv[1])\n"
        "adapter = pathlib.Path(sys.argv[2])\n"
        "adapter.mkdir(parents=True, exist_ok=True)\n"
        "(adapter / 'adapter.pt').write_bytes(b'fake')\n"
        "comparison.write_text(json.dumps({'examples': 1, "
        "'baseline': {'loss': 1.2, 'ppl': 3.3}, "
        "'adapter': {'loss': 1.0, 'ppl': 3.0}}))\n",
        encoding="utf-8")
    result = run_adapter_loop(
        [_example(index) for index in range(4)],
        tmp_path / "candidate",
        base="base-fixture",
        trainer_command=[sys.executable, str(script),
                         "{comparison_path}", "{adapter_dir}"],
        eval_ratio=0.25,
        timeout=60,
        trainer_timeout=60)
    assert result["status"] == "pass"
    assert result["source"] == "trainer"
    assert result["trainer"]["ok"] is True
    assert result["evaluation"]["promotable"] is True
    assert Path(result["paths"]["comparison_json"]).is_file()
    assert Path(result["paths"]["adapter_dir"]).is_dir()


def test_run_adapter_loop_with_explicit_splits(tmp_path):
    train = [_example(index) for index in range(4)]
    eval_rows = [{"user": f"cross {index}", "assistant": f"answer {index}",
                  "verification": {"verified": True, "promotable": True}}
                 for index in range(2)]
    result = run_adapter_loop(
        [], tmp_path / "candidate", train_examples=train,
        eval_examples=eval_rows,
        comparison={"examples": 2,
                    "baseline": {"loss": 1.2, "ppl": 3.3},
                    "adapter": {"loss": 1.0, "ppl": 3.0}})
    assert result["status"] == "pass"
    assert result["candidate"]["split_method"] == (
        "explicit_cross_distribution")
    assert result["candidate"]["train_examples"] == 4
    assert result["candidate"]["eval_examples"] == 2


def test_content_gate_requires_content_metrics():
    candidate = _candidate()
    comparison = _comparison()
    missing = evaluate_candidate(
        candidate, comparison, require_content=True)
    assert missing["status"] == "invalid" or missing["ok"] is False
    assert "content_metrics_missing" in missing["reasons"]
    comparison["baseline"]["content_loss"] = 2.0
    comparison["baseline"]["content_ppl"] = 7.4
    comparison["adapter"]["content_loss"] = 1.5
    comparison["adapter"]["content_ppl"] = 4.5
    passed = evaluate_candidate(
        candidate, comparison, require_content=True)
    assert passed["status"] == "pass"
    assert "content_loss_improved" in passed["reasons"]
    comparison["adapter"]["content_loss"] = 2.5
    failed = evaluate_candidate(
        candidate, comparison, require_content=True)
    assert failed["status"] == "fail"
    assert "content_loss_not_improved" in failed["reasons"]


def test_exact_match_gate_is_opt_in_and_thresholded():
    candidate = _candidate()
    comparison = _comparison()
    comparison["baseline"]["exact_match"] = {"rate": 0.2}
    comparison["adapter"]["exact_match"] = {"rate": 0.75}
    default = evaluate_candidate(candidate, comparison)
    assert default["status"] == "pass"
    assert "exact_match_ok" not in default["reasons"]
    assert default["adapter_exact_match"] == pytest.approx(0.75)

    passed = evaluate_candidate(
        candidate, comparison, min_exact_match=0.5)
    assert passed["status"] == "pass"
    assert "exact_match_ok" in passed["reasons"]
    assert passed["baseline_exact_match"] == pytest.approx(0.2)
    assert passed["adapter_exact_match"] == pytest.approx(0.75)
    assert passed["exact_match_delta"] == pytest.approx(0.55)
    assert passed["min_exact_match"] == pytest.approx(0.5)

    comparison["adapter"]["exact_match"] = 0.6
    low = evaluate_candidate(candidate, comparison, min_exact_match=0.9)
    assert low["status"] == "fail"
    assert low["ok"] is False
    assert "exact_match_low" in low["reasons"]

    missing = evaluate_candidate(
        _candidate(), _comparison(), min_exact_match=0.5)
    assert missing["status"] == "fail"
    assert "exact_match_missing" in missing["reasons"]


def test_min_exact_match_must_be_a_rate():
    with pytest.raises(ValueError):
        evaluate_candidate(_candidate(), _comparison(),
                           min_exact_match=1.5)
    with pytest.raises(ValueError):
        evaluate_candidate(_candidate(), _comparison(),
                           min_exact_match="high")


def test_cli_evaluate_wires_min_exact_match(tmp_path, capsys):
    comparison = _comparison()
    comparison["baseline"]["exact_match"] = 0.2
    comparison["adapter"]["exact_match"] = 0.5
    candidate_path = tmp_path / "candidate.json"
    comparison_path = tmp_path / "comparison.json"
    candidate_path.write_text(
        json.dumps(_candidate()["metadata"]), encoding="utf-8")
    comparison_path.write_text(json.dumps(comparison), encoding="utf-8")
    code = main([
        "evaluate",
        "--candidate-json", str(candidate_path),
        "--comparison-json", str(comparison_path),
        "--min-exact-match", "0.8",
        "--json",
    ])
    assert code == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["min_exact_match"] == pytest.approx(0.8)
    assert payload["adapter_exact_match"] == pytest.approx(0.5)
    assert "exact_match_low" in payload["reasons"]


def test_promotion_requires_passing_gate(tmp_path):
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter.pt").write_bytes(b"fake")
    candidate = {"candidate_id": "ad-1", "family_separation": True}
    pass_gate = {"status": "pass", "ok": True, "promotable": True,
                 "candidate_id": "ad-1",
                 "reasons": ["content_loss_improved",
                             "content_ppl_not_worse"]}
    record = promote_candidate(
        candidate, pass_gate, adapter, tmp_path / "active.json")
    assert record["candidate_id"] == "ad-1"
    assert record["promotable"] is True
    assert record["weights_changed"] is False
    assert (tmp_path / "active.json").is_file()
    with pytest.raises(ValueError):
        promote_candidate(
            candidate, {"status": "fail", "ok": False, "promotable": False},
            adapter, tmp_path / "bad.json")
    with pytest.raises(ValueError):
        promote_candidate(
            {"candidate_id": "ad-2"}, pass_gate, adapter,
            tmp_path / "mismatch.json")


def test_cli_export(tmp_path, capsys):
    examples = tmp_path / "examples.jsonl"
    examples.write_text(
        "".join(json.dumps(_example(index)) + "\n" for index in range(4)),
        encoding="utf-8")
    out = tmp_path / "out"
    code = main([
        "export", "--examples-jsonl", str(examples),
        "--out-dir", str(out), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["eval_examples"] == 1
    assert (out / "candidate.json").is_file()
    assert (out / "train.jsonl").is_file()
    assert (out / "eval.jsonl").is_file()


def test_cli_export_cross_reports_family_counts(tmp_path, capsys):
    train_path = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    train_path.write_text(
        "".join(json.dumps(_example(f"train-{index}",
                                    family="person_json")) + "\n"
                for index in range(2)), encoding="utf-8")
    eval_path.write_text(
        "".join(json.dumps(_example(f"eval-{index}",
                                    family="city_json")) + "\n"
                for index in range(2)), encoding="utf-8")
    out = tmp_path / "out"
    code = main([
        "export-cross", "--train-jsonl", str(train_path),
        "--eval-jsonl", str(eval_path), "--out-dir", str(out), "--json"])
    assert code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate"]["family_separation"] is True
    assert payload["candidate"]["train_families"] == ["person_json"]
    assert payload["candidate"]["eval_families"] == ["city_json"]
    assert payload["family_counts"] == {
        "train": {"person_json": 2}, "eval": {"city_json": 2}}
    assert payload["auto_apply"] is False
    assert payload["weights_changed"] is False
    metadata_text = (out / "candidate.json").read_text(encoding="utf-8")
    assert "question" not in metadata_text
    assert "answer" not in metadata_text
    assert (out / "train.jsonl").is_file()
    assert (out / "eval.jsonl").is_file()


def test_summarize_rejects_content_keys():
    with pytest.raises(ValueError):
        summarize_adapter_ledger([{"status": "pass", "text": "secret"}])
