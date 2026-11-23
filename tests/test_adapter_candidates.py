import json
import sys
from pathlib import Path

import pytest

from atomzero.adapter_candidates import (
    build_candidate,
    evaluate_candidate,
    main,
    run_adapter_loop,
    summarize_adapter_ledger,
    write_candidate,
)


def _example(index, verified=True):
    return {
        "user": f"question {index}",
        "assistant": f"answer {index}",
        "verification": {
            "verified": verified,
            "promotable": verified,
            "status": "verified" if verified else "unverified",
        },
    }


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


def test_summarize_rejects_content_keys():
    with pytest.raises(ValueError):
        summarize_adapter_ledger([{"status": "pass", "text": "secret"}])
