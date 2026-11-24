import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.run_weak_stage_rehearsal import (
    RehearsalError,
    allocate_counts,
    build_ledger,
    build_rehearsal_mix,
    build_trainer_command,
    comparison_stage_metrics,
    discover_stages,
    evaluate_stages,
    load_stage_metrics,
    mix_weights,
    normalize_stage_metrics,
    rank_stages,
    read_jsonl,
    run_rehearsal,
    run_trainer,
    sample_rehearsal_rows,
    stage_groups,
)


BEFORE = {
    "echo_kv": {"exact": 0.62, "key_copy": 0.74},
    "echo_json": {"exact": 0.76, "key_copy": 0.80},
    "two_kv": {"exact": 0.44, "key_copy": 0.54},
    "two_json": {"exact": 0.46, "key_copy": 0.72},
    "multi": {"exact": 0.12, "key_copy": 0.22},
}

AFTER = {
    "echo_kv": {"exact": 0.70, "key_copy": 0.78},
    "echo_json": {"exact": 0.74, "key_copy": 0.78},
    "two_kv": {"exact": 0.50, "key_copy": 0.60},
    "two_json": {"exact": 0.50, "key_copy": 0.76},
    "multi": {"exact": 0.30, "key_copy": 0.40},
}


def _stage_rows(per_stage=20):
    return {
        stage: [
            {"id": f"{stage}-{index}", "user": "TOP SECRET PROMPT",
             "assistant": "TOP SECRET ANSWER"}
            for index in range(per_stage)
        ]
        for stage in BEFORE
    }


def _stage_data(tmp_path, rows=8):
    root = tmp_path / "stages"
    for stage in BEFORE:
        directory = root / f"stage_{stage}"
        directory.mkdir(parents=True)
        payload = [{"user": "q", "assistant": "a", "id": index}
                   for index in range(rows)]
        (directory / "train.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in payload),
            encoding="utf-8")
        (directory / "eval.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in payload),
            encoding="utf-8")
    return root


def test_normalize_stage_metrics_accepts_per_stage_pairs(tmp_path):
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({
        "date": "2026-11-24",
        "adapter": "multi_recover",
        "per_stage": {
            "echo_kv": [0.62, 0.74],
            "multi": [0.12, 0.22],
        },
    }), encoding="utf-8")
    metrics = load_stage_metrics(path)
    assert list(metrics) == ["echo_kv", "multi"]
    assert metrics["echo_kv"] == {"exact": 0.62, "key_copy": 0.74}


def test_normalize_stage_metrics_accepts_stage_objects_and_aliases():
    metrics = normalize_stage_metrics({
        "stages": {
            "two_json": {"exact_match_rate": 0.46,
                         "key_copy_rate": 0.72},
            "echo_json": {"metrics": {"exact": 0.76, "key_copy": 0.80}},
        },
    })
    assert list(metrics) == ["echo_json", "two_json"]
    assert metrics["two_json"] == {"exact": 0.46, "key_copy": 0.72}


def test_normalize_stage_metrics_reads_a_previous_rehearsal_ledger():
    metrics = normalize_stage_metrics({
        "stages": [
            {"stage": "echo_kv",
             "before": {"exact": 0.60, "key_copy": 0.70},
             "after": {"exact": 0.65, "key_copy": 0.75}},
            {"stage": "multi",
             "before": {"exact": 0.12, "key_copy": 0.22},
             "after": None},
        ],
    })
    assert metrics["echo_kv"] == {"exact": 0.65, "key_copy": 0.75}
    assert metrics["multi"] == {"exact": 0.12, "key_copy": 0.22}


def test_normalize_stage_metrics_rejects_bad_rates_and_shapes():
    with pytest.raises(RehearsalError, match=r"\[0, 1\]"):
        normalize_stage_metrics({"stages": {
            "echo_kv": {"exact": 62.0, "key_copy": 0.74}}})
    with pytest.raises(RehearsalError, match="numeric key_copy"):
        normalize_stage_metrics({"stages": {
            "echo_kv": {"exact": 0.6, "key_copy": None}}})
    with pytest.raises(RehearsalError, match="at least one stage"):
        normalize_stage_metrics({"stages": {}})


def test_rank_stages_weakest_first_and_tie_breaks_canonically():
    ranked = rank_stages(BEFORE)
    assert [item["stage"] for item in ranked] == [
        "multi", "two_kv", "two_json", "echo_kv", "echo_json"]
    assert [item["rank"] for item in ranked] == [1, 2, 3, 4, 5]
    assert ranked[0]["score"] == pytest.approx(0.34)
    tied = rank_stages({
        "echo_kv": {"exact": 0.2, "key_copy": 0.2},
        "multi": {"exact": 0.2, "key_copy": 0.2},
        "two_kv": {"exact": 0.1, "key_copy": 0.1},
    })
    assert [item["stage"] for item in tied] == [
        "two_kv", "echo_kv", "multi"]


def test_mix_weights_follow_50_30_20_groups():
    ranked = rank_stages(BEFORE)
    groups = stage_groups(ranked)
    assert groups == {
        "weakest": ["multi", "two_kv"],
        "middle": ["two_json"],
        "strongest": ["echo_kv", "echo_json"],
    }
    weights = mix_weights(ranked)
    assert weights["multi"] == pytest.approx(0.25)
    assert weights["two_kv"] == pytest.approx(0.25)
    assert weights["two_json"] == pytest.approx(0.30)
    assert weights["echo_kv"] == pytest.approx(0.10)
    assert weights["echo_json"] == pytest.approx(0.10)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_mix_weights_redistribute_missing_groups():
    ranked = rank_stages({
        "echo_kv": {"exact": 0.6, "key_copy": 0.6},
        "two_kv": {"exact": 0.4, "key_copy": 0.4},
        "multi": {"exact": 0.1, "key_copy": 0.1},
    })
    weights = mix_weights(ranked)
    assert [item["stage"] for item in ranked] == [
        "multi", "two_kv", "echo_kv"]
    assert weights["multi"] == pytest.approx(0.3125)
    assert weights["two_kv"] == pytest.approx(0.3125)
    assert weights["echo_kv"] == pytest.approx(0.375)
    assert sum(weights.values()) == pytest.approx(1.0)


def test_allocate_counts_preserves_total_and_uses_remainders():
    counts = allocate_counts({
        "multi": 0.25, "two_json": 0.25, "two_kv": 0.30,
        "echo_kv": 0.10, "echo_json": 0.10,
    }, 1000)
    assert counts == {
        "multi": 250, "two_json": 250, "two_kv": 300,
        "echo_kv": 100, "echo_json": 100,
    }
    odd = allocate_counts({
        "multi": 0.5, "two_json": 0.3, "two_kv": 0.2,
    }, 10)
    assert sum(odd.values()) == 10
    assert odd == {"multi": 5, "two_json": 3, "two_kv": 2}


def test_sampling_is_deterministic_per_seed_and_keeps_counts():
    rows = _stage_rows()
    mix = build_rehearsal_mix(BEFORE, rows, rows=20, seed=23)
    first = [row["id"] for row in mix["sample"]]
    second = [row["id"] for row in mix["sample"]]
    assert first == second
    assert len(first) == 20
    counts = {stage: sum(1 for row_id in first
                         if row_id.startswith(stage + "-"))
              for stage in BEFORE}
    assert counts == mix["counts"]
    assert counts == {"multi": 5, "two_kv": 5, "two_json": 6,
                      "echo_kv": 2, "echo_json": 2}
    other = [row["id"] for row in sample_rehearsal_rows(
        rows, mix["counts"], seed=24)]
    assert other != first


def test_sample_rehearsal_rows_rejects_missing_stage_rows():
    with pytest.raises(RehearsalError, match="no training rows"):
        sample_rehearsal_rows({}, {"multi": 1}, seed=1)


def test_discover_stages_and_load_rows_from_fake_files(tmp_path):
    root = _stage_data(tmp_path)
    assert discover_stages(root) == list(BEFORE)
    rows = read_jsonl(root / "stage_multi" / "train.jsonl")
    assert len(rows) == 8
    assert rows[0]["id"] == 0


def test_evaluate_stages_uses_injected_evaluator_and_paths(tmp_path):
    root = _stage_data(tmp_path)
    calls = []

    def evaluator(base, adapter, eval_path, stage):
        calls.append((base, adapter, eval_path, stage))
        return {"exact": 0.5 if stage == "echo_kv" else 0.25,
                "key_copy": 0.75}

    metrics = evaluate_stages(
        "base", "adapter", root, ["echo_kv", "two_kv"],
        evaluator=evaluator)
    assert metrics == {
        "echo_kv": {"exact": 0.5, "key_copy": 0.75},
        "two_kv": {"exact": 0.25, "key_copy": 0.75},
    }
    assert Path(calls[0][2]).parts[-2:] == ("stage_echo_kv", "eval.jsonl")


def test_comparison_stage_metrics_extracts_adapter_rates():
    payload = {"models": [
        {"label": "base", "metrics": {"exact_match": {"rate": 0.0}}},
        {"label": "weak", "metrics": {
            "exact_match": {"rate": 0.4},
            "key_copy": {"rate": 0.6, "key_recall": 0.7},
        }},
    ]}
    assert comparison_stage_metrics(payload) == {
        "exact": 0.4, "key_copy": 0.6}


def test_build_trainer_command_contains_rehearsal_flags():
    command = build_trainer_command(
        base="base", train="mix.jsonl", eval_path="eval.jsonl",
        out="adapter", init_adapter="init", steps=300, batch_size=4,
        lr=0.0005, content_weight=2.0, seed=23,
        result="result.json", comparison="comparison.json")
    assert command[0] == sys.executable
    assert command[1] == "research/train_adapter_candidate.py"

    def value(flag):
        return command[command.index(flag) + 1]

    assert value("--base") == "base"
    assert value("--train") == "mix.jsonl"
    assert value("--eval") == "eval.jsonl"
    assert value("--out") == "adapter"
    assert value("--init-adapter") == "init"
    assert value("--steps") == "300"
    assert value("--batch-size") == "4"
    assert value("--lr") == "0.0005"
    assert value("--content-weight") == "2.0"
    assert value("--seed") == "23"
    assert value("--result") == "result.json"
    assert value("--comparison") == "comparison.json"
    assert "--rank" not in command


def _mix():
    return build_rehearsal_mix(BEFORE, _stage_rows(), rows=20, seed=23)


def test_ledger_has_before_after_counts_and_trainer_status():
    mix = _mix()
    trainer = {"status": "completed", "returncode": 0}
    ledger = build_ledger(
        seed=23, rows=20, data_dir="data/stages",
        base_adapter="checkpoints/init", new_adapter="checkpoints/new",
        ranking=mix["ranking"], groups=mix["groups"],
        weights=mix["weights"], counts=mix["counts"],
        before=BEFORE, after=AFTER, trainer=trainer, status="completed")
    assert ledger["schema"] == "atomzero.weak_stage_rehearsal.v1"
    assert ledger["content_free"] is True
    assert ledger["status"] == "completed"
    assert ledger["mix"]["total"] == 20
    assert ledger["mix"]["counts"]["two_json"] == 6
    assert ledger["trainer"]["status"] == "completed"
    stages = {record["stage"]: record for record in ledger["stages"]}
    multi = stages["multi"]
    assert multi["group"] == "weakest"
    assert multi["mix_rows"] == 5
    assert multi["before"] == {"exact": 0.12, "key_copy": 0.22}
    assert multi["after"] == {"exact": 0.30, "key_copy": 0.40}
    assert multi["delta"]["exact"] == pytest.approx(0.18)
    assert multi["delta"]["key_copy"] == pytest.approx(0.18)
    assert ledger["stage_order"][0] == "multi"
    assert ledger["stages"][0]["rank"] == 1


def test_run_trainer_records_nonzero_exit(tmp_path):
    def runner(command, **kwargs):
        assert command == ["train"]
        return SimpleNamespace(returncode=2, stdout="", stderr="")

    status = run_trainer(
        ["train"], runner=runner, adapter_path=tmp_path / "missing",
        result_path=tmp_path / "missing.json")
    assert status["status"] == "failed"
    assert status["returncode"] == 2
    assert status["adapter_exists"] is False
    assert "non-zero" in status["error"]


def test_run_trainer_records_launch_failure():
    def runner(command, **kwargs):
        raise FileNotFoundError("python not found")

    status = run_trainer(["train"], runner=runner)
    assert status["status"] == "failed"
    assert status["returncode"] is None
    assert "FileNotFoundError" in status["error"]


def _trainer_kwargs(tmp_path, runner, evaluator=None):
    return {
        "base": "base",
        "init_adapter": "init",
        "data_dir": str(_stage_data(tmp_path)),
        "stages": list(BEFORE),
        "before": BEFORE,
        "stage_rows": _stage_rows(),
        "rows": 20,
        "seed": 23,
        "out_adapter": str(tmp_path / "adapter"),
        "train_jsonl": str(tmp_path / "mix.jsonl"),
        "eval_jsonl": str(tmp_path / "eval.jsonl"),
        "trainer_result": str(tmp_path / "trainer.json"),
        "trainer_comparison": str(tmp_path / "comparison.json"),
        "steps": 30,
        "batch_size": 2,
        "lr": 5e-4,
        "content_weight": 2.0,
        "trainer_runner": runner,
        "evaluator": evaluator,
    }


def test_run_rehearsal_writes_content_free_ledger_on_success(tmp_path):
    adapter = tmp_path / "adapter"
    calls = []

    def runner(command, **kwargs):
        calls.append(command)
        adapter.mkdir()
        (tmp_path / "trainer.json").write_text(json.dumps({
            "train_loss": 0.5, "optimizer_steps": 10,
            "lora_parameters": 42, "user": "TOP SECRET PROMPT",
        }), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def evaluator(base, adapter_path, eval_path, stage):
        return dict(AFTER[stage])

    ledger = run_rehearsal(
        **_trainer_kwargs(tmp_path, runner, evaluator))
    assert ledger["status"] == "completed"
    assert ledger["trainer"]["status"] == "completed"
    assert ledger["trainer"]["result"] == {
        "train_loss": 0.5, "optimizer_steps": 10,
        "lora_parameters": 42}
    assert len(read_jsonl(tmp_path / "mix.jsonl")) == 20
    assert calls and calls[0][1] == "research/train_adapter_candidate.py"
    stages = {record["stage"]: record for record in ledger["stages"]}
    assert stages["multi"]["after"] == AFTER["multi"]
    assert stages["multi"]["delta"]["exact"] == pytest.approx(0.18)
    text = json.dumps(ledger)
    assert "TOP SECRET" not in text
    assert "assistant" not in text


def test_run_rehearsal_records_trainer_failure_and_skips_eval(tmp_path):
    def runner(command, **kwargs):
        return SimpleNamespace(returncode=1, stdout="", stderr="bad")

    def evaluator(*args):
        raise AssertionError("evaluation must not run after trainer failure")

    ledger = run_rehearsal(
        **_trainer_kwargs(tmp_path, runner, evaluator))
    assert ledger["status"] == "trainer_failed"
    assert ledger["trainer"]["status"] == "failed"
    assert ledger["trainer"]["returncode"] == 1
    assert all(record["after"] is None for record in ledger["stages"])
    assert all(record["delta"] is None for record in ledger["stages"])
    assert len(read_jsonl(tmp_path / "mix.jsonl")) == 20


def test_run_rehearsal_records_eval_failure(tmp_path):
    adapter = tmp_path / "adapter"

    def runner(command, **kwargs):
        adapter.mkdir()
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def evaluator(*args):
        raise RuntimeError("stage eval exploded")

    ledger = run_rehearsal(
        **_trainer_kwargs(tmp_path, runner, evaluator))
    assert ledger["status"] == "eval_failed"
    assert ledger["error"] == {
        "type": "RuntimeError", "message": "stage eval exploded"}
    assert all(record["after"] is None for record in ledger["stages"])


def test_run_rehearsal_rejects_metric_stage_mismatch(tmp_path):
    kwargs = _trainer_kwargs(
        tmp_path, lambda command, **options: SimpleNamespace(returncode=1))
    kwargs["stages"] = list(BEFORE) + ["extra"]
    with pytest.raises(RehearsalError, match="do not match"):
        run_rehearsal(**kwargs)
