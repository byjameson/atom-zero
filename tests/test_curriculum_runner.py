import ast
import json
import sys
from pathlib import Path

import pytest

from research import run_staged_curriculum as runner
from research.run_staged_curriculum import (
    CurriculumError,
    build_comparator_command,
    build_trainer_command,
    summarize_comparison,
)

FAKE_BACKEND = r'''
import argparse
import json
import sys
from pathlib import Path

SECRET = "SECRET_PROMPT_TEXT"


def _flags(items):
    flags = {}
    index = 0
    while index < len(items):
        item = items[index]
        if item.startswith("--"):
            key = item[2:]
            if index + 1 < len(items) and not items[index + 1].startswith("--"):
                flags[key] = items[index + 1]
                index += 2
            else:
                flags[key] = True
                index += 1
        else:
            index += 1
    return flags


def main():
    meta = argparse.ArgumentParser(allow_abbrev=False)
    meta.add_argument("--role", required=True)
    meta.add_argument("--log", required=True)
    meta.add_argument("--key-copy-rate", type=float, default=0.0)
    meta.add_argument("--fail-stage", default="")
    meta.add_argument("--fail-code", type=int, default=1)
    options, extra = meta.parse_known_args()
    with open(options.log, "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"role": options.role, "argv": extra}) + "\n")
    flags = _flags(extra)
    if options.role == "trainer":
        stage = Path(flags["out"]).name
        if options.fail_stage == stage:
            print(f"trainer exploded for {stage}", file=sys.stderr)
            return options.fail_code
        out = Path(flags["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "config.json").write_text(json.dumps({
            "r_mlp": 8, "alpha_mlp": 8, "r_attn": 4, "alpha_attn": 8,
            "steps": 10, "batch_size": 4, "method": "fake",
            "prompt": SECRET,
        }), encoding="utf-8")
        result = Path(flags["result"])
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps({
            "train_loss": 0.5, "train_steps": 10, "optimizer_steps": 3,
            "rank_mlp": 8, "alpha_mlp": 8, "rank_attn": 4,
            "alpha_attn": 8, "lora_parameters": 1234,
            "prompt": SECRET,
        }), encoding="utf-8")
        Path(flags["comparison"]).write_text(
            json.dumps({"models": []}), encoding="utf-8")
        return 0

    rows = [{"user": SECRET, "assistant": SECRET}]
    rate = options.key_copy_rate
    payload = {
        "examples": 4,
        "models": [
            {"label": "base", "path": flags["base"],
             "metrics": {"loss": 3.0, "ppl": 20.0, "tokens": 40,
                         "rows": rows}},
            {"label": Path(flags["adapters"]).name,
             "path": flags["adapters"],
             "metrics": {
                 "loss": 1.5, "ppl": 4.5, "tokens": 40,
                 "content_loss": 1.4, "content_tokens": 30,
                 "key_loss": 1.7, "key_tokens": 5,
                 "value_loss": 1.6, "value_tokens": 4,
                 "syntax_loss": 1.2, "syntax_tokens": 1,
                 "exact_match": {"examples": 4, "matches": 2, "rate": 0.5,
                                 "families": {"family": {
                                     "examples": 4, "matches": 2,
                                     "rate": 0.5}}},
                 "key_copy": {"examples": 4, "matches": 2, "rate": rate,
                              "key_recall": 0.75,
                              "families": {"family": {
                                  "examples": 4, "matches": 2, "rate": rate,
                                  "key_recall": 0.75}}},
                 "rows": rows,
             },
             "families": {"family": {
                 "baseline": {"loss": 3.0, "tokens": 40},
                 "adapter": {"loss": 1.5, "tokens": 40},
             }}},
        ],
    }
    target = Path(flags["out"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")
    return 0


raise SystemExit(main())
'''


def _fake_backend(tmp_path):
    script = tmp_path / "fake_backend.py"
    script.write_text(FAKE_BACKEND, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    trainer = [sys.executable, str(script), "--role", "trainer",
               "--log", str(log)]
    comparator = [sys.executable, str(script), "--role", "comparator",
                  "--log", str(log)]
    return trainer, comparator, log


def _write_stage_files(data_dir, stages):
    for stage in stages:
        stage_dir = data_dir / f"stage_{stage}"
        stage_dir.mkdir(parents=True, exist_ok=True)
        (stage_dir / "train.jsonl").write_text("", encoding="utf-8")
        (stage_dir / "eval.jsonl").write_text("", encoding="utf-8")


def _calls(log):
    if not log.exists():
        return []
    return [json.loads(line) for line in
            log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _flags(argv):
    flags = {}
    index = 0
    while index < len(argv):
        item = argv[index]
        if item.startswith("--"):
            key = item[2:]
            if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
                flags[key] = argv[index + 1]
                index += 2
            else:
                flags[key] = True
                index += 1
        else:
            index += 1
    return flags


def _trainer_command(init_adapter=None):
    return build_trainer_command(
        base="checkpoints/base",
        train="data/staged_copy_v1/stage_echo_kv/train.jsonl",
        eval="data/staged_copy_v1/stage_echo_kv/eval.jsonl",
        out="checkpoints/staged_curriculum/echo_kv",
        result="checkpoints/staged_curriculum/echo_kv/trainer_result.json",
        comparison="tmp/comparison.json",
        steps=2000,
        batch_size=4,
        lr=3e-4,
        content_weight=2.0,
        replay_weight=0.0,
        replay_jsonl=None,
        rank=8,
        alpha=8,
        rank_attn=4,
        alpha_attn=8,
        seed=23,
        init_adapter=init_adapter,
    )


def test_runner_does_not_import_torch():
    tree = ast.parse(Path(runner.__file__).read_text(encoding="utf-8"))
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
    assert not any(name == "torch" or name.startswith("torch.")
                   for name in imported)


def test_trainer_command_chains_init_adapter():
    first = _trainer_command()
    assert "--init-adapter" not in first
    assert first[first.index("--steps") + 1] == "2000"
    assert first[first.index("--batch-size") + 1] == "4"
    assert first[first.index("--rank") + 1] == "8"
    assert first[first.index("--alpha") + 1] == "8"
    assert first[first.index("--rank-attn") + 1] == "4"
    assert first[first.index("--alpha-attn") + 1] == "8"
    assert first[first.index("--content-weight") + 1] == "2.0"
    assert first[first.index("--replay-weight") + 1] == "0.0"

    previous = "checkpoints/staged_curriculum/echo_kv"
    second = _trainer_command(init_adapter=previous)
    assert second[second.index("--init-adapter") + 1] == previous

    comparator = build_comparator_command(
        base="checkpoints/base",
        data="data/staged_copy_v1/stage_two_kv/eval.jsonl",
        adapter="checkpoints/staged_curriculum/two_kv",
        out="tmp/comparison.json",
    )
    assert "--per-family" in comparator
    assert "--exact-match" in comparator
    assert "--key-copy" in comparator
    adapter_index = comparator.index("--adapters") + 1
    assert comparator[adapter_index] == (
        "checkpoints/staged_curriculum/two_kv")


def test_ledger_strips_rows_and_prompt_text():
    payload = {
        "examples": 4,
        "models": [
            {"label": "base", "metrics": {
                "loss": 3.0, "tokens": 40,
                "rows": [{"user": "SECRET", "assistant": "SECRET"}]}},
            {"label": "echo_kv", "metrics": {
                "loss": 1.5, "tokens": 40, "content_loss": 1.4,
                "content_tokens": 30, "key_loss": 1.7, "key_tokens": 5,
                "value_loss": 1.6, "value_tokens": 4,
                "syntax_loss": 1.2, "syntax_tokens": 1,
                "exact_match": {"examples": 4, "matches": 2, "rate": 0.5,
                                "families": {"family": {
                                    "examples": 4, "matches": 2,
                                    "rate": 0.5}}},
                "key_copy": {"examples": 4, "matches": 3, "rate": 0.75,
                             "key_recall": 0.875,
                             "families": {"family": {
                                 "examples": 4, "matches": 3,
                                 "rate": 0.75, "key_recall": 0.875}}},
                "rows": [{"user": "SECRET", "assistant": "SECRET"}],
            }, "families": {"family": {
                "baseline": {"loss": 3.0, "tokens": 40},
                "adapter": {"loss": 1.5, "tokens": 40},
            }}},
        ],
    }
    summary = summarize_comparison(payload)
    text = json.dumps(summary)
    assert "SECRET" not in text
    assert "rows" not in text
    assert "user" not in text
    assert "assistant" not in text
    assert summary["adapter"]["loss"] == 1.5
    assert summary["adapter"]["key_loss"] == 1.7
    assert summary["adapter"]["exact_match_rate"] == 0.5
    assert summary["adapter"]["key_copy_rate"] == 0.75
    assert summary["adapter"]["key_recall"] == 0.875
    assert summary["per_family"]["family"]["adapter"]["loss"] == 1.5
    assert summary["per_family"]["family"]["exact_match"]["rate"] == 0.5
    assert summary["per_family"]["family"]["key_copy"]["rate"] == 0.75


def test_stage_order_and_init_adapter_chaining(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    _write_stage_files(data_dir, ("echo_kv", "echo_json"))
    multi_dir = data_dir / "stage_multi"
    multi_dir.mkdir(parents=True)
    (multi_dir / "eval.jsonl").write_text("", encoding="utf-8")

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        stages="echo_json,echo_kv",
        steps=10,
        trainer_command=trainer,
        comparator_command=comparator,
    )

    assert result["stage_order"] == ["echo_kv", "echo_json"]
    assert result["stopped_early"] is False
    assert result["stages"][0]["init_adapter"] is None
    assert Path(result["stages"][1]["init_adapter"]) == out_dir / "echo_kv"
    assert result["stages"][0]["trainer"]["returncode"] == 0
    assert result["stages"][0]["trainer"]["adapter_config"]["r_mlp"] == 8
    assert result["stages"][0]["trainer"]["elapsed_seconds"] >= 0
    assert set(result["stages"][0]["evals"]) == {"stage", "multi"}
    assert Path(result["stages"][0]["evals"]["multi"]["data"]) == (
        data_dir / "stage_multi" / "eval.jsonl")
    assert "SECRET_PROMPT_TEXT" not in json.dumps(result)

    calls = _calls(log)
    trainer_calls = [call for call in calls if call["role"] == "trainer"]
    assert len(trainer_calls) == 2
    first = _flags(trainer_calls[0]["argv"])
    second = _flags(trainer_calls[1]["argv"])
    assert Path(first["out"]) == out_dir / "echo_kv"
    assert "init-adapter" not in first
    assert Path(second["out"]) == out_dir / "echo_json"
    assert Path(second["init-adapter"]) == out_dir / "echo_kv"
    assert first["steps"] == "10"


def test_early_stop_when_key_copy_gate_reached(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    _write_stage_files(data_dir, ("echo_kv", "echo_json"))

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        stages="echo_kv,echo_json",
        stop_on_key_copy=0.5,
        trainer_command=trainer,
        comparator_command=comparator + ["--key-copy-rate", "0.9"],
    )

    assert result["stopped_early"] is True
    assert result["stop_stage"] == "echo_kv"
    assert result["stage_order"] == ["echo_kv"]
    assert result["stages"][0]["key_copy_rate"] == 0.9
    assert result["stages"][0]["key_copy_gate"] == {
        "threshold": 0.5, "passed": True}
    trainer_calls = [call for call in _calls(log)
                     if call["role"] == "trainer"]
    assert len(trainer_calls) == 1


def test_failed_trainer_returncode_aborts_with_clear_error(
        tmp_path, capsys):
    trainer, comparator, _ = _fake_backend(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    result_path = tmp_path / "result.json"
    _write_stage_files(data_dir, ("echo_kv",))

    code = runner.main(
        ["--data-dir", str(data_dir), "--out-dir", str(out_dir),
         "--base", "checkpoints/base", "--stages", "echo_kv",
         "--result", str(result_path)],
        trainer_command=trainer + ["--fail-stage", "echo_kv",
                                   "--fail-code", "3"],
        comparator_command=comparator,
    )

    assert code == 2
    error = capsys.readouterr().err
    assert "echo_kv" in error
    assert "trainer failed with returncode 3" in error
    assert "trainer exploded" in error
    assert not result_path.exists()


def test_dry_run_writes_valid_result_json(tmp_path):
    result_path = tmp_path / "result.json"
    out_dir = tmp_path / "adapters"

    code = runner.main([
        "--dry-run",
        "--out-dir", str(out_dir),
        "--result", str(result_path),
        "--stages", "echo_kv,echo_json,multi",
        "--steps", "20",
        "--batch-size", "4",
        "--stop-on-key-copy", "1.0",
    ])

    assert code == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["schema"] == runner.LEDGER_SCHEMA
    assert payload["dry_run"] is True
    assert payload["stage_order"] == ["echo_kv", "echo_json", "multi"]
    assert payload["stopped_early"] is False
    assert payload["stop_on_key_copy"] == 1.0
    assert payload["options"]["steps"] == 20
    assert not out_dir.exists()
    assert len(payload["stages"]) == 3
    for record in payload["stages"]:
        assert record["trainer"]["returncode"] == 0
        assert record["trainer"]["adapter_config"]["r_mlp"] == 8
        assert record["trainer"]["adapter_config"]["r_attn"] == 4
        assert record["evals"]["stage"]["metrics"]["adapter"]["loss"] > 0
    assert "multi" in payload["stages"][0]["evals"]
    assert "multi" not in payload["stages"][-1]["evals"]
