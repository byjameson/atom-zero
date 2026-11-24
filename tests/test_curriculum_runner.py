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
    meta.add_argument("--gate-status", default="pass")
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
        (out / "adapter.pt").write_bytes(b"fake-adapter")
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

    if options.role == "gate":
        candidate = json.loads(
            Path(flags["candidate-json"]).read_text(encoding="utf-8"))
        comparison = json.loads(
            Path(flags["comparison-json"]).read_text(encoding="utf-8"))
        passed = options.gate_status == "pass"
        adapter = comparison.get("adapter") or {}
        reasons = ["loss_improved", "ppl_not_worse"]
        if "require-content" in flags:
            reasons.extend(["content_loss_improved", "content_ppl_not_worse"])
        if "min-exact-match" in flags:
            reasons.append("exact_match_ok" if passed else "exact_match_low")
        if "min-key-copy-rate" in flags:
            reasons.append("key_copy_ok" if passed else "key_copy_low")
        payload = {
            "status": "pass" if passed else "fail",
            "ok": passed,
            "promotable": passed,
            "candidate_id": candidate.get("candidate_id", ""),
            "eval_examples": comparison.get("examples"),
            "adapter_exact_match": adapter.get("exact_match_rate"),
            "min_exact_match": (float(flags["min-exact-match"])
                                if "min-exact-match" in flags else None),
            "adapter_key_copy_rate": adapter.get("key_copy_rate"),
            "min_key_copy_rate": (float(flags["min-key-copy-rate"])
                                  if "min-key-copy-rate" in flags else None),
            "min_content_delta": float(flags["min-content-delta"]),
            "max_content_ppl_ratio": float(flags["max-content-ppl-ratio"]),
            "require_content": "require-content" in flags,
            "reasons": reasons,
        }
        print(json.dumps(payload))
        return 0 if passed else 1

    if options.role == "promote":
        gate = json.loads(
            Path(flags["gate-json"]).read_text(encoding="utf-8"))
        target = Path(flags["active-file"])
        target.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "path": flags["adapter-dir"],
            "candidate_id": gate.get("candidate_id", ""),
            "gate_status": gate.get("status"),
            "promotable": True,
            "family_separation": True,
            "promoted_at": "2026-11-24T00:00:00Z",
            "previous": "",
            "weights_changed": False,
            "auto_apply": False,
            "method": "gated_adapter_promotion_v1",
        }
        target.write_text(json.dumps(record), encoding="utf-8")
        print(json.dumps(record))
        return 0

    rows = [{"user": SECRET, "assistant": SECRET}]
    rate = options.key_copy_rate
    payload = {
        "examples": 4,
        "models": [
            {"label": "base", "path": flags["base"],
             "metrics": {"loss": 3.0, "ppl": 20.0, "tokens": 40,
                         "content_loss": 3.2, "content_ppl": 24.5,
                         "content_tokens": 30, "rows": rows}},
            {"label": Path(flags["adapters"]).name,
             "path": flags["adapters"],
             "metrics": {
                 "loss": 1.5, "ppl": 4.5, "tokens": 40,
                 "content_loss": 1.4, "content_ppl": 4.1,
                 "content_tokens": 30,
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


def _gate_commands(tmp_path, *, gate_status="pass"):
    script = tmp_path / "fake_backend.py"
    log = tmp_path / "calls.jsonl"
    gate = [sys.executable, str(script), "--role", "gate",
            "--log", str(log), "--gate-status", gate_status]
    promote = [sys.executable, str(script), "--role", "promote",
               "--log", str(log)]
    return gate, promote


def _write_final_eval(path, *, rows=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps({
                "user": f"SECRET_FINAL_PROMPT_{index}",
                "assistant": f"SECRET_FINAL_ANSWER_{index}",
                "family": "staged_copy_mixed",
                "keys": [f"key{index}"],
                "values": [f"value{index}"],
            }) + "\n"
            for index in range(rows)),
        encoding="utf-8")


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
    assert result["gate"]["status"] == "skipped"
    assert result["gate"]["reasons"] == ["final_eval_empty"]
    assert result["promotion"]["reason"] == "not_requested"
    assert result["active_file"] is None
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


def test_first_stage_init_adapter(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    _write_stage_files(data_dir, ("echo_kv",))
    seed = tmp_path / "seed_adapter"
    seed.mkdir()

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        init_adapter=str(seed),
        stages="echo_kv",
        steps=10,
        trainer_command=trainer,
        comparator_command=comparator,
    )

    assert Path(result["stages"][0]["init_adapter"]) == seed
    assert result["options"]["init_adapter"] == str(seed)
    calls = _calls(log)
    trainer_calls = [call for call in calls if call["role"] == "trainer"]
    first = _flags(trainer_calls[0]["argv"])
    assert Path(first["init-adapter"]) == seed


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


def test_final_gate_thresholds_propagate(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    gate, promote = _gate_commands(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    final_eval = tmp_path / "mixed_eval.jsonl"
    candidate_json = tmp_path / "candidate.json"
    gate_json = tmp_path / "gate.json"
    _write_stage_files(data_dir, ("echo_kv",))
    _write_final_eval(final_eval)

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        stages="echo_kv",
        steps=10,
        final_eval=final_eval,
        require_content=True,
        min_exact_match=0.25,
        min_key_copy_rate=0.3,
        min_content_delta=0.07,
        max_content_ppl_ratio=0.9,
        candidate_json=candidate_json,
        gate_json=gate_json,
        trainer_command=trainer,
        comparator_command=comparator + ["--key-copy-rate", "0.548"],
        gate_command=gate,
        promote_command=promote,
    )

    gate_calls = [call for call in _calls(log) if call["role"] == "gate"]
    assert len(gate_calls) == 1
    flags = _flags(gate_calls[0]["argv"])
    assert flags["require-content"] is True
    assert flags["min-exact-match"] == "0.25"
    assert flags["min-key-copy-rate"] == "0.3"
    assert flags["min-content-delta"] == "0.07"
    assert flags["max-content-ppl-ratio"] == "0.9"
    assert Path(flags["candidate-json"]) == candidate_json
    assert not Path(flags["comparison-json"]).exists()  # temp aggregate

    assert result["gate"]["status"] == "pass"
    assert result["gate"]["thresholds"] == {
        "require_content": True,
        "min_exact_match": 0.25,
        "min_key_copy_rate": 0.3,
        "min_content_delta": 0.07,
        "max_content_ppl_ratio": 0.9,
    }
    assert result["exact_match_rate"] == 0.5
    assert result["key_copy_rate"] == 0.548
    assert result["final_eval"]["examples"] == 4
    assert result["final_eval"]["metrics"]["adapter"]["content_loss"] == 1.4
    assert result["final_eval"]["metrics"]["adapter"]["content_ppl"] == 4.1

    candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
    assert candidate["eval_examples"] == 4
    assert candidate["candidate_id"] == result["candidate_id"]
    assert candidate["family_separation"] is True
    assert candidate["eval_family_counts"] == {"staged_copy_mixed": 4}
    assert gate_json.is_file()
    gate_file = json.loads(gate_json.read_text(encoding="utf-8"))
    assert gate_file["status"] == "pass"
    assert "SECRET_FINAL" not in json.dumps(result)
    assert "SECRET_FINAL" not in candidate_json.read_text(encoding="utf-8")
    assert "SECRET_FINAL" not in gate_json.read_text(encoding="utf-8")


def test_promotion_only_when_gate_passes(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    gate, promote = _gate_commands(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    final_eval = tmp_path / "mixed_eval.jsonl"
    active_file = tmp_path / "curriculum_active_adapter.json"
    _write_stage_files(data_dir, ("echo_kv",))
    _write_final_eval(final_eval)

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        stages="echo_kv",
        steps=10,
        final_eval=final_eval,
        require_content=True,
        min_exact_match=0.25,
        min_key_copy_rate=0.3,
        active_file=active_file,
        promote=True,
        trainer_command=trainer,
        comparator_command=comparator + ["--key-copy-rate", "0.548"],
        gate_command=gate,
        promote_command=promote,
    )

    assert result["promotion"]["requested"] is True
    assert result["promotion"]["promoted"] is True
    assert result["promotion"]["reason"] == "promoted"
    assert Path(result["promotion"]["active_file"]) == active_file
    assert result["active_file"] == str(active_file)
    assert result["promotion_error"] is None
    registry = json.loads(active_file.read_text(encoding="utf-8"))
    assert registry["candidate_id"] == result["candidate_id"]
    assert registry["gate_status"] == "pass"
    assert registry["weights_changed"] is False
    promote_calls = [call for call in _calls(log)
                     if call["role"] == "promote"]
    assert len(promote_calls) == 1
    flags = _flags(promote_calls[0]["argv"])
    assert Path(flags["adapter-dir"]) == out_dir / "echo_kv"
    assert Path(flags["gate-json"]).is_file()
    assert "SECRET_FINAL" not in active_file.read_text(encoding="utf-8")


def test_failed_gate_never_touches_registry(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    gate, promote = _gate_commands(tmp_path, gate_status="fail")
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    final_eval = tmp_path / "mixed_eval.jsonl"
    active_file = tmp_path / "curriculum_active_adapter.json"
    active_file.write_text('{"path": "old/adapter"}\n', encoding="utf-8")
    _write_stage_files(data_dir, ("echo_kv",))
    _write_final_eval(final_eval)

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        stages="echo_kv",
        steps=10,
        final_eval=final_eval,
        require_content=True,
        min_exact_match=0.25,
        min_key_copy_rate=0.3,
        active_file=active_file,
        promote=True,
        trainer_command=trainer,
        comparator_command=comparator + ["--key-copy-rate", "0.548"],
        gate_command=gate,
        promote_command=promote,
    )

    assert result["gate"]["status"] == "fail"
    assert result["gate"]["ok"] is False
    assert result["promotion"]["requested"] is True
    assert result["promotion"]["promoted"] is False
    assert result["promotion"]["reason"] == "gate_failed"
    assert active_file.read_text(encoding="utf-8") == '{"path": "old/adapter"}\n'
    assert not [call for call in _calls(log) if call["role"] == "promote"]


def test_dry_run_gate_writes_fake_outputs_and_no_registry(tmp_path):
    result_path = tmp_path / "result.json"
    candidate_json = tmp_path / "candidate.json"
    gate_json = tmp_path / "gate.json"
    active_file = tmp_path / "curriculum_active_adapter.json"
    out_dir = tmp_path / "adapters"

    code = runner.main([
        "--dry-run", "--promote", "--require-content",
        "--min-exact-match", "0.25", "--min-key-copy-rate", "0.3",
        "--stages", "echo_kv,echo_json,multi",
        "--steps", "20", "--batch-size", "4",
        "--out-dir", str(out_dir),
        "--result", str(result_path),
        "--candidate-json", str(candidate_json),
        "--gate-json", str(gate_json),
        "--active-file", str(active_file),
    ])

    assert code == 0
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["final_eval"]["examples"] == 250
    assert payload["exact_match_rate"] == 0.4
    assert payload["key_copy_rate"] == 0.548
    assert payload["gate"]["status"] == "pass"
    assert payload["gate"]["thresholds"]["min_exact_match"] == 0.25
    assert payload["gate"]["thresholds"]["min_key_copy_rate"] == 0.3
    assert payload["promotion"]["requested"] is True
    assert payload["promotion"]["promoted"] is False
    assert payload["promotion"]["reason"] == "dry_run"
    assert not active_file.exists()
    assert not out_dir.exists()
    assert candidate_json.is_file()
    assert gate_json.is_file()
    candidate = json.loads(candidate_json.read_text(encoding="utf-8"))
    assert candidate["eval_examples"] == 250
    assert candidate["family_separation"] is True


def test_real_gate_and_promote_cli(tmp_path):
    trainer, comparator, log = _fake_backend(tmp_path)
    data_dir = tmp_path / "data"
    out_dir = tmp_path / "adapters"
    final_eval = tmp_path / "mixed_eval.jsonl"
    active_file = tmp_path / "curriculum_active_adapter.json"
    _write_stage_files(data_dir, ("echo_kv",))
    _write_final_eval(final_eval)

    result = runner.run_curriculum(
        base="checkpoints/base",
        data_dir=data_dir,
        out_dir=out_dir,
        stages="echo_kv",
        steps=10,
        final_eval=final_eval,
        require_content=True,
        min_exact_match=0.25,
        min_key_copy_rate=0.3,
        active_file=active_file,
        promote=True,
        trainer_command=trainer,
        comparator_command=comparator + ["--key-copy-rate", "0.548"],
    )

    assert result["gate"]["status"] == "pass"
    assert "content_loss_improved" in result["gate"]["reasons"]
    assert "content_ppl_not_worse" in result["gate"]["reasons"]
    assert "exact_match_ok" in result["gate"]["reasons"]
    assert "key_copy_ok" in result["gate"]["reasons"]
    assert result["promotion"]["promoted"] is True
    registry = json.loads(active_file.read_text(encoding="utf-8"))
    assert registry["candidate_id"] == result["candidate_id"]
    assert registry["gate_status"] == "pass"
    assert registry["family_separation"] is True
    assert registry["weights_changed"] is False


def test_promote_requires_active_file_and_behavior_thresholds(tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        runner.main(["--dry-run", "--promote", "--require-content",
                     "--min-exact-match", "0.25",
                     "--result", str(tmp_path / "result.json")])
    assert excinfo.value.code == 2

    with pytest.raises(SystemExit) as excinfo:
        runner.main(["--dry-run", "--promote", "--require-content",
                     "--min-key-copy-rate", "0.3",
                     "--active-file", str(tmp_path / "active.json"),
                     "--result", str(tmp_path / "result.json")])
    assert excinfo.value.code == 2
