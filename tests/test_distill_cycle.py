import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.run_distill_cycle import (
    DistillCycleError,
    format_command,
    main,
    read_teacher_manifest,
    run_cycle,
    write_ledger,
)

SECRET = "SECRET_PROMPT_ANSWER_KEY_VALUE_9f3a"

FAKE_BACKEND = r'''
import argparse
import json
import sys
from pathlib import Path

SECRET = "SECRET_PROMPT_ANSWER_KEY_VALUE_9f3a"


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


def _write_json(path, payload):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload), encoding="utf-8")


def _count_rows(path):
    raw = Path(path).read_text(encoding="utf-8")
    return sum(1 for line in raw.splitlines() if line.strip())


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--role", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--fail-code", type=int, default=0)
    options, extra = parser.parse_known_args()
    flags = _flags(extra)
    log = Path(options.log)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"role": options.role, "argv": extra}) + "\n")
    if options.fail_code:
        print(SECRET, file=sys.stderr)
        return options.fail_code

    if options.role == "teacher":
        out_dir = Path(flags["out-dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "train_distill.jsonl").write_text(
            json.dumps({
                "user": SECRET,
                "assistant": SECRET,
                "keys": [SECRET],
                "values": [SECRET],
            }) + "\n",
            encoding="utf-8")
        _write_json(flags["result"], {
            "requested": 8,
            "generated": 7,
            "teacher_valid": 6,
            "fallback_oracle": 1,
            "rejected": 1,
            "by_format": {"json": 4, "kv": 2, "yaml": 1},
            "by_stage": {"2short": 3, "3short": 2, "4short": 2},
            "note": SECRET,
            "source_files": [SECRET],
        })
        return 0

    if options.role == "trainer":
        out = Path(flags["out"])
        out.mkdir(parents=True, exist_ok=True)
        (out / "adapter.pt").write_bytes(b"fake-adapter")
        (out / "config.json").write_text(json.dumps({
            "r_mlp": 8,
            "alpha_mlp": 8,
            "r_attn": 4,
            "alpha_attn": 8,
            "steps": 7,
            "batch_size": 2,
            "content_weight": 2.0,
            "key_weight": 2.0,
            "value_weight": 2.0,
            "prompt": SECRET,
        }), encoding="utf-8")
        _write_json(flags["result"], {
            "train_loss": 0.42,
            "train_steps": 7,
            "optimizer_steps": 4,
            "rows_seen": 7,
            "device": "cuda",
            "prompt": SECRET,
            "keys": [SECRET],
            "values": [SECRET],
            "rows": [{"user": SECRET, "assistant": SECRET}],
        })
        _write_json(flags["comparison"], {
            "models": [{"label": "base", "metrics": {
                "rows": [{"user": SECRET, "assistant": SECRET}]}}],
            "prompt": SECRET,
        })
        return 0

    if options.role == "evaluator":
        rows = _count_rows(flags["data"])
        payload = {
            "examples": rows,
            "prompt": SECRET,
            "keys": [SECRET],
            "values": [SECRET],
            "models": [
                {"label": "base", "path": flags["base"], "metrics": {
                    "examples": rows, "tokens": rows * 20, "loss": 3.5,
                    "ppl": 33.1, "content_loss": 3.0,
                    "content_tokens": rows * 16,
                    "exact_match": {"examples": rows, "matches": 0,
                                    "rate": 0.0},
                    "key_copy": {"examples": rows, "matches": 0,
                                 "rate": 0.0, "key_recall": 0.0},
                    "rows": [{"user": SECRET, "assistant": SECRET}],
                }},
                {"label": "adapter", "path": flags["adapters"], "metrics": {
                    "examples": rows, "tokens": rows * 20, "loss": 1.1,
                    "ppl": 3.0, "content_loss": 1.2,
                    "content_tokens": rows * 16, "key_loss": 1.3,
                    "key_tokens": rows * 2, "value_loss": 1.4,
                    "value_tokens": rows, "syntax_loss": 0.9,
                    "syntax_tokens": rows,
                    "exact_match": {"examples": rows, "matches": 3,
                                    "rate": 0.75},
                    "key_copy": {"examples": rows, "matches": 2,
                                 "rate": 0.5, "key_recall": 0.6,
                                 "families": {"distill_copy_2short": {
                                     "examples": rows, "matches": 2,
                                     "rate": 0.5, "key_recall": 0.6}}},
                    "rows": [{"user": SECRET, "assistant": SECRET}],
                }, "families": {"distill_copy_2short": {
                    "baseline": {"loss": 3.5, "tokens": rows * 20},
                    "adapter": {"loss": 1.1, "tokens": rows * 20},
                }}},
            ],
        }
        _write_json(flags["out"], payload)
        return 0

    return 0


raise SystemExit(main())
'''


def _write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _secret_rows(count):
    return [
        {
            "user": f"{SECRET}_PROMPT_{index}",
            "assistant": f"{SECRET}_ANSWER_{index}",
            "keys": [f"{SECRET}_KEY_{index}"],
            "values": [f"{SECRET}_VALUE_{index}"],
            "family": "distill_copy_2short",
        }
        for index in range(count)
    ]


def _fake_commands(tmp_path, *, teacher_fail=0, trainer_fail=0,
                   eval_fail=0):
    script = tmp_path / "fake_backend.py"
    script.write_text(FAKE_BACKEND, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    teacher = [
        sys.executable, str(script), "--role", "teacher",
        "--log", str(log), "--fail-code", str(teacher_fail),
        "--source", "{source}", "--out-dir", "{out_dir}",
        "--result", "{result}",
    ]
    trainer = [
        sys.executable, str(script), "--role", "trainer",
        "--log", str(log), "--fail-code", str(trainer_fail),
        "--base", "{base}", "--train", "{train}", "--eval", "{eval}",
        "--out", "{out}", "--result", "{result}",
        "--comparison", "{comparison}",
    ]
    evaluator = [
        sys.executable, str(script), "--role", "evaluator",
        "--log", str(log), "--fail-code", str(eval_fail),
        "--base", "{base}", "--data", "{eval}", "--adapters", "{out}",
        "--out", "{comparison}",
    ]
    return teacher, trainer, evaluator, log


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


def _paths(tmp_path, *, eval_rows=6):
    base = tmp_path / "base"
    base.mkdir(parents=True, exist_ok=True)
    source = _write_jsonl(tmp_path / "source_train.jsonl", _secret_rows(2))
    eval_path = _write_jsonl(
        tmp_path / "holdout_eval.jsonl", _secret_rows(eval_rows))
    teacher_data = tmp_path / "teacher" / "train_distill.jsonl"
    _write_jsonl(teacher_data, _secret_rows(1))
    (teacher_data.parent / "manifest.json").write_text(json.dumps({
        "requested": 8, "generated": 7, "teacher_valid": 6,
        "fallback_oracle": 1, "rejected": 1,
        "by_format": {"json": 4, "kv": 2, "yaml": 1},
        "by_stage": {"2short": 3, "3short": 2, "4short": 2},
        "note": SECRET,
    }), encoding="utf-8")
    out = tmp_path / "adapter"
    result = tmp_path / "ledger.json"
    return {
        "base": base,
        "source": source,
        "eval": eval_path,
        "teacher_data": teacher_data,
        "out": out,
        "result": result,
    }


def test_format_command_substitutes_all_placeholders_and_quotes():
    values = {
        "base": r"C:\repo\base",
        "train": r"C:\repo\data\train.jsonl",
        "eval": r"C:\repo\data\eval.jsonl",
        "out": r"C:\repo\checkpoints\adapter",
        "result": r"C:\repo\checkpoints\adapter\result.json",
        "comparison": r"C:\repo\tmp\comparison.json",
    }
    command = format_command(
        'python research/train_adapter_candidate.py --base {base} '
        '--train "{train}" --eval {eval} --out {out} --result {result} '
        '--comparison {comparison}',
        values,
        label="trainer command",
    )
    assert command == [
        "python", "research/train_adapter_candidate.py",
        "--base", r"C:\repo\base",
        "--train", r"C:\repo\data\train.jsonl",
        "--eval", r"C:\repo\data\eval.jsonl",
        "--out", r"C:\repo\checkpoints\adapter",
        "--result", r"C:\repo\checkpoints\adapter\result.json",
        "--comparison", r"C:\repo\tmp\comparison.json",
    ]
    teacher = format_command(
        "python research/build_distill_copy.py --source {source} "
        "--out-dir {out_dir} --result {result}",
        {"source": "data/source.jsonl", "out_dir": "data/distill",
         "result": "data/distill/manifest.json"},
        label="teacher command",
    )
    assert teacher[2:] == [
        "--source", "data/source.jsonl",
        "--out-dir", "data/distill",
        "--result", "data/distill/manifest.json",
    ]
    spaced = format_command(
        "python x.py --out {out}", {"out": r"C:\a b\out"})
    assert spaced == ["python", "x.py", "--out", r"C:\a b\out"]
    with pytest.raises(DistillCycleError, match="invalid trainer command"):
        format_command(["--base", "{unknown}"], values, label="trainer command")
    with pytest.raises(DistillCycleError, match="empty"):
        format_command("   ", values, label="teacher command")


def test_teacher_template_cannot_reference_the_heldout_eval(tmp_path):
    paths = _paths(tmp_path)
    with pytest.raises(DistillCycleError, match="teacher command"):
        run_cycle(
            base=str(paths["base"]),
            eval_path=str(paths["eval"]),
            out_dir=str(paths["out"]),
            teacher_data=str(paths["teacher_data"]),
            train=str(paths["source"]),
            teacher_command=[sys.executable, "-c", "pass", "{eval}"],
        )


def test_teacher_failure_aborts_before_trainer_and_eval(tmp_path):
    paths = _paths(tmp_path)
    teacher, trainer, evaluator, log = _fake_commands(
        tmp_path, teacher_fail=3)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        result_path=str(paths["result"]),
        teacher_data=str(paths["teacher_data"]),
        train=str(paths["source"]),
        teacher_command=teacher,
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "teacher_failed"
    assert ledger["teacher"]["status"] == "failed"
    assert ledger["teacher"]["returncode"] == 3
    assert ledger["teacher"]["reason"] == "returncode"
    assert ledger["trainer"]["status"] == "skipped"
    assert ledger["evaluation"]["status"] == "skipped"
    assert ledger["error"] == {
        "stage": "teacher", "reason": "returncode", "returncode": 3}
    assert [call["role"] for call in _calls(log)] == ["teacher"]
    assert SECRET not in json.dumps(ledger)


def test_silent_teacher_success_cannot_reuse_stale_output(tmp_path):
    paths = _paths(tmp_path)
    _, trainer, evaluator, log = _fake_commands(tmp_path)
    noop = [sys.executable, "-c", "pass"]
    assert paths["teacher_data"].is_file()
    assert (paths["teacher_data"].parent / "manifest.json").is_file()
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        teacher_data=str(paths["teacher_data"]),
        train=str(paths["source"]),
        teacher_command=noop,
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "teacher_failed"
    assert ledger["teacher"]["returncode"] == 0
    assert ledger["teacher"]["reason"] == "teacher_data_missing"
    assert ledger["teacher"]["manifest"]["found"] is False
    assert ledger["trainer"]["status"] == "skipped"
    assert ledger["evaluation"]["status"] == "skipped"
    assert not paths["teacher_data"].exists()
    assert _calls(log) == []


def test_trainer_failure_aborts_before_eval(tmp_path):
    paths = _paths(tmp_path)
    _, trainer, evaluator, log = _fake_commands(
        tmp_path, trainer_fail=4)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        result_path=str(paths["result"]),
        teacher_data=str(paths["teacher_data"]),
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "trainer_failed"
    assert ledger["teacher"]["status"] == "skipped"
    assert ledger["trainer"]["status"] == "failed"
    assert ledger["trainer"]["returncode"] == 4
    assert ledger["evaluation"]["status"] == "skipped"
    assert ledger["error"] == {
        "stage": "trainer", "reason": "returncode", "returncode": 4}
    assert [call["role"] for call in _calls(log)] == ["trainer"]
    assert SECRET not in json.dumps(ledger)


def test_silent_trainer_success_without_result_aborts(tmp_path):
    paths = _paths(tmp_path)
    _, _, evaluator, log = _fake_commands(tmp_path)
    noop = [sys.executable, "-c", "pass"]
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        teacher_data=str(paths["teacher_data"]),
        trainer_command=noop,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "trainer_failed"
    assert ledger["trainer"]["returncode"] == 0
    assert ledger["trainer"]["reason"] == "trainer_result_missing"
    assert ledger["evaluation"]["status"] == "skipped"
    assert _calls(log) == []


def test_evaluator_failure_sets_eval_failed(tmp_path):
    paths = _paths(tmp_path)
    _, trainer, evaluator, log = _fake_commands(tmp_path, eval_fail=5)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        result_path=str(paths["result"]),
        teacher_data=str(paths["teacher_data"]),
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "eval_failed"
    assert ledger["trainer"]["status"] == "completed"
    assert ledger["evaluation"]["status"] == "failed"
    assert ledger["evaluation"]["returncode"] == 5
    assert ledger["evaluation"]["reason"] == "returncode"
    assert ledger["error"] == {
        "stage": "evaluation", "reason": "returncode", "returncode": 5}
    assert [call["role"] for call in _calls(log)] == [
        "trainer", "evaluator"]
    assert SECRET not in json.dumps(ledger)


def test_success_path_uses_fake_commands_and_records_metrics(tmp_path):
    paths = _paths(tmp_path)
    teacher, trainer, evaluator, log = _fake_commands(tmp_path)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        result_path=str(paths["result"]),
        teacher_data=str(paths["teacher_data"]),
        train=str(paths["source"]),
        teacher_command=teacher,
        trainer_command=trainer,
        evaluator_command=evaluator,
        steps=7,
        batch_size=2,
        lr=1e-3,
        content_weight=2.0,
        key_weight=3.0,
        value_weight=4.0,
        seed=99,
        backend="gpu",
    )
    assert ledger["status"] == "completed"
    assert ledger["dry_run"] is False
    assert ledger["backend"] == "gpu"
    assert ledger["teacher"]["status"] == "completed"
    assert ledger["teacher"]["returncode"] == 0
    assert ledger["teacher"]["data_bytes"] > 0
    assert ledger["teacher"]["manifest"]["found"] is True
    assert ledger["teacher"]["manifest"]["counts"] == {
        "requested": 8,
        "generated": 7,
        "teacher_valid": 6,
        "fallback_oracle": 1,
        "rejected": 1,
        "by_format.json": 4,
        "by_format.kv": 2,
        "by_format.yaml": 1,
        "by_stage.2short": 3,
        "by_stage.3short": 2,
        "by_stage.4short": 2,
    }
    assert ledger["teacher"]["manifest"]["error"] is None
    assert ledger["trainer"]["status"] == "completed"
    assert ledger["trainer"]["returncode"] == 0
    assert ledger["trainer"]["adapter_exists"] is True
    assert ledger["trainer"]["result"]["train_loss"] == 0.42
    assert ledger["trainer"]["adapter_config"]["r_mlp"] == 8
    assert ledger["settings"] == {
        "steps": 7,
        "batch_size": 2,
        "lr": 1e-3,
        "content_weight": 2.0,
        "key_weight": 3.0,
        "value_weight": 4.0,
        "seed": 99,
        "backend": "gpu",
        "max_eval_rows": None,
        "dry_run": False,
    }
    assert ledger["evaluation"]["status"] == "completed"
    assert ledger["evaluation"]["returncode"] == 0
    assert ledger["evaluation"]["rows_available"] == 6
    assert ledger["evaluation"]["rows_used"] == 6
    assert ledger["evaluation"]["examples"] == 6
    assert ledger["evaluation"]["adapter"]["loss"] == 1.1
    assert ledger["evaluation"]["adapter"]["content_loss"] == 1.2
    assert ledger["evaluation"]["exact_match_rate"] == 0.75
    assert ledger["evaluation"]["key_copy_rate"] == 0.5
    assert ledger["evaluation"]["loss"] == 1.1
    assert ledger["evaluation"]["content_loss"] == 1.2
    assert ledger["evaluation"]["base"]["loss"] == 3.5
    assert ledger["evaluation"]["delta"]["loss"] == pytest.approx(-2.4)
    assert ledger["evaluation"]["delta"]["content_loss"] == pytest.approx(-1.8)
    assert ledger["evaluation"]["delta"]["exact_match_rate"] == 0.75
    assert ledger["evaluation"]["delta"]["key_copy_rate"] == 0.5
    assert ledger["evaluation"]["per_family"]["distill_copy_2short"][
        "adapter"]["loss"] == 1.1

    calls = _calls(log)
    assert [call["role"] for call in calls] == [
        "teacher", "trainer", "evaluator"]
    teacher_flags = _flags(calls[0]["argv"])
    assert teacher_flags["source"] == str(paths["source"].resolve())
    assert Path(teacher_flags["out-dir"]) == paths["teacher_data"].parent
    assert Path(teacher_flags["result"]).name == "manifest.json"
    trainer_flags = _flags(calls[1]["argv"])
    assert Path(trainer_flags["train"]) == paths["teacher_data"]
    assert Path(trainer_flags["eval"]) == paths["eval"]
    assert Path(trainer_flags["out"]) == paths["out"]
    assert trainer_flags["base"] == str(paths["base"].resolve())

    target = write_ledger(ledger, paths["result"])
    written = json.loads(target.read_text(encoding="utf-8"))
    assert written["schema"] == "distill_cycle_ledger_v1"
    assert SECRET not in json.dumps(written)


def test_cli_success_path_with_fake_command_strings(tmp_path):
    paths = _paths(tmp_path, eval_rows=3)
    second = _write_jsonl(
        tmp_path / "holdout_eval_2.jsonl", _secret_rows(3))
    teacher, trainer, evaluator, log = _fake_commands(tmp_path)
    code = main([
        "--base", str(paths["base"]),
        "--eval", str(paths["eval"]),
        "--eval", str(second),
        "--train", str(paths["source"]),
        "--teacher-data", str(paths["teacher_data"]),
        "--teacher-command", subprocess.list2cmdline(teacher),
        "--trainer-command", subprocess.list2cmdline(trainer),
        "--evaluator-command", subprocess.list2cmdline(evaluator),
        "--out", str(paths["out"]),
        "--steps", "9",
        "--batch-size", "3",
        "--lr", "5e-4",
        "--content-weight", "2",
        "--key-weight", "3",
        "--value-weight", "4",
        "--seed", "31",
        "--backend", "gpu",
        "--max-eval-rows", "4",
        "--result", str(paths["result"]),
    ])
    assert code == 0
    ledger = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert ledger["status"] == "completed"
    assert ledger["teacher"]["returncode"] == 0
    assert ledger["trainer"]["returncode"] == 0
    assert ledger["evaluation"]["returncode"] == 0
    assert ledger["evaluation"]["rows_available"] == 6
    assert ledger["evaluation"]["rows_used"] == 4
    assert ledger["evaluation"]["examples"] == 4
    assert ledger["settings"]["steps"] == 9
    assert ledger["settings"]["batch_size"] == 3
    assert ledger["settings"]["key_weight"] == 3.0
    assert ledger["settings"]["value_weight"] == 4.0
    assert [call["role"] for call in _calls(log)] == [
        "teacher", "trainer", "evaluator"]
    assert SECRET not in json.dumps(ledger)


def test_cli_failure_still_writes_ledger_and_returns_one(tmp_path):
    paths = _paths(tmp_path)
    teacher, trainer, evaluator, log = _fake_commands(
        tmp_path, teacher_fail=7)
    code = main([
        "--base", str(paths["base"]),
        "--eval", str(paths["eval"]),
        "--train", str(paths["source"]),
        "--teacher-data", str(paths["teacher_data"]),
        "--teacher-command", subprocess.list2cmdline(teacher),
        "--trainer-command", subprocess.list2cmdline(trainer),
        "--evaluator-command", subprocess.list2cmdline(evaluator),
        "--out", str(paths["out"]),
        "--result", str(paths["result"]),
    ])
    assert code == 1
    ledger = json.loads(paths["result"].read_text(encoding="utf-8"))
    assert ledger["status"] == "teacher_failed"
    assert ledger["teacher"]["returncode"] == 7
    assert ledger["trainer"]["status"] == "skipped"
    assert ledger["evaluation"]["status"] == "skipped"
    assert [call["role"] for call in _calls(log)] == ["teacher"]
    assert SECRET not in json.dumps(ledger)


def test_ledger_is_content_free_even_when_children_echo_rows(tmp_path):
    paths = _paths(tmp_path)
    teacher, trainer, evaluator, _ = _fake_commands(tmp_path)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        result_path=str(paths["result"]),
        teacher_data=str(paths["teacher_data"]),
        train=str(paths["source"]),
        teacher_command=teacher,
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    serialized = json.dumps(ledger)
    assert SECRET not in serialized
    for forbidden in ('"prompt"', '"keys"', '"values"'):
        assert forbidden not in serialized
    # ``rows`` is allowed only as an aggregate count, never as scored rows.
    assert '"rows"' not in json.dumps(ledger["trainer"])
    assert '"rows"' not in json.dumps(ledger["evaluation"])
    assert ledger["teacher"]["manifest"]["counts"]["requested"] == 8
    assert "rows_available" in serialized
    assert "rows_used" in serialized
    assert "exact_match_rate" in serialized
    assert "key_copy_rate" in serialized
    assert "content_loss" in serialized


def test_dry_run_writes_synthetic_ledger_without_subprocess(tmp_path):
    marker = tmp_path / "ran.txt"
    script = tmp_path / "marker.py"
    script.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "Path(sys.argv[1]).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8")
    teacher_command = (
        f'"{sys.executable}" "{script}" "{marker}" {{source}}')
    result = tmp_path / "ledger.json"
    code = main([
        "--dry-run",
        "--result", str(result),
        "--out", str(tmp_path / "adapter"),
        "--teacher-command", teacher_command,
        "--teacher-data", "data/teacher.jsonl",
        "--train", "data/source.jsonl",
        "--base", "checkpoints/base",
        "--eval", "data/eval.jsonl",
        "--backend", "gpu",
    ])
    assert code == 0
    assert not marker.exists()
    ledger = json.loads(result.read_text(encoding="utf-8"))
    assert ledger["status"] == "completed"
    assert ledger["dry_run"] is True
    assert ledger["backend"] == "gpu"
    assert ledger["teacher"]["status"] == "dry_run"
    assert ledger["trainer"]["status"] == "dry_run"
    assert ledger["evaluation"]["status"] == "dry_run"
    assert ledger["evaluation"]["synthetic"] is True
    assert ledger["evaluation"]["exact_match_rate"] == 0.4
    assert ledger["evaluation"]["key_copy_rate"] == 0.548
    assert ledger["evaluation"]["loss"] == 0.97
    assert ledger["evaluation"]["content_loss"] == 1.0
    assert str(marker) in ledger["teacher"]["command"]


def test_max_eval_rows_limits_the_heldout_input(tmp_path):
    paths = _paths(tmp_path, eval_rows=6)
    _, trainer, evaluator, log = _fake_commands(tmp_path)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        result_path=str(paths["result"]),
        teacher_data=str(paths["teacher_data"]),
        trainer_command=trainer,
        evaluator_command=evaluator,
        max_eval_rows=3,
    )
    assert ledger["status"] == "completed"
    assert ledger["evaluation"]["max_eval_rows"] == 3
    assert ledger["evaluation"]["rows_available"] == 6
    assert ledger["evaluation"]["rows_used"] == 3
    assert ledger["evaluation"]["examples"] == 3
    calls = _calls(log)
    evaluator_flags = _flags(calls[1]["argv"])
    assert evaluator_flags["data"] != str(paths["eval"].resolve())
    assert "holdout_eval.limited.jsonl" in evaluator_flags["data"]
    trainer_flags = _flags(calls[0]["argv"])
    assert Path(trainer_flags["eval"]) == paths["eval"]


def test_multiple_eval_sets_are_merged_for_the_evaluator(tmp_path):
    paths = _paths(tmp_path, eval_rows=3)
    second = _write_jsonl(
        tmp_path / "holdout_eval_2.jsonl", _secret_rows(3))
    _, trainer, evaluator, log = _fake_commands(tmp_path)
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=[str(paths["eval"]), str(second)],
        out_dir=str(paths["out"]),
        teacher_data=str(paths["teacher_data"]),
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "completed"
    assert ledger["evaluation"]["rows_available"] == 6
    assert ledger["evaluation"]["rows_used"] == 6
    assert ledger["evaluation"]["examples"] == 6
    assert len(ledger["evaluation"]["sources"]) == 2
    assert len(ledger["paths"]["evals"]) == 2
    evaluator_flags = _flags(_calls(log)[1]["argv"])
    assert "holdout_eval.combined.jsonl" in evaluator_flags["data"]
    trainer_flags = _flags(_calls(log)[0]["argv"])
    assert Path(trainer_flags["eval"]) == Path(evaluator_flags["data"])


def test_missing_eval_marks_eval_failed_without_running_evaluator(tmp_path):
    paths = _paths(tmp_path)
    _, trainer, evaluator, log = _fake_commands(tmp_path)
    missing_eval = tmp_path / "missing_eval.jsonl"
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(missing_eval),
        out_dir=str(paths["out"]),
        teacher_data=str(paths["teacher_data"]),
        trainer_command=trainer,
        evaluator_command=evaluator,
    )
    assert ledger["status"] == "eval_failed"
    assert ledger["trainer"]["status"] == "skipped"
    assert ledger["evaluation"]["status"] == "failed"
    assert ledger["evaluation"]["reason"] == "eval_missing"
    assert ledger["evaluation"]["returncode"] is None
    assert _calls(log) == []


def test_missing_teacher_data_without_command_is_teacher_failed(tmp_path):
    paths = _paths(tmp_path)
    missing = tmp_path / "missing" / "train_distill.jsonl"
    ledger = run_cycle(
        base=str(paths["base"]),
        eval_path=str(paths["eval"]),
        out_dir=str(paths["out"]),
        teacher_data=str(missing),
    )
    assert ledger["status"] == "teacher_failed"
    assert ledger["teacher"]["status"] == "failed"
    assert ledger["teacher"]["reason"] == "teacher_data_missing"
    assert ledger["trainer"]["status"] == "skipped"
    assert ledger["evaluation"]["status"] == "skipped"


def test_manifest_reader_keeps_only_numeric_counts(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({
        "rows": 10,
        "accepted": 9,
        "note": SECRET,
        "counts": {"attempts": 11, "secret": SECRET, "nested": {
            "kept": 2}},
        "paths": {"data": SECRET},
        "summary": {"rejected": 1, "seed": 23, "prompt": SECRET},
    }), encoding="utf-8")
    record = read_teacher_manifest(manifest)
    assert record["found"] is True
    assert record["error"] is None
    assert record["counts"] == {
        "rows": 10,
        "accepted": 9,
        "counts.attempts": 11,
        "counts.nested.kept": 2,
        "summary.rejected": 1,
    }
    assert SECRET not in json.dumps(record)
