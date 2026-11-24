import json
import sys
from pathlib import Path

import pytest

from research.run_qkv_capacity_ablation import (
    DEFAULT_ARMS,
    DEFAULT_SEEDS,
    AblationError,
    aggregate_metric,
    arm_geometry,
    build_evaluator_command,
    build_summary,
    build_trainer_command,
    expand_runs,
    main,
    parse_arms,
    parse_seeds,
    rows_digest,
    select_eval_rows,
)

SECRET_PROMPT = "SECRET_PROMPT_TEXT"
SECRET_ANSWER = "SECRET_ANSWER_TEXT"

TRAINER_KWARGS = {
    "base": "base",
    "train": "train.jsonl",
    "eval_path": "eval.jsonl",
    "out": "adapter",
    "result": "result.json",
    "comparison": "comparison.json",
    "steps": 4000,
    "batch_size": 4,
    "lr": 3e-4,
    "content_weight": 2.0,
    "key_weight": 3.0,
    "value_weight": 2.0,
    "rank": 8,
    "alpha": 8,
    "rank_attn": 4,
    "alpha_attn": 8,
    "qkv_rank": 4,
    "qkv_alpha": 8,
}


FAKE_BACKEND = r'''
import argparse
import json
import sys
from pathlib import Path

SECRET_PROMPT = "SECRET_PROMPT_TEXT"
SECRET_ANSWER = "SECRET_ANSWER_TEXT"


def parse_flags(items):
    flags = {}
    index = 0
    while index < len(items):
        item = items[index]
        if item.startswith("--"):
            if index + 1 < len(items) and not items[index + 1].startswith("--"):
                flags[item[2:]] = items[index + 1]
                index += 2
            else:
                flags[item[2:]] = True
                index += 1
        else:
            index += 1
    return flags


parser = argparse.ArgumentParser(allow_abbrev=False)
parser.add_argument("--role", required=True)
parser.add_argument("--log", required=True)
parser.add_argument("--fail-role", default="")
parser.add_argument("--fail-seed", type=int, default=-1)
options, extra = parser.parse_known_args()
flags = parse_flags(extra)
with open(options.log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({"role": options.role, "argv": extra}) + "\n")

if options.role == "trainer":
    seed = int(flags["seed"])
    if options.fail_role == "trainer" and seed == options.fail_seed:
        sys.exit(1)
    adapter = Path(flags["out"])
    adapter.mkdir(parents=True, exist_ok=True)
    (adapter / "adapter.pt").write_bytes(b"fake-adapter")
    (adapter / "config.json").write_text(json.dumps({
        "r_mlp": int(flags["rank"]),
        "alpha_mlp": int(flags["alpha"]),
        "r_attn": int(flags["rank-attn"]),
        "alpha_attn": int(flags["alpha-attn"]),
        "r_qkv": int(flags["qkv-rank"]),
        "alpha_qkv": int(flags["qkv-alpha"]),
        "prompt": SECRET_PROMPT,
    }), encoding="utf-8")
    result = Path(flags["result"])
    result.parent.mkdir(parents=True, exist_ok=True)
    result.write_text(json.dumps({
        "train_loss": 0.5,
        "train_steps": int(flags["steps"]),
        "optimizer_steps": 3,
        "lora_parameters": 500000 if int(flags["qkv-rank"]) else 400000,
        "rank_mlp": int(flags["rank"]),
        "alpha_mlp": int(flags["alpha"]),
        "rank_attn": int(flags["rank-attn"]),
        "alpha_attn": int(flags["alpha-attn"]),
        "rank_qkv": int(flags["qkv-rank"]),
        "alpha_qkv": int(flags["qkv-alpha"]),
        "batch_size": int(flags["batch-size"]),
        "effective_lr": float(flags["lr"]),
        "content_weight": float(flags["content-weight"]),
        "key_weight": float(flags["key-weight"]),
        "value_weight": float(flags["value-weight"]),
        "prompt": SECRET_PROMPT,
    }), encoding="utf-8")
    Path(flags["comparison"]).write_text(json.dumps({"models": [
        {"label": "base", "metrics": {"rows": [
            {"user": SECRET_PROMPT, "assistant": SECRET_ANSWER}]}},
        {"label": "adapter", "metrics": {"rows": [
            {"user": SECRET_PROMPT, "assistant": SECRET_ANSWER}]}},
    ]}), encoding="utf-8")
    sys.exit(0)

adapter_path = Path(flags["adapters"])
arm, seed_text = adapter_path.parent.name.rsplit("_s", 1)
seed = int(seed_text)
if options.fail_role == "evaluator" and seed == options.fail_seed:
    sys.exit(1)
rows = [line for line in Path(flags["data"]).read_text(
    encoding="utf-8").splitlines() if line.strip()]
with open(options.log, "a", encoding="utf-8") as handle:
    handle.write(json.dumps({
        "role": "evaluator-rows", "arm": arm, "seed": seed,
        "rows": len(rows)}) + "\n")
base_rate = 0.80 if arm == "qkv4" else 0.20
variation = (seed % 10) * 0.001
payload = {"examples": len(rows), "models": [
    {"label": "base", "path": flags["base"], "metrics": {
        "loss": 3.0, "ppl": 20.0, "tokens": 40, "content_loss": 3.2,
        "rows": [{"user": SECRET_PROMPT, "assistant": SECRET_ANSWER}]}},
    {"label": arm, "path": flags["adapters"], "metrics": {
        "loss": 1.0 + variation, "ppl": 3.0, "tokens": 40,
        "content_loss": 0.9 + variation,
        "exact_match": {
            "examples": len(rows), "matches": 1,
            "rate": base_rate + variation},
        "key_copy": {
            "examples": len(rows), "matches": 1,
            "rate": base_rate + 0.1 + variation,
            "key_recall": base_rate + 0.05},
        "rows": [{"user": SECRET_PROMPT, "assistant": SECRET_ANSWER}]}},
]}
target = Path(flags["out"])
target.parent.mkdir(parents=True, exist_ok=True)
target.write_text(json.dumps(payload), encoding="utf-8")
sys.exit(0)
'''


def _fake_backend(tmp_path):
    script = tmp_path / "fake_backend.py"
    script.write_text(FAKE_BACKEND, encoding="utf-8")
    log = tmp_path / "calls.jsonl"
    trainer = [sys.executable, str(script), "--role", "trainer",
               "--log", str(log)]
    evaluator = [sys.executable, str(script), "--role", "evaluator",
                 "--log", str(log)]
    return trainer, evaluator, log


def _log_entries(log):
    if not Path(log).is_file():
        return []
    return [json.loads(line) for line in Path(log).read_text(
        encoding="utf-8").splitlines() if line.strip()]


def _flag(command, name):
    return command[command.index(name) + 1]


def _write_rows(path, count, prefix):
    with Path(path).open("w", encoding="utf-8", newline="\n") as handle:
        for index in range(count):
            handle.write(json.dumps({
                "user": f"{SECRET_PROMPT} {prefix}{index}",
                "assistant": SECRET_ANSWER,
                "family": "secret",
            }) + "\n")


def _main_args(tmp_path, *, max_eval_rows=None, extra=()):
    base = tmp_path / "base"
    base.mkdir(exist_ok=True)
    train = tmp_path / "train.jsonl"
    eval_path = tmp_path / "eval.jsonl"
    if not train.exists():
        _write_rows(train, 12, "train")
    if not eval_path.exists():
        _write_rows(eval_path, 10, "eval")
    args = [
        "--base", str(base),
        "--train", str(train),
        "--eval", str(eval_path),
        "--out", str(tmp_path / "out"),
        "--result", str(tmp_path / "result.json"),
        "--arms", "qkv4,mlp",
        "--seeds", "23,29",
        "--steps", "10",
        "--batch-size", "2",
        "--lr", "0.0003",
    ]
    if max_eval_rows is not None:
        args.extend(["--max-eval-rows", str(max_eval_rows)])
    args.extend(extra)
    return args


def _settings(**overrides):
    settings = {
        "arms": ["qkv4", "mlp"],
        "seeds": [23, 29],
        "steps": 4000,
        "batch_size": 4,
        "lr": 3e-4,
        "content_weight": 2.0,
        "key_weight": 3.0,
        "value_weight": 2.0,
        "rank": 8,
        "alpha": 8,
        "rank_attn": 4,
        "alpha_attn": 8,
        "qkv_rank": 4,
        "qkv_alpha": 8,
        "max_eval_rows": None,
    }
    settings.update(overrides)
    return settings


def _record(arm, seed, exact, key_copy, *, status="completed"):
    return {
        "arm": arm,
        "seed": seed,
        "status": status,
        "returncode": 0,
        "elapsed_seconds": 1.0,
        "adapter": f"out/{arm}_s{seed}/adapter",
        "exact": exact,
        "key_copy": key_copy,
        "key_recall": key_copy,
        "loss": 1.0,
        "content_loss": 0.9,
        "eval_examples": 10,
        "train_loss": 0.5,
        "lora_parameters": 500000 if arm == "qkv4" else 400000,
        "trainer": {
            "status": "completed", "returncode": 0,
            "elapsed_seconds": 0.5, "command": [], "result": {},
        },
        "evaluator": {
            "status": "completed", "returncode": 0,
            "elapsed_seconds": 0.5, "command": [], "metrics": {},
        },
    }


def _summary(records, **overrides):
    return build_summary(
        records=records, settings=_settings(**overrides),
        arms=["qkv4", "mlp"], base="base", train="train.jsonl",
        eval_path="eval.jsonl",
        eval_subset={
            "applied": False, "selection": "all", "seed": None,
            "rows": 10, "total_rows": 10,
        })


def test_parse_arms_and_seeds_dedupe_and_reject_unknown_values():
    assert parse_arms("qkv4, mlp,qkv4") == ["qkv4", "mlp"]
    assert parse_seeds("23, 29,23") == [23, 29]
    with pytest.raises(AblationError, match="unknown arm"):
        parse_arms("qkv8")
    with pytest.raises(AblationError, match="at least one arm"):
        parse_arms(" , ")
    with pytest.raises(AblationError, match="invalid seed"):
        parse_seeds("23,abc")
    with pytest.raises(AblationError, match="at least one seed"):
        parse_seeds(" , ")


def test_expand_runs_covers_every_arm_and_seed_in_order():
    runs = expand_runs(["qkv4", "mlp"], [23, 29])
    assert [(run["arm"], run["seed"]) for run in runs] == [
        ("qkv4", 23), ("qkv4", 29), ("mlp", 23), ("mlp", 29)]
    default_runs = expand_runs(
        parse_arms(DEFAULT_ARMS), parse_seeds(DEFAULT_SEEDS))
    assert len(default_runs) == 6
    assert {run["arm"] for run in default_runs} == {"qkv4", "mlp"}
    assert {run["seed"] for run in default_runs} == {23, 29, 37}


def test_arm_geometry_disables_qkv_for_the_mlp_arm():
    qkv = arm_geometry(
        arm="qkv4", rank=8, alpha=8, rank_attn=4, alpha_attn=8,
        qkv_rank=4, qkv_alpha=8)
    mlp = arm_geometry(
        arm="mlp", rank=8, alpha=8, rank_attn=4, alpha_attn=8,
        qkv_rank=4, qkv_alpha=8)
    assert qkv["rank_qkv"] == 4 and qkv["alpha_qkv"] == 8
    assert mlp["rank_qkv"] == 0 and mlp["alpha_qkv"] == 0
    assert qkv["rank_mlp"] == mlp["rank_mlp"] == 8
    assert qkv["rank_attn"] == mlp["rank_attn"] == 4


def test_trainer_command_passes_qkv_flags_for_qkv_arm():
    command = build_trainer_command(
        arm="qkv4", seed=23, command_prefix=["python", "trainer.py"],
        **TRAINER_KWARGS)
    assert command[:2] == ["python", "trainer.py"]
    assert _flag(command, "--qkv-rank") == "4"
    assert _flag(command, "--qkv-alpha") == "8"
    assert _flag(command, "--rank") == "8"
    assert _flag(command, "--alpha") == "8"
    assert _flag(command, "--rank-attn") == "4"
    assert _flag(command, "--alpha-attn") == "8"
    assert _flag(command, "--steps") == "4000"
    assert _flag(command, "--batch-size") == "4"
    assert _flag(command, "--content-weight") == "2.0"
    assert _flag(command, "--key-weight") == "3.0"
    assert _flag(command, "--value-weight") == "2.0"
    assert _flag(command, "--seed") == "23"
    assert _flag(command, "--train") == "train.jsonl"
    assert _flag(command, "--eval") == "eval.jsonl"


def test_trainer_command_disables_qkv_for_mlp_arm():
    command = build_trainer_command(
        arm="mlp", seed=29, command_prefix=["python", "trainer.py"],
        **TRAINER_KWARGS)
    assert _flag(command, "--qkv-rank") == "0"
    assert _flag(command, "--qkv-alpha") == "0"
    assert _flag(command, "--rank") == "8"
    assert _flag(command, "--rank-attn") == "4"
    assert _flag(command, "--seed") == "29"


def test_evaluator_command_requests_exact_match_and_key_copy():
    command = build_evaluator_command(
        base="base", data="eval.jsonl", adapter="adapter",
        out="eval.json", command_prefix=["python", "compare.py"])
    assert command[:2] == ["python", "compare.py"]
    assert "--exact-match" in command
    assert "--key-copy" in command
    assert _flag(command, "--data") == "eval.jsonl"
    assert _flag(command, "--adapters") == "adapter"
    assert _flag(command, "--out") == "eval.json"


def test_aggregate_metric_reports_mean_std_min_max_and_interval():
    aggregate = aggregate_metric([0.1, 0.2, 0.3])
    assert aggregate["count"] == 3
    assert aggregate["mean"] == pytest.approx(0.2)
    assert aggregate["std"] == pytest.approx(0.1)
    assert aggregate["min"] == pytest.approx(0.1)
    assert aggregate["max"] == pytest.approx(0.3)
    low, high = aggregate["ci95"]
    assert low < aggregate["mean"] < high
    assert aggregate_metric([]) is None
    single = aggregate_metric([0.5, None, "ignored", True])
    assert single["count"] == 1
    assert single["std"] is None
    assert single["ci95"] is None


def test_select_eval_rows_is_deterministic_and_keeps_order():
    rows = [{"id": index} for index in range(20)]
    first = select_eval_rows(rows, 5, seed=7)
    second = select_eval_rows(rows, 5, seed=7)
    assert first == second
    assert len(first) == 5
    assert len({row["id"] for row in first}) == 5
    assert [row["id"] for row in first] == sorted(
        row["id"] for row in first)
    assert select_eval_rows(rows, 20) == rows
    assert select_eval_rows(rows, None) == rows


def test_rows_digest_is_stable_and_content_sensitive():
    rows = [{"id": 1, "user": SECRET_PROMPT}]
    assert rows_digest(rows) == rows_digest(list(rows))
    assert rows_digest(rows) != rows_digest([{"id": 2, "user": SECRET_PROMPT}])


def test_summary_aggregates_per_arm_and_seed():
    summary = _summary([
        _record("qkv4", 23, 0.50, 0.60),
        _record("qkv4", 29, 0.52, 0.62),
        _record("mlp", 23, 0.40, 0.50),
        _record("mlp", 29, 0.42, 0.52),
    ])
    qkv = summary["arms"]["qkv4"]["aggregate"]["exact"]
    assert summary["arms"]["qkv4"]["completed"] == 2
    assert qkv["count"] == 2
    assert qkv["mean"] == pytest.approx(0.51)
    assert qkv["std"] == pytest.approx(0.014142, abs=1e-5)
    assert qkv["min"] == pytest.approx(0.50)
    assert qkv["max"] == pytest.approx(0.52)
    assert len(summary["arms"]["qkv4"]["runs"]) == 2
    assert [run["seed"] for run in summary["arms"]["qkv4"]["runs"]] == [23, 29]


def test_summary_reports_no_claim_when_intervals_overlap():
    summary = _summary([
        _record("qkv4", 23, 0.50, 0.60),
        _record("qkv4", 29, 0.52, 0.62),
        _record("mlp", 23, 0.49, 0.59),
        _record("mlp", 29, 0.51, 0.61),
    ])
    verdict = summary["verdict"]
    assert verdict["claim"] == "inconclusive"
    assert verdict["qkv_improves"] is None
    assert verdict["metrics"]["exact"]["interval_overlap"] is True
    assert verdict["metrics"]["exact"]["winner"] is None


def test_summary_reports_qkv_improvement_when_intervals_do_not_overlap():
    summary = _summary([
        _record("qkv4", 23, 0.80, 0.90),
        _record("qkv4", 29, 0.82, 0.92),
        _record("mlp", 23, 0.20, 0.30),
        _record("mlp", 29, 0.22, 0.32),
    ])
    verdict = summary["verdict"]
    assert verdict["claim"] == "qkv_improves"
    assert verdict["qkv_improves"] is True
    assert verdict["metrics"]["exact"]["winner"] == "qkv4"
    assert verdict["metrics"]["exact"]["interval_overlap"] is False
    assert summary["fair_comparison"][
        "qkv_adds_trainable_parameters"] is True


def test_summary_is_inconclusive_when_metrics_disagree():
    summary = _summary([
        _record("qkv4", 23, 0.80, 0.30),
        _record("qkv4", 29, 0.82, 0.32),
        _record("mlp", 23, 0.20, 0.30),
        _record("mlp", 29, 0.22, 0.32),
    ])
    assert summary["verdict"]["claim"] == "inconclusive"
    assert summary["verdict"]["qkv_improves"] is None


def test_summary_marks_partial_runs_and_ignores_failed_seeds():
    failed = _record("qkv4", 29, None, None, status="trainer_failed")
    failed["trainer"]["status"] = "failed"
    failed["evaluator"]["status"] = "skipped"
    summary = _summary([
        _record("qkv4", 23, 0.50, 0.60),
        failed,
        _record("mlp", 23, 0.40, 0.50),
        _record("mlp", 29, 0.42, 0.52),
    ])
    assert summary["status"] == "partial"
    assert summary["arms"]["qkv4"]["completed"] == 1
    assert summary["arms"]["qkv4"]["failed"] == 1
    assert summary["arms"]["qkv4"]["aggregate"]["exact"]["count"] == 1
    assert summary["verdict"]["claim"] == "inconclusive"


def test_summary_drops_unknown_record_fields_and_row_text():
    record = _record("qkv4", 23, 0.50, 0.60)
    record["user"] = SECRET_PROMPT
    record["trainer"]["prompt"] = SECRET_PROMPT
    record["trainer"]["result"]["prompt"] = SECRET_PROMPT
    record["evaluator"]["prompt"] = SECRET_PROMPT
    record["evaluator"]["metrics"]["rows"] = [
        {"user": SECRET_PROMPT, "assistant": SECRET_ANSWER}]
    summary = _summary([record, _record("mlp", 23, 0.40, 0.50)])
    text = json.dumps(summary)
    assert SECRET_PROMPT not in text
    assert SECRET_ANSWER not in text
    public = summary["arms"]["qkv4"]["runs"][0]
    assert "user" not in public
    assert "prompt" not in public["trainer"]
    assert "prompt" not in public["trainer"]["result"]
    assert public["evaluator"]["metrics"] == {}


def test_main_dry_run_prints_plan_without_writing(tmp_path, capsys):
    result = tmp_path / "result.json"
    exit_code = main(
        _main_args(tmp_path, extra=["--dry-run"]),
        trainer_command=["python", "fake_trainer.py"],
        evaluator_command=["python", "fake_evaluator.py"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dry_run"] is True
    assert len(payload["runs"]) == 4
    qkv_run = next(run for run in payload["runs"] if run["arm"] == "qkv4")
    mlp_run = next(run for run in payload["runs"] if run["arm"] == "mlp")
    assert _flag(qkv_run["trainer_command"], "--qkv-rank") == "4"
    assert _flag(mlp_run["trainer_command"], "--qkv-rank") == "0"
    assert not result.exists()
    assert not (tmp_path / "out").exists()


def test_main_runs_all_arms_and_writes_content_free_reports(
        tmp_path, capsys):
    trainer, evaluator, log = _fake_backend(tmp_path)
    exit_code = main(
        _main_args(tmp_path, max_eval_rows=4),
        trainer_command=trainer, evaluator_command=evaluator)
    assert exit_code == 0
    result = tmp_path / "result.json"
    markdown = tmp_path / "result.md"
    assert result.is_file() and markdown.is_file()
    summary = json.loads(result.read_text(encoding="utf-8"))
    assert summary["schema"] == "atomzero.qkv_capacity_ablation.v1"
    assert summary["status"] == "completed"
    assert summary["content_free"] is True
    assert summary["eval_rows"] == 10
    assert summary["eval_rows_used"] == 4
    assert summary["eval_subset"]["applied"] is True
    assert summary["arms"]["qkv4"]["completed"] == 2
    assert summary["arms"]["mlp"]["completed"] == 2
    assert summary["arms"]["qkv4"]["aggregate"]["exact"]["count"] == 2
    assert summary["verdict"]["claim"] == "qkv_improves"
    assert summary["fair_comparison"][
        "qkv_adds_trainable_parameters"] is True
    qkv_run = summary["arms"]["qkv4"]["runs"][0]
    mlp_run = summary["arms"]["mlp"]["runs"][0]
    assert qkv_run["trainer"]["result"]["lora_parameters"] == 500000
    assert mlp_run["trainer"]["result"]["lora_parameters"] == 400000
    text = result.read_text(encoding="utf-8") + markdown.read_text(
        encoding="utf-8")
    assert SECRET_PROMPT not in text
    assert SECRET_ANSWER not in text
    row_logs = [entry for entry in _log_entries(log)
                if entry["role"] == "evaluator-rows"]
    assert len(row_logs) == 4
    assert {entry["rows"] for entry in row_logs} == {4}
    assert not list((tmp_path / "out").rglob("eval.json"))
    assert not list((tmp_path / "out").rglob("*comparison.json"))


def test_main_records_evaluator_failure_after_training(tmp_path, capsys):
    trainer, evaluator, _ = _fake_backend(tmp_path)
    evaluator = evaluator + [
        "--fail-role", "evaluator", "--fail-seed", "29"]
    exit_code = main(
        _main_args(tmp_path),
        trainer_command=trainer, evaluator_command=evaluator)
    assert exit_code == 1
    summary = json.loads(
        (tmp_path / "result.json").read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    qkv_runs = summary["arms"]["qkv4"]["runs"]
    failed = next(run for run in qkv_runs if run["seed"] == 29)
    assert failed["status"] == "eval_failed"
    assert failed["returncode"] == 1
    assert failed["trainer"]["status"] == "completed"
    assert failed["evaluator"]["status"] == "failed"
    assert failed["exact"] is None


def test_main_records_trainer_failure_and_skips_evaluation(
        tmp_path, capsys):
    trainer, evaluator, log = _fake_backend(tmp_path)
    trainer = trainer + ["--fail-role", "trainer", "--fail-seed", "29"]
    exit_code = main(
        _main_args(tmp_path),
        trainer_command=trainer, evaluator_command=evaluator)
    assert exit_code == 1
    summary = json.loads(
        (tmp_path / "result.json").read_text(encoding="utf-8"))
    assert summary["status"] == "partial"
    qkv_runs = summary["arms"]["qkv4"]["runs"]
    failed = next(run for run in qkv_runs if run["seed"] == 29)
    assert failed["status"] == "trainer_failed"
    assert failed["returncode"] == 1
    assert failed["evaluator"]["status"] == "skipped"
    assert failed["exact"] is None
    assert summary["verdict"]["claim"] == "inconclusive"
    row_logs = [entry for entry in _log_entries(log)
                if entry["role"] == "evaluator-rows"]
    assert len(row_logs) == 2
    assert {entry["seed"] for entry in row_logs} == {23}
