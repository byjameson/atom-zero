from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from research.capability_scorecard import (
    CRITERIA,
    build_scorecard,
    classify_document,
    render_markdown,
)

ROOT = Path(__file__).resolve().parents[1]

CURRICULUM = {
    "first_attempt": {
        "gate": "fail",
        "exact_match": 0.37,
        "key_copy": 0.55,
        "promotion": False,
        "reasons": ["content_loss_not_improved", "content_ppl_regression"],
    },
    "successful_attempt": {
        "gate": "pass",
        "exact_match": 0.53,
        "key_copy": 0.60,
        "promotion": True,
    },
}

TOOL_BENCHMARK = {
    "dataset": "data/multi_short_v1/stage_4short/eval.jsonl",
    "examples": 200,
    "exact_match": 1.0,
    "key_copy_rate": 1.0,
    "value_copy_rate": 1.0,
    "formats": {"json": 100, "kv": 100},
    "note": "deterministic runtime transcription; not a model capability",
}

HELDOUT = {
    "adapter": "multi_recover",
    "eval_examples_per_task": 30,
    "tasks": {
        "arithmetic": {
            "base_exact": 0.0,
            "adapter_exact": 0.0,
            "base_content_loss": 4.391,
            "adapter_content_loss": 5.142,
        },
        "city_yaml": {
            "base_exact": 0.0,
            "adapter_exact": 0.0,
            "base_content_loss": 3.484,
            "adapter_content_loss": 3.632,
        },
        "city_river": {
            "base_exact": 0.0,
            "adapter_exact": 0.0,
            "base_content_loss": 4.794,
            "adapter_content_loss": 3.243,
        },
    },
}


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT), str(ROOT / "src")])
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_scorecard(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(ROOT / "research" / "capability_scorecard.py"), *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_env(),
        check=False,
    )


def _criteria(card: dict) -> dict[str, dict]:
    return {item["id"]: item for item in card["criteria"]}


def test_six_criteria_present_and_missing_metrics_unknown():
    card = build_scorecard([])
    assert [item[0] for item in CRITERIA] == [item["id"] for item in card["criteria"]]
    assert len(card["criteria"]) == 6
    assert all(item["status"] == "unknown" for item in card["criteria"])
    assert all(item["score"] is None for item in card["criteria"])
    assert card["summary"]["aggregate_score"] is None
    assert card["summary"]["status_counts"]["unknown"] == 6
    assert card["held_out_tasks"]["runtime_copy_tool"]["status"] == "unknown"
    assert card["held_out_tasks"]["copy_transfer"]["city_river"]["status"] == "unknown"


def test_partial_curriculum_and_unknown_transfer():
    card = build_scorecard(
        [("curriculum.json", CURRICULUM)],
        answerability_flags={
            "brier": 0.233,
            "ece": 0.064,
            "refusal_recall": 0.546,
            "refusal_precision": 0.631,
            "answerable_coverage": 0.699,
            "n": 200,
            "answerability_specs_failed": 2,
            "answerability_specs_total": 2,
        },
    )
    criteria = _criteria(card)
    assert criteria["gap_discovery"]["status"] == "partial"
    assert criteria["gap_discovery"]["score"] == 0.35
    assert criteria["learning_policy"]["status"] == "partial"
    assert criteria["learning_policy"]["score"] == 0.6
    assert criteria["retention"]["status"] == "unknown"
    assert criteria["transfer"]["status"] == "unknown"
    assert criteria["meta_learning"]["status"] == "unknown"
    assert criteria["low_cost"]["status"] == "unknown"
    assert "2/2 answerability specs FAIL" in criteria["gap_discovery"]["evidence"]
    assert "a recorded candidate was rejected" in " ".join(
        criteria["learning_policy"]["evidence"]
    )


def test_copy_tool_is_not_counted_as_model_transfer():
    assert classify_document(TOOL_BENCHMARK) == "copy_tool"
    card = build_scorecard([("tool_benchmark.json", TOOL_BENCHMARK)])
    transfer = _criteria(card)["transfer"]
    assert transfer["status"] == "unknown"
    assert transfer["score"] is None
    assert "tool use" in " ".join(transfer["evidence"]).lower()
    held = card["held_out_tasks"]
    assert held["runtime_copy_tool"]["adapter_exact"] == 1.0
    assert held["runtime_copy_tool"]["counts_as_model_transfer"] is False
    assert held["runtime_copy_tool"]["status"] == "met"
    assert held["tool_use_is_model_transfer"] is False
    assert held["copy_transfer"]["city_river"]["status"] == "unknown"
    assert held["copy_transfer"]["best_model_exact"] is None


def test_heldout_transfer_evidence_is_not_met_and_cited():
    card = build_scorecard(
        [("heldout_task_transfer_summary.json", HELDOUT)],
    )
    transfer = _criteria(card)["transfer"]
    assert transfer["status"] == "not_met"
    assert transfer["score"] == 0.0
    evidence = " ".join(transfer["evidence"])
    assert "held-out arithmetic: base 0.000 / adapter 0.000 exact" in evidence
    assert "content loss 4.391 -> 5.142 (adapter worse)" in evidence
    assert "held-out city_yaml" in evidence
    assert "held-out city_river" in evidence
    assert "heldout_task_transfer_summary.json" in transfer["sources"]
    tasks = card["held_out_tasks"]
    assert tasks["arithmetic_yaml"]["arithmetic"]["status"] == "not_met"
    assert tasks["arithmetic_yaml"]["city_yaml"]["status"] == "not_met"
    assert (
        tasks["arithmetic_yaml"]["arithmetic"]["content_loss_direction"]
        == "adapter worse"
    )
    assert tasks["arithmetic_yaml"]["city_yaml"]["content_loss_direction"] == (
        "adapter worse"
    )
    assert tasks["arithmetic_yaml"]["counts_as_model_transfer"] is True


def test_hardware_statuses():
    def low(card):
        return _criteria(card)["low_cost"]

    inside = low(
        build_scorecard([], hardware={"cpu": 8, "ram_gb": 16, "disk_gb": 40})
    )
    assert inside["status"] == "met"
    assert inside["score"] == 1.0
    over = low(
        build_scorecard([], hardware={"cpu": 16, "ram_gb": 32, "disk_gb": 80})
    )
    assert over["status"] == "partial"
    assert over["score"] == 0.5
    partial = low(build_scorecard([], hardware={"cpu": 8}))
    assert partial["status"] == "partial"
    assert partial["score"] == 0.8
    assert low(build_scorecard([]))["status"] == "unknown"


def test_output_is_deterministic_regardless_of_input_order():
    documents = [
        ("curriculum.json", CURRICULUM),
        ("heldout_task_transfer_summary.json", HELDOUT),
        ("tool_benchmark.json", TOOL_BENCHMARK),
    ]
    hardware = {"cpu": 8, "ram_gb": 16, "disk_gb": 40}
    first = build_scorecard(documents, hardware=hardware)
    second = build_scorecard(list(reversed(documents)), hardware=hardware)
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert render_markdown(first) == render_markdown(second)
    assert json.dumps(first, sort_keys=True) == json.dumps(first, sort_keys=True)


def test_no_prompt_or_answer_text_in_output():
    leaky = {
        **CURRICULUM,
        "prompt": "SECRET_PROMPT_123",
        "answer": "SECRET_ANSWER_456",
        "user": "SECRET_USER_789",
        "nested": {"completion": "SECRET_COMPLETION_000"},
        "first_attempt": {
            **CURRICULUM["first_attempt"],
            "reasons": ["SECRET reason text"],
        },
    }
    card = build_scorecard([("leaky.json", leaky)])
    blob = json.dumps(card, sort_keys=True) + render_markdown(card)
    assert "SECRET" not in blob

    def walk(value):
        if isinstance(value, dict):
            for key, child in value.items():
                assert key.lower() not in {
                    "prompt",
                    "answer",
                    "completion",
                    "user",
                    "assistant",
                }
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            assert "\n" not in value
            assert len(value) <= 400

    walk(card)


def test_known_result_files_scorecard():
    paths = [
        "research/results/self_improve_curriculum_summary.json",
        "research/results/multi_recover_summary.json",
        "research/results/multi_key_0.30_summary.json",
        "research/results/weak_stage_rehearsal.json",
        "research/results/weak_stage_rehearsal_gentle.json",
        "research/results/copy_curriculum_summary.json",
        "research/results/copy_tool_benchmark_summary.json",
        "research/results/heldout_task_transfer_summary.json",
        "benchmarks/results/answerability_hybrid/answerability_profile.json",
    ]
    documents = []
    for relative in paths:
        data = json.loads((ROOT / relative).read_text(encoding="utf-8"))
        documents.append((relative, data))
    card = build_scorecard(
        documents,
        hardware={"cpu": 8, "ram_gb": 16, "disk_gb": 40},
        answerability_flags={
            "answerability_specs_failed": 2,
            "answerability_specs_total": 2,
        },
    )
    criteria = _criteria(card)
    assert criteria["gap_discovery"]["status"] == "partial"
    assert criteria["learning_policy"]["status"] == "partial"
    assert criteria["retention"]["status"] == "partial"
    assert criteria["transfer"]["status"] == "not_met"
    assert criteria["meta_learning"]["status"] == "not_met"
    assert criteria["low_cost"]["status"] == "met"
    assert criteria["transfer"]["score"] <= 0.05
    assert "heldout_task_transfer_summary.json" in " ".join(
        criteria["transfer"]["sources"]
    )
    assert "QKV LoRA first result is negative" in " ".join(
        criteria["meta_learning"]["evidence"]
    )
    assert card["held_out_tasks"]["runtime_copy_tool"]["counts_as_model_transfer"] is False
    assert card["summary"]["aggregate_score"] is None
    assert card["summary"]["status_counts"] == {
        "met": 1,
        "partial": 3,
        "not_met": 2,
        "unknown": 0,
    }


def test_cli_writes_json_and_markdown(tmp_path: Path):
    result_files = [
        "--result",
        str(ROOT / "research/results/heldout_task_transfer_summary.json"),
        "--result",
        str(ROOT / "research/results/copy_tool_benchmark_summary.json"),
        "--result",
        str(ROOT / "research/results/self_improve_curriculum_summary.json"),
        "--cpu",
        "8",
        "--ram-gb",
        "16",
        "--disk-gb",
        "40",
    ]
    json_one = tmp_path / "one.json"
    md_one = tmp_path / "one.md"
    json_two = tmp_path / "two.json"
    md_two = tmp_path / "two.md"
    first = _run_scorecard(
        *result_files, "--json-out", str(json_one), "--md-out", str(md_one)
    )
    assert first.returncode == 0, first.stderr
    second = _run_scorecard(
        *result_files, "--json-out", str(json_two), "--md-out", str(md_two)
    )
    assert second.returncode == 0, second.stderr
    assert json_one.read_bytes() == json_two.read_bytes()
    assert md_one.read_bytes() == md_two.read_bytes()
    card = json.loads(json_one.read_text(encoding="utf-8"))
    assert len(card["criteria"]) == 6
    assert card["schema"] == "atomzero.capability_scorecard.v1"
    markdown = md_one.read_text(encoding="utf-8")
    assert "| # | Criterion | Status | Score | Evidence |" in markdown
    assert "runtime copy tool" in markdown
    assert "counted as model transfer" in markdown
    for name in (
        "Discovers its own knowledge gaps",
        "Decides what, when and how to learn",
        "Retains old knowledge while learning",
        "Transfers learning to different problems",
        "Improves its own learning algorithm",
        "Does this at low hardware cost",
    ):
        assert name in markdown


def test_cli_rejects_unreadable_result(tmp_path: Path):
    missing = tmp_path / "missing.json"
    result = _run_scorecard("--result", str(missing))
    assert result.returncode == 2
    assert "capability_scorecard:" in result.stderr


def test_capacity_probe_is_reported_and_not_counted_as_transfer():
    probe = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "rows_per_stage": 20,
        "status": "capacity_evidence_not_product_claim",
        "stages": [
            {"stage": "2short", "pairs_per_row": 2, "rows": 20,
             "baseline": {"exact_rate": 0.65},
             "flat_prompt": {"exact_rate": 0.85}},
            {"stage": "4short", "pairs_per_row": 4, "rows": 20,
             "baseline": {"exact_rate": 0.55},
             "flat_prompt": {"exact_rate": 0.85}},
        ],
    }
    card = build_scorecard([], capacity_probe=probe)
    assert card["capacity_probe"]["model"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert card["capacity_probe"]["stages"][1]["flat_prompt_exact"] == 0.85
    assert "capacity evidence only" in card["capacity_probe"]["note"]
    text = render_markdown(card)
    assert "Capacity probe" in text
    assert "not counted as 49M transfer" in text
    assert "4short" in text
