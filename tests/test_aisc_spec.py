import ast
import json
from pathlib import Path

import pytest

from atomzero import aisc_spec

ROOT = Path(__file__).resolve().parents[1]
REAL_PROFILE = (
    ROOT / "benchmarks" / "results" / "answerability_hybrid"
    / "answerability_profile.json")

EVIDENCE = {
    "source": "benchmarks/results/answerability_hybrid/answerability_profile.json",
    "generated_on": "2026-11-23",
    "n": 200,
    "brier": 0.244274,
    "ece": 0.129867,
    "threshold": 0.6,
    "refusal_precision": 0.631,
    "refusal_recall": 0.5464,
    "answerable_coverage": 0.699,
    "commit": "b4b68b18b5d7122d182e387527f4f54d8a59b094",
    "dataset_hash": (
        "d5f6a89dca4ccfb1e44d0b10b5270a7e7e8acbf367d8dd04b4b94487958de15f"),
    "calibration_date": "2026-11-23",
    "recalibration_due": "2027-02-21",
}

GOLDEN = """name = "AtomZeroAnswerabilityGenerated"
version = "0.1"
description = "Generated from measured held-out answerability and refusal-policy evidence"
min_assurance = 0.80
max_context_tokens = 50000
answerability_n = 200
answerability_brier = 0.244274
answerability_ece = 0.129867
refusal_precision = 0.631000
refusal_recall = 0.546400
answerable_coverage = 0.699000
refusal_threshold = 0.600000
min_refusal_precision = 0.90
min_refusal_recall = 0.90
min_answerable_coverage = 0.80

[[components]]
name = "AnswerabilitySignal"
input_type = "Query"
output_type = "Decision"
kind = "model"
assurance = 0.80
verified = true
evidence = "held-out answerability experiment; source=benchmarks/results/answerability_hybrid/answerability_profile.json; generated_on=2026-11-23; n=200; commit=b4b68b18b5d7122d182e387527f4f54d8a59b094; dataset_hash=d5f6a89dca4ccfb1e44d0b10b5270a7e7e8acbf367d8dd04b4b94487958de15f; calibration_date=2026-11-23; recalibration_due=2027-02-21"
guarantees = ["calibrated answerability probability"]
context_tokens = 2000
"""


def _profile() -> dict:
    return {
        key: EVIDENCE[key]
        for key in aisc_spec.REQUIRED_KEYS
    }


def test_build_spec_matches_deterministic_golden_string():
    spec = aisc_spec.build_spec(EVIDENCE, generated_on="2026-11-23")
    assert spec == GOLDEN
    assert "\r" not in spec
    assert spec.endswith("\n")
    assert not spec.endswith("\n\n")


def test_build_spec_uses_evidence_generated_on_when_not_injected():
    spec = aisc_spec.build_spec(EVIDENCE)
    assert "generated_on=2026-11-23" in spec
    assert spec == GOLDEN


@pytest.mark.parametrize("key", aisc_spec.REQUIRED_KEYS)
def test_build_spec_rejects_each_missing_required_key(key):
    evidence = dict(EVIDENCE)
    del evidence[key]
    with pytest.raises(ValueError, match=key):
        aisc_spec.build_spec(evidence, generated_on="2026-11-23")


@pytest.mark.parametrize("key", aisc_spec.REQUIRED_KEYS)
def test_collect_evidence_rejects_each_missing_required_key(tmp_path, key):
    profile = _profile()
    del profile[key]
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    with pytest.raises(ValueError, match=key):
        aisc_spec.collect_evidence(tmp_path, path)


def test_collect_evidence_honors_injected_generated_on(tmp_path):
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(_profile()), encoding="utf-8")
    evidence = aisc_spec.collect_evidence(
        tmp_path, path, generated_on="2030-01-02")
    assert evidence["generated_on"] == "2030-01-02"
    assert evidence["source"] == "profile.json"
    assert evidence["brier"] == 0.244274


def test_collect_evidence_defaults_to_profile_created_date(tmp_path):
    profile = _profile()
    profile["created"] = "2026-11-23T18:54:31"
    path = tmp_path / "profile.json"
    path.write_text(json.dumps(profile), encoding="utf-8")
    evidence = aisc_spec.collect_evidence(tmp_path, path)
    assert evidence["generated_on"] == "2026-11-23"


def test_targets_cannot_be_lowered_by_evidence():
    weakened = {
        **EVIDENCE,
        "min_refusal_precision": 0.10,
        "min_refusal_recall": 0.10,
        "min_answerable_coverage": 0.10,
        "min_assurance": 0.10,
    }
    spec = aisc_spec.build_spec(weakened, generated_on="2026-11-23")
    assert "min_refusal_precision = 0.90" in spec
    assert "min_refusal_recall = 0.90" in spec
    assert "min_answerable_coverage = 0.80" in spec
    assert "min_assurance = 0.80" in spec
    assert "refusal_precision = 0.631000" in spec
    assert "refusal_recall = 0.546400" in spec
    assert "answerable_coverage = 0.699000" in spec


def test_check_reports_drift_without_writing(tmp_path):
    target = tmp_path / "spec.toml"
    result = aisc_spec.write_spec(
        target, EVIDENCE, check=True, generated_on="2026-11-23")
    assert result == {"ok": False, "changed": True, "path": str(target)}
    assert not target.exists()


def test_write_then_check_is_clean_and_uses_lf(tmp_path):
    target = tmp_path / "spec.toml"
    written = aisc_spec.write_spec(
        target, EVIDENCE, generated_on="2026-11-23")
    assert written == {"ok": True, "changed": True, "path": str(target)}
    assert target.read_text(encoding="utf-8") == GOLDEN
    assert b"\r" not in target.read_bytes()

    clean = aisc_spec.write_spec(
        target, EVIDENCE, check=True, generated_on="2026-11-23")
    assert clean == {"ok": True, "changed": False, "path": str(target)}


def test_check_detects_edited_spec(tmp_path):
    target = tmp_path / "spec.toml"
    aisc_spec.write_spec(target, EVIDENCE, generated_on="2026-11-23")
    target.write_text(GOLDEN + "\n", encoding="utf-8")
    result = aisc_spec.write_spec(
        target, EVIDENCE, check=True, generated_on="2026-11-23")
    assert result["ok"] is False
    assert result["changed"] is True


def test_main_check_exit_codes(tmp_path):
    profile = tmp_path / "profile.json"
    profile.write_text(json.dumps(_profile()), encoding="utf-8")
    target = tmp_path / "spec.toml"
    base = [
        "--profile", str(profile),
        "--out", str(target),
        "--generated-on", "2026-11-23",
    ]

    assert aisc_spec.main(base + ["--check"]) == 1
    assert not target.exists()

    assert aisc_spec.main(base) == 0
    assert target.exists()
    assert aisc_spec.main(base + ["--check"]) == 0


def test_collect_evidence_reads_real_profile_if_present():
    if not REAL_PROFILE.exists():
        pytest.skip("real answerability profile is not present")
    profile = json.loads(REAL_PROFILE.read_text(encoding="utf-8"))
    evidence = aisc_spec.collect_evidence(ROOT)

    for key in aisc_spec.REQUIRED_KEYS:
        assert evidence[key] == profile[key]
    assert evidence["source"] == (
        "benchmarks/results/answerability_hybrid/answerability_profile.json")
    created = str(profile.get("created", ""))
    expected_stamp = (
        created[:10] if created else profile["calibration_date"])
    assert evidence["generated_on"] == expected_stamp

    spec = aisc_spec.build_spec(evidence)
    assert f"answerability_brier = {profile['brier']:.6f}" in spec
    assert f"answerability_ece = {profile['ece']:.6f}" in spec
    assert f"refusal_precision = {profile['refusal_precision']:.6f}" in spec
    assert f"refusal_recall = {profile['refusal_recall']:.6f}" in spec
    assert (
        f"answerable_coverage = {profile['answerable_coverage']:.6f}"
        in spec
    )
    assert "min_refusal_precision = 0.90" in spec
    assert "min_refusal_recall = 0.90" in spec
    assert "min_answerable_coverage = 0.80" in spec


def test_module_does_not_import_aisc():
    source = (ROOT / "src" / "atomzero" / "aisc_spec.py").read_text(
        encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name == "aisc" or name.startswith("aisc.") for name in imported)
    assert "importlib" not in source
    assert "__import__" not in source
