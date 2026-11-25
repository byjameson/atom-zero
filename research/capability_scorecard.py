#!/usr/bin/env python3
"""Content-free six-criterion capability scorecard for Atom Zero.

The scorecard deliberately reports six separate criteria instead of one
aggregate loss number.  Every criterion gets a status, a bounded score where a
score is defensible, short evidence strings, source paths and one next
measurable target.

Only numeric metrics and fixed labels are copied out of the input documents.
Prompt text, answer text, raw completions and free-form notes are never written
to the JSON or Markdown output.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "atomzero.capability_scorecard.v1"

CRITERIA: tuple[tuple[str, str], ...] = (
    ("gap_discovery", "Discovers its own knowledge gaps"),
    ("learning_policy", "Decides what, when and how to learn"),
    ("retention", "Retains old knowledge while learning"),
    ("transfer", "Transfers learning to different problems"),
    ("meta_learning", "Improves its own learning algorithm"),
    ("low_cost", "Does this at low hardware cost"),
)

HARDWARE_ENVELOPE = {"cpu": 8.0, "ram_gb": 16.0, "disk_gb": 40.0}

# Keys that must never appear in scorecard output.  The extractor only reads
# whitelisted numeric fields, so this is a second line of defence.
_BANNED_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "answer",
        "answers",
        "completion",
        "completions",
        "user",
        "assistant",
        "question",
        "questions",
        "query",
        "queries",
        "source_text",
        "raw_text",
        "raw",
        "messages",
        "input_text",
        "output_text",
        "response",
        "responses",
    }
)

_TOKEN_RE = re.compile(r"^[A-Za-z0-9_.:/+-]{1,64}$")
_REASON_RE = re.compile(r"^[a-z0-9_]{1,40}$")


def _number(value: Any) -> float | None:
    """Return a finite float, or None for missing / non-numeric input."""
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _integer(value: Any) -> int | None:
    number = _number(value)
    if number is None:
        return None
    return int(round(number))


def _round(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 3)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _ratio(value: float | None, target: float) -> float | None:
    if value is None or target <= 0:
        return None
    return _clamp01(value / target)


def _token(value: Any, default: str) -> str:
    if isinstance(value, str) and _TOKEN_RE.match(value):
        return value
    return default


def _safe_reasons(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    reasons = []
    for item in value:
        if isinstance(item, str) and _REASON_RE.match(item):
            reasons.append(item)
    return reasons


def _source_label(path: str | Path) -> str:
    p = Path(path)
    try:
        resolved = p.resolve()
        return resolved.relative_to(ROOT).as_posix()
    except (OSError, ValueError):
        return p.name


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "unknown"
    return f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    if value is None:
        return "unknown"
    return f"{value * 100:.1f}%"


def _status(score: float | None, *, met: float = 0.85, partial: float = 0.30) -> str:
    if score is None:
        return "unknown"
    if score >= met:
        return "met"
    if score >= partial:
        return "partial"
    return "not_met"


def _loss_direction(base: float | None, adapter: float | None) -> str:
    if base is None or adapter is None:
        return "unknown"
    if adapter < base - 1e-9:
        return "adapter better"
    if adapter > base + 1e-9:
        return "adapter worse"
    return "equal"


def _empty_bundle() -> dict[str, list[dict[str, Any]]]:
    return {
        "curriculum_attempts": [],
        "curriculum_runs": [],
        "adapter_evals": [],
        "rehearsals": [],
        "multi_keys": [],
        "copy_curricula": [],
        "copy_tools": [],
        "answerability": [],
        "heldout_transfers": [],
        "unclassified": [],
    }


def classify_document(data: dict[str, Any]) -> str:
    """Classify a content-free result document by its shape."""
    if "successful_attempt" in data or "first_attempt" in data:
        return "curriculum_summary"
    if isinstance(data.get("plan"), dict) and isinstance(data.get("curriculum"), dict):
        return "curriculum_run"
    if "brier" in data or "ece" in data:
        return "answerability"
    if isinstance(data.get("tasks"), dict) and any(
        key in data["tasks"] for key in ("arithmetic", "city_yaml", "city_river")
    ):
        return "heldout_transfer"
    if isinstance(data.get("mlp"), dict) or isinstance(data.get("attn"), dict):
        return "copy_curriculum"
    if "stages" in data and "ranking" in data and "mix" in data:
        return "rehearsal"
    if "best_adapter" in data or "multi_short_standard" in data:
        return "multi_key"
    if "mixed" in data and "per_stage" in data:
        return "adapter_eval"
    if ("tool" in data or "model_used" in data) and (
        "exact_match" in data or "exact_rate" in data
    ):
        return "copy_tool"
    if (
        "exact_match" in data
        and "key_copy_rate" in data
        and "value_copy_rate" in data
        and ("formats" in data or "dataset" in data)
    ):
        return "copy_tool"
    return "unclassified"


def _extract_curriculum_summary(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    for label in ("first_attempt", "successful_attempt"):
        item = data.get(label)
        if not isinstance(item, dict):
            continue
        gate = item.get("gate")
        gate = gate if gate in ("pass", "fail") else None
        promotion = item.get("promotion")
        promotion = promotion if isinstance(promotion, bool) else None
        bundle["curriculum_attempts"].append(
            {
                "source": source,
                "label": label,
                "gate": gate,
                "promotion": promotion,
                "exact": _number(item.get("exact_match")),
                "key_copy": _number(item.get("key_copy")),
                "reasons": _safe_reasons(item.get("reasons")),
            }
        )


def _extract_curriculum_run(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    plan = data.get("plan")
    if not isinstance(plan, dict):
        return
    actions = plan.get("actions")
    gap_actions = []
    if isinstance(actions, list):
        for action in actions:
            if not isinstance(action, dict):
                continue
            kind = _token(action.get("kind"), "")
            if not kind:
                continue
            gap_actions.append(
                {
                    "kind": kind,
                    "requires_human": action.get("requires_human") is True,
                }
            )
    curriculum = data.get("curriculum")
    curriculum = curriculum if isinstance(curriculum, dict) else {}
    promotion = curriculum.get("promotion")
    promoted = promotion.get("promoted") if isinstance(promotion, dict) else None
    bundle["curriculum_runs"].append(
        {
            "source": source,
            "auto_apply": plan.get("auto_apply") is True,
            "gap_actions": gap_actions,
            "gate_status": curriculum.get("gate_status")
            if curriculum.get("gate_status") in ("pass", "fail")
            else None,
            "promoted": promoted if isinstance(promoted, bool) else None,
            "exact": _number(curriculum.get("exact_match")),
            "key_copy": _number(curriculum.get("key_copy")),
        }
    )


def _extract_answerability(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    selection = data.get("selection")
    selection = selection if isinstance(selection, dict) else {}
    misses = selection.get("misses")
    bundle["answerability"].append(
        {
            "source": source,
            "brier": _number(data.get("brier")),
            "ece": _number(data.get("ece")),
            "refusal_recall": _number(data.get("refusal_recall")),
            "refusal_precision": _number(data.get("refusal_precision")),
            "answerable_coverage": _number(data.get("answerable_coverage")),
            "n": _integer(data.get("n")),
            "policy_misses": len(misses) if isinstance(misses, list) else None,
            "feasible": selection.get("feasible_on_calibration")
            if isinstance(selection.get("feasible_on_calibration"), bool)
            else None,
        }
    )


def _extract_rehearsal(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    stages = data.get("stages")
    normalised = []
    if isinstance(stages, list):
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                continue
            name = _token(stage.get("stage"), f"stage_{index + 1}")
            before = stage.get("before")
            after = stage.get("after")
            delta = stage.get("delta")
            before = before if isinstance(before, dict) else {}
            after = after if isinstance(after, dict) else {}
            delta = delta if isinstance(delta, dict) else {}
            exact_before = _number(before.get("exact"))
            exact_after = _number(after.get("exact"))
            key_before = _number(before.get("key_copy"))
            key_after = _number(after.get("key_copy"))
            exact_delta = _number(delta.get("exact"))
            key_delta = _number(delta.get("key_copy"))
            if exact_delta is None and exact_before is not None and exact_after is not None:
                exact_delta = exact_after - exact_before
            if key_delta is None and key_before is not None and key_after is not None:
                key_delta = key_after - key_before
            normalised.append(
                {
                    "stage": name,
                    "group": _token(stage.get("group"), "unknown"),
                    "exact_before": exact_before,
                    "exact_after": exact_after,
                    "key_before": key_before,
                    "key_after": key_after,
                    "exact_delta": exact_delta,
                    "key_delta": key_delta,
                }
            )
    trainer = data.get("trainer")
    trainer = trainer if isinstance(trainer, dict) else {}
    result = trainer.get("result")
    result = result if isinstance(result, dict) else {}
    bundle["rehearsals"].append(
        {
            "source": source,
            "stages": normalised,
            "ranking_count": len(data.get("ranking"))
            if isinstance(data.get("ranking"), list)
            else None,
            "trainer_seconds": _number(trainer.get("elapsed_seconds")),
            "adapter_params": _integer(result.get("lora_parameters")),
        }
    )


def _extract_adapter_eval(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    mixed = data.get("mixed")
    mixed = mixed if isinstance(mixed, dict) else {}
    per_stage: dict[str, list[float | None]] = {}
    raw_per_stage = data.get("per_stage")
    if isinstance(raw_per_stage, dict):
        for name, values in raw_per_stage.items():
            name = _token(name, "stage")
            if isinstance(values, list) and len(values) >= 2:
                per_stage[name] = [
                    _number(values[0]),
                    _number(values[1]),
                ]
    bundle["adapter_evals"].append(
        {
            "source": source,
            "exact": _number(mixed.get("exact_match")),
            "key_copy": _number(mixed.get("key_copy_rate")),
            "examples": _integer(data.get("mixed_eval_examples")),
            "per_stage": per_stage,
        }
    )


def _extract_multi_key(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    methods = []
    for name in ("best_adapter", "key_weighted", "multi_short_standard"):
        block = data.get(name)
        if not isinstance(block, dict):
            continue
        stage_exact: dict[str, float | None] = {}
        per_stage = block.get("per_stage")
        if isinstance(per_stage, dict):
            for stage in ("2short", "2mixed", "3short", "4short"):
                values = per_stage.get(stage)
                if isinstance(values, list) and values:
                    stage_exact[stage] = _number(values[0])
        methods.append(
            {
                "name": name,
                "mixed_exact": _number(block.get("mixed_exact")),
                "mixed_key_copy": _number(block.get("mixed_key_copy")),
                "stage_exact": stage_exact,
            }
        )
    bundle["multi_keys"].append(
        {
            "source": source,
            "examples": _integer(data.get("mixed_eval_examples")),
            "methods": methods,
        }
    )


def _extract_copy_curriculum(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    models = []
    for model_name in ("mlp", "attn"):
        block = data.get(model_name)
        if not isinstance(block, dict):
            continue
        exact = block.get("exact_match")
        exact = exact if isinstance(exact, dict) else {}
        families: dict[str, dict[str, float | int | None]] = {}
        raw_families = exact.get("families")
        if isinstance(raw_families, dict):
            for family, family_data in raw_families.items():
                family = _token(family, "")
                if not family or not isinstance(family_data, dict):
                    continue
                families[family] = {
                    "examples": _integer(family_data.get("examples")),
                    "matches": _integer(family_data.get("matches")),
                    "rate": _number(family_data.get("rate")),
                }
        models.append(
            {
                "model": model_name,
                "examples": _integer(exact.get("examples")),
                "matches": _integer(exact.get("matches")),
                "rate": _number(exact.get("rate")),
                "families": families,
            }
        )
    if models:
        bundle["copy_curricula"].append({"source": source, "models": models})


def _extract_copy_tool(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    exact = _number(data.get("exact_match"))
    if exact is None:
        exact = _number(data.get("exact_rate"))
    bundle["copy_tools"].append(
        {
            "source": source,
            "tool": _token(data.get("tool"), "copy tool"),
            "model_used": data.get("model_used")
            if isinstance(data.get("model_used"), bool)
            else None,
            "exact": exact,
            "key_copy": _number(data.get("key_copy_rate")),
            "examples": _integer(data.get("examples"))
            or _integer(data.get("rows_total")),
        }
    )


def _extract_heldout_transfer(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> None:
    tasks = data.get("tasks")
    if not isinstance(tasks, dict):
        return
    normalised: dict[str, dict[str, float | None]] = {}
    for name in ("arithmetic", "city_yaml", "city_river"):
        block = tasks.get(name)
        if not isinstance(block, dict):
            continue
        normalised[name] = {
            "base_exact": _number(block.get("base_exact")),
            "adapter_exact": _number(block.get("adapter_exact")),
            "base_content_loss": _number(block.get("base_content_loss")),
            "adapter_content_loss": _number(block.get("adapter_content_loss")),
        }
    if normalised:
        bundle["heldout_transfers"].append(
            {
                "source": source,
                "examples_per_task": _integer(data.get("eval_examples_per_task")),
                "tasks": normalised,
            }
        )


def _extract(
    source: str, data: dict[str, Any], bundle: dict[str, list[dict[str, Any]]]
) -> str:
    kind = classify_document(data)
    if kind == "curriculum_summary":
        _extract_curriculum_summary(source, data, bundle)
    elif kind == "curriculum_run":
        _extract_curriculum_run(source, data, bundle)
    elif kind == "answerability":
        _extract_answerability(source, data, bundle)
    elif kind == "heldout_transfer":
        _extract_heldout_transfer(source, data, bundle)
    elif kind == "copy_curriculum":
        _extract_copy_curriculum(source, data, bundle)
    elif kind == "rehearsal":
        _extract_rehearsal(source, data, bundle)
    elif kind == "multi_key":
        _extract_multi_key(source, data, bundle)
    elif kind == "adapter_eval":
        _extract_adapter_eval(source, data, bundle)
    elif kind == "copy_tool":
        _extract_copy_tool(source, data, bundle)
    else:
        bundle["unclassified"].append({"source": source})
    return kind


def _merged_answerability(
    bundle: dict[str, list[dict[str, Any]]],
    flags: dict[str, Any] | None,
) -> tuple[dict[str, Any] | None, list[str]]:
    fields = (
        "brier",
        "ece",
        "refusal_recall",
        "refusal_precision",
        "answerable_coverage",
        "n",
        "policy_misses",
        "feasible",
    )
    merged: dict[str, Any] = {field: None for field in fields}
    sources = []
    for item in sorted(bundle["answerability"], key=lambda row: row["source"]):
        used = False
        for field in fields:
            if item.get(field) is not None:
                merged[field] = item[field]
                used = True
        if used:
            sources.append(item["source"])
    flags = flags or {}
    used_flags = False
    for field in fields:
        if field in ("policy_misses", "feasible"):
            continue
        value = _number(flags.get(field))
        if value is not None:
            merged[field] = value
            used_flags = True
    if used_flags:
        sources.append("cli flags")
    if all(merged[field] is None for field in fields):
        return None, []
    # Normalise integer fields.
    merged["n"] = _integer(merged.get("n"))
    merged["policy_misses"] = _integer(merged.get("policy_misses"))
    return merged, sources


def _gap_actions(
    bundle: dict[str, list[dict[str, Any]]]
) -> tuple[int, list[str], bool]:
    kinds: list[str] = []
    requires_human = False
    for run in bundle["curriculum_runs"]:
        for action in run["gap_actions"]:
            if action["kind"] not in kinds:
                kinds.append(action["kind"])
            if action["requires_human"]:
                requires_human = True
    return len(kinds), kinds, requires_human


def _criterion_gap_discovery(
    bundle: dict[str, list[dict[str, Any]]],
    answerability: dict[str, Any] | None,
    answerability_sources: list[str],
) -> dict[str, Any]:
    gap_count, gap_kinds, requires_human = _gap_actions(bundle)
    evidence: list[str] = []
    sources: list[str] = []
    score = 0.0
    if gap_count:
        score += 0.25
        kinds = ", ".join(gap_kinds[:3])
        evidence.append(
            f"self-improve plan recorded {gap_count} gap action(s) "
            f"(kind={kinds}); human approval required={str(requires_human).lower()}"
        )
        sources.extend(run["source"] for run in bundle["curriculum_runs"])
    if answerability is not None:
        score += 0.20
        brier = answerability.get("brier")
        ece = answerability.get("ece")
        n = answerability.get("n")
        if brier is not None:
            evidence.append(
                f"answerability Brier {_fmt(brier)} (target <=0.15): "
                f"{'met' if brier <= 0.15 else 'missed'}"
            )
            if brier <= 0.15:
                score += 0.20
        if ece is not None:
            evidence.append(
                f"answerability ECE {_fmt(ece)} (target <=0.10): "
                f"{'met' if ece <= 0.10 else 'missed'}"
            )
            if ece <= 0.10:
                score += 0.15
        recall = answerability.get("refusal_recall")
        precision = answerability.get("refusal_precision")
        if recall is not None:
            evidence.append(
                f"refusal recall {_fmt(recall)} (target >=0.80): "
                f"{'met' if recall >= 0.80 else 'missed'}"
            )
            if recall >= 0.80:
                score += 0.20
        if precision is not None:
            evidence.append(f"refusal precision {_fmt(precision)}")
        if n is not None:
            evidence.append(f"answerability held-out rows: {n}")
        sources.extend(answerability_sources)
    specs_failed = answerability.get("specs_failed") if answerability else None
    specs_total = answerability.get("specs_total") if answerability else None
    if specs_failed is not None and specs_total is not None:
        evidence.append(f"{specs_failed}/{specs_total} answerability specs FAIL")
    elif answerability is not None and answerability.get("policy_misses") is not None:
        evidence.append(
            f"{answerability['policy_misses']} refusal-policy target(s) missed"
        )
    if not evidence:
        return {
            "id": "gap_discovery",
            "name": CRITERIA[0][1],
            "status": "unknown",
            "score": None,
            "evidence": ["no gap-discovery or answerability artifact was supplied"],
            "sources": [],
            "next_target": (
                "record at least three self-discovered gap types, close one end-to-end, "
                "and reach Brier <=0.15 plus refusal recall >=0.80"
            ),
        }
    status = _status(score)
    if status == "met" and specs_failed not in (None, 0):
        status = "partial"
    return {
        "id": "gap_discovery",
        "name": CRITERIA[0][1],
        "status": status,
        "score": _round(_clamp01(score)),
        "evidence": evidence,
        "sources": sorted(set(sources)),
        "next_target": (
            "close at least one self-discovered gap end-to-end with a held-out "
            "check; reach Brier <=0.15, ECE <=0.10, refusal recall >=0.80 and "
            "zero failed answerability specs"
        ),
    }


def _criterion_learning_policy(
    bundle: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    attempts = bundle["curriculum_attempts"]
    runs = bundle["curriculum_runs"]
    evidence: list[str] = []
    sources: list[str] = []
    score = 0.0
    if attempts:
        score += 0.25
        sources.extend(item["source"] for item in attempts)
    fail_closed = any(
        item["gate"] == "fail" or item["promotion"] is False for item in attempts
    )
    if fail_closed:
        score += 0.20
    promoted = any(
        item["gate"] == "pass" or item["promotion"] is True for item in attempts
    )
    if promoted:
        score += 0.15
    for item in attempts:
        label = item["label"].replace("_", " ")
        details = [f"{label}: gate={item['gate'] or 'unknown'}"]
        if item["exact"] is not None:
            details.append(f"exact {_fmt(item['exact'])}")
        if item["key_copy"] is not None:
            details.append(f"key copy {_fmt(item['key_copy'])}")
        if item["promotion"] is not None:
            details.append(f"promoted={str(item['promotion']).lower()}")
        if item["reasons"]:
            details.append("reasons=" + ",".join(item["reasons"][:3]))
        evidence.append("; ".join(details))
    if fail_closed:
        evidence.append("a recorded candidate was rejected and not promoted")
    if promoted:
        evidence.append("a later candidate passed the gate and was promoted")
    selection_count = 0
    for rehearsal in bundle["rehearsals"]:
        if rehearsal["ranking_count"] and rehearsal["ranking_count"] >= 2:
            selection_count += 1
            sources.append(rehearsal["source"])
    if selection_count:
        score += 0.20
        evidence.append(
            f"selection evidence: {selection_count} rehearsal run(s) ranked "
            "two or more stages before choosing a mix"
        )
    multi_key_methods = 0
    for item in bundle["multi_keys"]:
        if len(item["methods"]) >= 2:
            multi_key_methods += len(item["methods"])
            sources.append(item["source"])
    if multi_key_methods:
        evidence.append(
            f"selection evidence: multi-key summary compared "
            f"{multi_key_methods} measured methods"
        )
    autonomous = False
    for run in runs:
        any_human = any(action["requires_human"] for action in run["gap_actions"])
        if run["promoted"] and run["auto_apply"] and not any_human:
            autonomous = True
        if run["gap_actions"]:
            evidence.append(
                "plan: auto_apply="
                f"{str(run['auto_apply']).lower()}; "
                f"{len(run['gap_actions'])} action(s), "
                f"human approval required={str(any_human).lower()}"
            )
            sources.append(run["source"])
    if autonomous:
        score += 0.20
        evidence.append("an end-to-end learning decision was applied without a human gate")
    if not attempts and not runs and not selection_count and not multi_key_methods:
        return {
            "id": "learning_policy",
            "name": CRITERIA[1][1],
            "status": "unknown",
            "score": None,
            "evidence": ["no curriculum, promotion or selection artifact was supplied"],
            "sources": [],
            "next_target": (
                "record one autonomous learn/stop decision with a rejected "
                "alternative and a passing held-out gate"
            ),
        }
    if not attempts and runs:
        # A run may carry metrics without a separate summary.
        for run in runs:
            if run["exact"] is not None:
                score += 0.25
                evidence.append(
                    "curriculum run: gate="
                    f"{run['gate_status'] or 'unknown'}, exact {_fmt(run['exact'])}, "
                    f"key copy {_fmt(run['key_copy'])}"
                )
                break
        if any(run["gate_status"] == "fail" for run in runs):
            score += 0.20
        if any(run["promoted"] for run in runs):
            score += 0.15
    status = _status(score)
    if status == "met" and not autonomous:
        status = "partial"
    return {
        "id": "learning_policy",
        "name": CRITERIA[1][1],
        "status": status,
        "score": _round(_clamp01(score)),
        "evidence": evidence,
        "sources": sorted(set(sources)),
        "next_target": (
            "make one autonomous learning decision (no human approval) with a "
            "recorded rejected alternative and a passing held-out gate"
        ),
    }


def _retention_stats(
    rehearsal: dict[str, Any]
) -> tuple[int, int, float | None, float | None, str | None]:
    stages = rehearsal["stages"]
    if not stages:
        return 0, 0, None, None, None
    retained = 0
    worst_exact = None
    worst_stage = None
    for stage in stages:
        exact_delta = stage["exact_delta"]
        key_delta = stage["key_delta"]
        if (
            exact_delta is not None
            and key_delta is not None
            and exact_delta >= -0.02
            and key_delta >= -0.02
        ):
            retained += 1
        if exact_delta is not None and (
            worst_exact is None or exact_delta < worst_exact
        ):
            worst_exact = exact_delta
            worst_stage = stage["stage"]
    return retained, len(stages), retained / len(stages), worst_exact, worst_stage


def _criterion_retention(
    bundle: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    rehearsals = bundle["rehearsals"]
    adapter_evals = bundle["adapter_evals"]
    evidence: list[str] = []
    sources: list[str] = []
    score = 0.0
    best_retention: float | None = None
    for rehearsal in rehearsals:
        retained, total, fraction, worst_exact, worst_stage = _retention_stats(
            rehearsal
        )
        if fraction is not None:
            best_retention = (
                fraction if best_retention is None else max(best_retention, fraction)
            )
            detail = (
                f"{retained}/{total} stages kept exact and key copy within 0.02"
            )
            if worst_exact is not None and worst_stage is not None:
                detail += (
                    f"; worst exact delta {worst_exact:+.3f} ({worst_stage})"
                )
            evidence.append(f"{rehearsal['source']}: {detail}")
            sources.append(rehearsal["source"])
    if best_retention is not None:
        score += 0.50 * best_retention
    mixed_exact = None
    mixed_key = None
    for item in adapter_evals:
        if item["exact"] is not None:
            mixed_exact = item["exact"]
            mixed_key = item["key_copy"]
            evidence.append(
                f"adapter eval: mixed exact {_fmt(mixed_exact)}, "
                f"key copy {_fmt(mixed_key)}"
                + (f", n={item['examples']}" if item["examples"] else "")
            )
            sources.append(item["source"])
            break
    if mixed_exact is not None:
        score += 0.30 * (_ratio(mixed_exact, 0.80) or 0.0)
    if mixed_key is not None:
        score += 0.20 * (_ratio(mixed_key, 0.80) or 0.0)
    multi_key_exact = _multi_key_exact(bundle)
    if multi_key_exact is not None:
        evidence.append(
            f"3-4 key copy remains weak: best exact {_fmt(multi_key_exact)}"
        )
        sources.extend(item["source"] for item in bundle["multi_keys"])
    if not evidence:
        return {
            "id": "retention",
            "name": CRITERIA[2][1],
            "status": "unknown",
            "score": None,
            "evidence": ["no retention or rehearsal artifact was supplied"],
            "sources": [],
            "next_target": (
                "after one new learning step, keep every stage within 0.02 of its "
                "pre-step exact and key-copy score"
            ),
        }
    if (
        best_retention is not None
        and best_retention >= 0.80
        and mixed_exact is not None
        and mixed_exact >= 0.70
    ):
        status = "met"
    elif score >= 0.30:
        status = "partial"
    else:
        status = "not_met"
    return {
        "id": "retention",
        "name": CRITERIA[2][1],
        "status": status,
        "score": _round(_clamp01(score)),
        "evidence": evidence,
        "sources": sorted(set(sources)),
        "next_target": (
            "after one new learning step, keep >=4/5 stages within 0.02 of their "
            "pre-step exact and key-copy score and reach mixed exact >=0.70"
        ),
    }


def _multi_key_exact(bundle: dict[str, list[dict[str, Any]]]) -> float | None:
    values = []
    for item in bundle["multi_keys"]:
        for method in item["methods"]:
            for stage in ("3short", "4short"):
                value = method["stage_exact"].get(stage)
                if value is not None:
                    values.append(value)
    if not values:
        return None
    return max(values)


def _heldout_tasks(
    bundle: dict[str, list[dict[str, Any]]],
    flags: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    flags = flags or {}
    tasks: dict[str, dict[str, Any]] = {}
    examples_per_task = None
    sources: list[str] = []
    for item in sorted(bundle["heldout_transfers"], key=lambda row: row["source"]):
        if item["examples_per_task"]:
            examples_per_task = item["examples_per_task"]
        for name, metrics in item["tasks"].items():
            tasks[name] = dict(metrics)
        sources.append(item["source"])
    # Explicit flags override the files field by field.
    if _number(flags.get("city_river_exact")) is not None:
        tasks.setdefault("city_river", {})["adapter_exact"] = _number(
            flags.get("city_river_exact")
        )
        tasks.setdefault("city_river", {})["base_exact"] = None
    if _integer(flags.get("city_river_examples")) is not None:
        examples_per_task = _integer(flags.get("city_river_examples"))
    if _number(flags.get("arithmetic_exact")) is not None:
        tasks.setdefault("arithmetic", {})["adapter_exact"] = _number(
            flags.get("arithmetic_exact")
        )
    if _number(flags.get("city_yaml_exact")) is not None:
        tasks.setdefault("city_yaml", {})["adapter_exact"] = _number(
            flags.get("city_yaml_exact")
        )
    if _integer(flags.get("arith_yaml_examples")) is not None:
        examples_per_task = _integer(flags.get("arith_yaml_examples"))
    if any(
        _number(flags.get(field)) is not None
        for field in (
            "city_river_exact",
            "multi_key_exact",
            "arithmetic_exact",
            "city_yaml_exact",
        )
    ):
        sources.append("cli flags")

    copy_curriculum_exact = None
    copy_curriculum_examples = None
    copy_curriculum_source = None
    for item in sorted(bundle["copy_curricula"], key=lambda row: row["source"]):
        for model in item["models"]:
            if model["rate"] is None:
                continue
            if copy_curriculum_exact is None or model["rate"] > copy_curriculum_exact:
                copy_curriculum_exact = model["rate"]
                copy_curriculum_examples = model["examples"]
                copy_curriculum_source = item["source"]
    if "city_river" not in tasks and copy_curriculum_exact is not None:
        tasks["city_river"] = {
            "base_exact": None,
            "adapter_exact": copy_curriculum_exact,
            "base_content_loss": None,
            "adapter_content_loss": None,
        }
        if examples_per_task is None:
            examples_per_task = copy_curriculum_examples
        if copy_curriculum_source:
            sources.append(copy_curriculum_source)

    multi_key_exact = _multi_key_exact(bundle)
    if _number(flags.get("multi_key_exact")) is not None:
        multi_key_exact = _number(flags.get("multi_key_exact"))
    multi_key_examples = None
    for item in bundle["multi_keys"]:
        if item["examples"]:
            multi_key_examples = item["examples"]
            break
    if _integer(flags.get("multi_key_examples")) is not None:
        multi_key_examples = _integer(flags.get("multi_key_examples"))

    def task_entry(name: str, family: str, status: str) -> dict[str, Any]:
        metrics = tasks.get(name, {})
        return {
            "family": family,
            "examples": examples_per_task,
            "base_exact": _round(metrics.get("base_exact")),
            "adapter_exact": _round(metrics.get("adapter_exact")),
            "base_content_loss": _round(metrics.get("base_content_loss")),
            "adapter_content_loss": _round(metrics.get("adapter_content_loss")),
            "content_loss_direction": _loss_direction(
                metrics.get("base_content_loss"),
                metrics.get("adapter_content_loss"),
            ),
            "status": status,
            "counts_as_model_transfer": True,
            "sources": sorted(set(sources)),
        }

    def measured_exact(name: str) -> float | None:
        metrics = tasks.get(name)
        if not metrics:
            return None
        return metrics.get("adapter_exact")

    city_river = measured_exact("city_river")
    arithmetic = measured_exact("arithmetic")
    city_yaml = measured_exact("city_yaml")
    arith_yaml_values = [v for v in (arithmetic, city_yaml) if v is not None]
    arith_yaml = max(arith_yaml_values) if arith_yaml_values else None

    def exact_status(value: float | None, met: float, partial: float) -> str:
        if value is None:
            return "unknown"
        if value >= met:
            return "met"
        if value >= partial:
            return "partial"
        return "not_met"

    city_river_status = exact_status(city_river, 0.50, 0.20)
    multi_key_status = exact_status(multi_key_exact, 0.30, 0.10)
    arith_yaml_status = exact_status(arith_yaml, 0.50, 0.20)
    arithmetic_status = exact_status(arithmetic, 0.50, 0.20)
    city_yaml_status = exact_status(city_yaml, 0.50, 0.20)

    tool = None
    if bundle["copy_tools"]:
        tool_item = sorted(bundle["copy_tools"], key=lambda row: row["source"])[0]
        tool = {
            "family": "explicit multi-key transcription",
            "examples": tool_item["examples"],
            "base_exact": None,
            "adapter_exact": _round(tool_item["exact"]),
            "key_copy": _round(tool_item["key_copy"]),
            "status": "met"
            if tool_item["exact"] is not None and tool_item["exact"] >= 0.99
            else "partial"
            if tool_item["exact"] is not None and tool_item["exact"] >= 0.50
            else "not_met"
            if tool_item["exact"] is not None
            else "unknown",
            "kind": "tool_use",
            "counts_as_model_transfer": False,
            "note": (
                "deterministic runtime transcription; this is tool use, not "
                "model transfer"
            ),
            "sources": [tool_item["source"]],
        }
    else:
        tool = {
            "family": "explicit multi-key transcription",
            "examples": None,
            "base_exact": None,
            "adapter_exact": None,
            "key_copy": None,
            "status": "unknown",
            "kind": "tool_use",
            "counts_as_model_transfer": False,
            "note": (
                "no runtime copy-tool benchmark supplied; tool use is never "
                "counted as model transfer"
            ),
            "sources": [],
        }

    held_out = {
        "copy_transfer": {
            "city_river": task_entry(
                "city_river", "city/river held-out entities", city_river_status
            ),
            "multi_key": {
                "family": "3-4 arbitrary keys",
                "examples": multi_key_examples,
                "base_exact": None,
                "adapter_exact": _round(multi_key_exact),
                "counts_as_model_transfer": True,
                "status": multi_key_status,
                "sources": sorted(
                    {item["source"] for item in bundle["multi_keys"]}
                ),
            },
            "best_model_exact": _round(
                max(
                    [v for v in (city_river, multi_key_exact) if v is not None],
                    default=None,
                )
                if any(v is not None for v in (city_river, multi_key_exact))
                else None
            ),
        },
        "arithmetic_yaml": {
            "arithmetic": task_entry(
                "arithmetic", "held-out arithmetic", arithmetic_status
            ),
            "city_yaml": task_entry(
                "city_yaml", "held-out city YAML", city_yaml_status
            ),
            "best_exact": _round(arith_yaml),
            "status": arith_yaml_status,
            "counts_as_model_transfer": True,
            "sources": sorted(set(sources)),
        },
        "runtime_copy_tool": tool,
        "tool_use_is_model_transfer": False,
    }
    selected = {
        "city_river": city_river,
        "arith_yaml": arith_yaml,
        "multi_key": multi_key_exact,
        "sources": sorted(set(sources)),
        "heldout_present": bool(tasks),
    }
    return held_out, selected


def _criterion_transfer(
    bundle: dict[str, list[dict[str, Any]]],
    held_out: dict[str, Any],
    selected: dict[str, Any],
) -> dict[str, Any]:
    evidence: list[str] = []
    sources: list[str] = list(selected["sources"])
    for name in ("city_river", "arithmetic", "city_yaml"):
        entry = None
        if name == "city_river":
            entry = held_out["copy_transfer"]["city_river"]
        else:
            entry = held_out["arithmetic_yaml"][name]
        if entry["adapter_exact"] is None and entry["base_exact"] is None:
            continue
        parts = [
            f"held-out {name}: base {_fmt(entry['base_exact'])} / "
            f"adapter {_fmt(entry['adapter_exact'])} exact"
        ]
        if entry["examples"] is not None:
            parts.append(f"n={entry['examples']}")
        if entry["base_content_loss"] is not None:
            parts.append(
                "content loss "
                f"{_fmt(entry['base_content_loss'])} -> "
                f"{_fmt(entry['adapter_content_loss'])} "
                f"({entry['content_loss_direction']})"
            )
        evidence.append("; ".join(parts))
    multi_key = held_out["copy_transfer"]["multi_key"]
    if multi_key["adapter_exact"] is not None:
        evidence.append(
            f"3-4 key copy: best exact {_fmt(multi_key['adapter_exact'])} "
            f"(n={multi_key['examples'] or 'unknown'})"
        )
        sources.extend(multi_key["sources"])
    for item in sorted(bundle["copy_curricula"], key=lambda row: row["source"]):
        best_model = None
        for model in item["models"]:
            if model["rate"] is None:
                continue
            if best_model is None or model["rate"] > best_model["rate"]:
                best_model = model
        if best_model is not None:
            family_count = len(best_model["families"])
            evidence.append(
                "copy curriculum "
                f"({best_model['model']}): "
                f"{best_model['matches'] or 0}/{best_model['examples'] or 0} "
                f"exact across {family_count} held-out families"
            )
            sources.append(item["source"])
    tool = held_out["runtime_copy_tool"]
    if tool["adapter_exact"] is not None:
        evidence.append(
            f"runtime copy tool {_fmt(tool['adapter_exact'])} exact is tool use "
            "and is excluded from model transfer"
        )
    components = []
    if selected["city_river"] is not None:
        components.append((0.50, selected["city_river"], 0.50, "city/river"))
    if selected["arith_yaml"] is not None:
        components.append((0.30, selected["arith_yaml"], 0.50, "arithmetic/YAML"))
    if selected["multi_key"] is not None:
        components.append((0.20, selected["multi_key"], 0.30, "3-4 key"))
    if not components:
        return {
            "id": "transfer",
            "name": CRITERIA[3][1],
            "status": "unknown",
            "score": None,
            "evidence": evidence
            or ["no held-out model-transfer result was supplied"],
            "sources": sorted(set(sources)),
            "next_target": (
                "measure held-out city/river, arithmetic/YAML and 3-4 key copy "
                "exact match with n>=100 per family"
            ),
        }
    weight_total = sum(component[0] for component in components)
    score = 0.0
    for weight, value, target, label in components:
        ratio = _ratio(value, target) or 0.0
        score += weight * ratio
        evidence.append(
            f"{label} transfer target: {_fmt(value)} / {_fmt(target)} "
            f"({_pct(ratio)} of target)"
        )
    score = score / weight_total if weight_total else 0.0
    measured = [value for _, value, _, _ in components]
    best = max(measured)
    met_breadth = (
        selected["city_river"] is not None
        and selected["city_river"] >= 0.50
        and held_out["arithmetic_yaml"]["arithmetic"]["adapter_exact"] is not None
        and held_out["arithmetic_yaml"]["arithmetic"]["adapter_exact"] >= 0.50
        and held_out["arithmetic_yaml"]["city_yaml"]["adapter_exact"] is not None
        and held_out["arithmetic_yaml"]["city_yaml"]["adapter_exact"] >= 0.50
    )
    if met_breadth:
        status = "met"
    elif best >= 0.20:
        status = "partial"
    else:
        status = "not_met"
    return {
        "id": "transfer",
        "name": CRITERIA[3][1],
        "status": status,
        "score": _round(_clamp01(score)),
        "evidence": evidence,
        "sources": sorted(set(sources)),
        "next_target": (
            "reach >=0.50 exact on held-out city/river and arithmetic/YAML with "
            "n>=100 per family, and >=0.30 exact on 3-4 arbitrary keys; keep "
            "the runtime copy tool out of the model path"
        ),
    }


def _criterion_meta_learning(
    bundle: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    evidence: list[str] = []
    sources: list[str] = []
    attempt_count = 0
    measured = False
    negative = False
    positive = False

    for rehearsal in bundle["rehearsals"]:
        attempt_count += 1
        measured = True
        sources.append(rehearsal["source"])
        retained, total, fraction, worst_exact, worst_stage = _retention_stats(
            rehearsal
        )
        if fraction is not None:
            evidence.append(
                f"weak-stage rehearsal changed the stage mix; {retained}/{total} "
                "stages retained after the change"
                + (
                    f"; worst exact delta {worst_exact:+.3f} ({worst_stage})"
                    if worst_exact is not None and worst_stage is not None
                    else ""
                )
            )
            if worst_exact is not None and worst_exact < -0.02:
                negative = True
    for item in bundle["multi_keys"]:
        methods = {method["name"]: method for method in item["methods"]}
        standard = methods.get("multi_short_standard", {}).get("mixed_exact")
        key_weighted = methods.get("key_weighted", {}).get("mixed_exact")
        if standard is not None and key_weighted is not None:
            attempt_count += 1
            measured = True
            sources.append(item["source"])
            delta = key_weighted - standard
            evidence.append(
                f"key-weighted loss: mixed exact {_fmt(key_weighted)} vs "
                f"{_fmt(standard)} standard ({delta:+.3f})"
            )
            if delta > 0.05:
                positive = True
            elif delta < -0.005:
                negative = True
    for item in bundle["copy_curricula"]:
        if len(item["models"]) >= 2:
            attempt_count += 1
            measured = True
            sources.append(item["source"])
            rates = [
                model["rate"]
                for model in item["models"]
                if model["rate"] is not None
            ]
            if rates:
                delta = max(rates) - min(rates)
                evidence.append(
                    "copy curriculum compared MLP and attention adapter loci; "
                    f"best exact {_fmt(max(rates))}, delta {delta:+.3f}"
                )
                if delta >= 0.05:
                    positive = True
                elif delta <= -0.05:
                    negative = True
    # The QKV first run is recorded as negative in the 0.60.0 plan.  It has no
    # positive result file, so it must never be counted as an improvement.
    qkv_source = "docs/BATCH_0.60.0.md"
    evidence.append(
        "QKV LoRA first result is negative/uncontrolled (6% exact / 18% "
        "key copy on multi-short); the controlled 3-seed ablation was "
        "inconclusive (3.33% exact both arms); no positive QKV result is "
        "counted"
    )
    if (ROOT / qkv_source).exists():
        sources.append(qkv_source)
    if not attempt_count and not measured:
        return {
            "id": "meta_learning",
            "name": CRITERIA[4][1],
            "status": "unknown",
            "score": None,
            "evidence": ["no learning-algorithm change artifact was supplied"],
            "sources": [],
            "next_target": (
                "record one learning-algorithm change that beats its own "
                "baseline by >=0.05 mixed exact and keeps >=4/5 stages"
            ),
        }
    score = 0.0
    if attempt_count:
        score += 0.10
        evidence.insert(
            0,
            f"{attempt_count} learning-algorithm change attempt(s) recorded",
        )
    if measured:
        score += 0.10
    if negative:
        score += 0.10
    if positive:
        score += 0.70
    else:
        evidence.append(
            "no learning-algorithm change improved the aggregate target by "
            ">=0.05; process evidence is not counted as improvement"
        )
    if positive:
        status = "met"
    else:
        status = "not_met"
    return {
        "id": "meta_learning",
        "name": CRITERIA[4][1],
        "status": status,
        "score": _round(_clamp01(score)),
        "evidence": evidence,
        "sources": sorted(set(sources)),
        "next_target": (
            "show one learning-algorithm change that improves mixed exact by "
            ">=0.05 over its own baseline, retains >=4/5 stages and reproduces "
            "across >=2 seeds"
        ),
    }


def _hardware_values(
    bundle: dict[str, list[dict[str, Any]]], hardware: dict[str, Any] | None
) -> tuple[dict[str, float | None], dict[str, Any]]:
    hardware = hardware or {}
    values = {
        "cpu": _number(hardware.get("cpu")),
        "ram_gb": _number(hardware.get("ram_gb")),
        "disk_gb": _number(hardware.get("disk_gb")),
    }
    extras = {
        "model_params_m": _number(hardware.get("model_params_m")),
        "adapter_params": _integer(hardware.get("adapter_params")),
        "train_seconds": _number(hardware.get("train_seconds")),
        "tests": _integer(hardware.get("tests")),
    }
    if extras["adapter_params"] is None:
        for rehearsal in bundle["rehearsals"]:
            if rehearsal["adapter_params"]:
                extras["adapter_params"] = rehearsal["adapter_params"]
                break
    if extras["train_seconds"] is None:
        seconds = [
            rehearsal["trainer_seconds"]
            for rehearsal in bundle["rehearsals"]
            if rehearsal["trainer_seconds"] is not None
        ]
        if seconds:
            extras["train_seconds"] = max(seconds)
    return values, extras


def _criterion_low_cost(
    bundle: dict[str, list[dict[str, Any]]], hardware: dict[str, Any] | None
) -> dict[str, Any]:
    values, extras = _hardware_values(bundle, hardware)
    evidence: list[str] = []
    sources: list[str] = []
    known = []
    for name, limit in HARDWARE_ENVELOPE.items():
        value = values.get(name)
        if value is None:
            continue
        known.append((name, value, limit))
        evidence.append(
            f"{name.replace('_', ' ')} {value:g} (envelope <={limit:g})"
        )
    for rehearsal in bundle["rehearsals"]:
        if rehearsal["trainer_seconds"] is not None:
            sources.append(rehearsal["source"])
            break
    if extras["model_params_m"] is not None:
        evidence.append(f"base model {extras['model_params_m']:g}M parameters")
    if extras["adapter_params"] is not None:
        evidence.append(
            f"adapter {extras['adapter_params']:,} trainable parameters"
        )
    if extras["train_seconds"] is not None:
        evidence.append(f"recorded adapter training {extras['train_seconds']:.1f}s")
    if extras["tests"] is not None:
        evidence.append(f"{extras['tests']:,} tests recorded")
    if not known:
        return {
            "id": "low_cost",
            "name": CRITERIA[5][1],
            "status": "unknown",
            "score": None,
            "evidence": evidence
            or ["no hardware metrics were supplied (use --cpu, --ram-gb, --disk-gb)"],
            "sources": sorted(set(sources)),
            "next_target": (
                "record CPU, RAM and disk for the full loop and keep them within "
                "8 vCPU / 16 GB / 40 GB"
            ),
        }
    ratios = []
    for _, value, limit in known:
        ratios.append(1.0 if value <= limit else limit / value)
    score = sum(ratios) / len(ratios)
    if len(known) < len(HARDWARE_ENVELOPE):
        score = min(score, 0.80)
        status = "partial" if score >= 0.50 else "not_met"
    elif score >= 0.999:
        status = "met"
    elif score >= 0.50:
        status = "partial"
    else:
        status = "not_met"
    return {
        "id": "low_cost",
        "name": CRITERIA[5][1],
        "status": status,
        "score": _round(_clamp01(score)),
        "evidence": evidence,
        "sources": sorted(set(sources)),
        "next_target": (
            "keep the full loop within 8 vCPU / 16 GB RAM / 40 GB disk while "
            "adding a held-out transfer eval; report wall-clock and peak RAM "
            "per cycle"
        ),
    }


def _validate_content_free(card: dict[str, Any]) -> None:
    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if str(key).lower() in _BANNED_KEYS:
                    raise ValueError(f"scorecard contains banned key: {key}")
                walk(child)
        elif isinstance(value, list):
            for child in value:
                walk(child)
        elif isinstance(value, str):
            if "\n" in value or "\r" in value:
                raise ValueError("scorecard string contains a newline")
            if len(value) > 400:
                raise ValueError("scorecard string is unexpectedly long")

    walk(card)


def _capacity_probe_summary(data: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(data, dict):
        return None
    stages = []
    for row in data.get("stages", []) or []:
        if not isinstance(row, dict):
            continue
        baseline = row.get("baseline") if isinstance(row.get("baseline"), dict) else {}
        flat = row.get("flat_prompt") if isinstance(row.get("flat_prompt"), dict) else {}
        stages.append({
            "stage": str(row.get("stage", "")),
            "pairs_per_row": _integer(row.get("pairs_per_row")),
            "rows": _integer(row.get("rows")),
            "baseline_exact": _round(_number(baseline.get("exact_rate"))),
            "flat_prompt_exact": _round(_number(flat.get("exact_rate"))),
        })
    return {
        "model": str(data.get("model", "")),
        "rows_per_stage": _integer(data.get("rows_per_stage")),
        "stages": stages,
        "status": str(data.get("status", "")),
        "note": "capacity evidence only; not counted as 49M model transfer",
    }


def build_scorecard(
    documents: Iterable[tuple[str, dict[str, Any]]],
    *,
    answerability_flags: dict[str, Any] | None = None,
    hardware: dict[str, Any] | None = None,
    held_out_flags: dict[str, Any] | None = None,
    capacity_probe: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic, content-free six-criterion scorecard."""
    bundle = _empty_bundle()
    kinds: dict[str, int] = {}
    input_sources = []
    for source, data in documents:
        if not isinstance(data, dict):
            raise ValueError(f"{source}: expected a JSON object")
        kind = _extract(source, data, bundle)
        kinds[kind] = kinds.get(kind, 0) + 1
        input_sources.append(source)

    for key, rows in bundle.items():
        if key != "unclassified":
            rows.sort(key=lambda row: row.get("source", ""))

    answerability, answerability_sources = _merged_answerability(
        bundle, answerability_flags
    )
    if answerability is not None:
        flags = answerability_flags or {}
        answerability["specs_failed"] = _integer(
            flags.get("answerability_specs_failed")
        )
        answerability["specs_total"] = _integer(
            flags.get("answerability_specs_total")
        )

    criteria = [
        _criterion_gap_discovery(bundle, answerability, answerability_sources),
        _criterion_learning_policy(bundle),
        _criterion_retention(bundle),
    ]
    held_out, selected = _heldout_tasks(bundle, held_out_flags)
    criteria.append(_criterion_transfer(bundle, held_out, selected))
    criteria.append(_criterion_meta_learning(bundle))
    criteria.append(_criterion_low_cost(bundle, hardware))

    status_counts = {"met": 0, "partial": 0, "not_met": 0, "unknown": 0}
    for criterion in criteria:
        status_counts[criterion["status"]] += 1
    hardware_values, hardware_extras = _hardware_values(bundle, hardware)
    card = {
        "schema": SCHEMA,
        "content_free": True,
        "criteria": criteria,
        "held_out_tasks": held_out,
        "hardware": {
            "envelope": {
                "cpu": 8,
                "ram_gb": 16,
                "disk_gb": 40,
            },
            "cpu": _round(hardware_values.get("cpu")),
            "ram_gb": _round(hardware_values.get("ram_gb")),
            "disk_gb": _round(hardware_values.get("disk_gb")),
            "model_params_m": _round(hardware_extras.get("model_params_m")),
            "adapter_params": hardware_extras.get("adapter_params"),
            "train_seconds": _round(hardware_extras.get("train_seconds")),
            "tests": hardware_extras.get("tests"),
        },
        "summary": {
            "criteria": len(criteria),
            "status_counts": status_counts,
            "scored_criteria": sum(
                1 for criterion in criteria if criterion["score"] is not None
            ),
            "aggregate_score": None,
            "note": (
                "No aggregate score is reported: the six criteria are not "
                "interchangeable and a single number would hide the failures."
            ),
        },
        "inputs": {
            "sources": sorted(set(input_sources)),
            "document_types": dict(sorted(kinds.items())),
            "unclassified": sorted(
                item["source"] for item in bundle["unclassified"]
            ),
        },
    }
    capacity = _capacity_probe_summary(capacity_probe)
    if capacity is not None:
        card["capacity_probe"] = capacity
    _validate_content_free(card)
    return card


def render_markdown(card: dict[str, Any]) -> str:
    lines = [
        "# Capability scorecard",
        "",
        "Content-free. No prompt, answer or dataset text is included.",
        "",
        f"Schema: `{card.get('schema', SCHEMA)}`",
        "",
        (
            "No aggregate score is reported. The six criteria are not "
            "interchangeable; status counts are shown instead."
        ),
        "",
        "## Six criteria",
        "",
        "| # | Criterion | Status | Score | Evidence |",
        "|---:|---|---|---:|---|",
    ]
    for index, criterion in enumerate(card["criteria"], start=1):
        score = criterion.get("score")
        score_text = "—" if score is None else f"{score:.2f}"
        evidence = criterion.get("evidence") or ["—"]
        cell = "<br>".join(
            str(item).replace("|", "\\|") for item in evidence[:2]
        )
        lines.append(
            f"| {index} | {criterion['name']} | {criterion['status']} | "
            f"{score_text} | {cell} |"
        )

    lines.extend(["", "## Criterion details", ""])
    for index, criterion in enumerate(card["criteria"], start=1):
        score = criterion.get("score")
        score_text = "—" if score is None else f"{score:.2f}"
        lines.append(f"### {index}. {criterion['name']}")
        lines.append("")
        lines.append(f"- **Status:** `{criterion['status']}` (score {score_text})")
        lines.append("- **Evidence:**")
        for item in criterion.get("evidence", []):
            lines.append(f"  - {item}")
        sources = criterion.get("sources") or []
        if sources:
            lines.append(
                "- **Sources:** "
                + ", ".join(f"`{source}`" for source in sources)
            )
        else:
            lines.append("- **Sources:** none")
        lines.append(f"- **Next measurable target:** {criterion['next_target']}")
        lines.append("")

    held = card["held_out_tasks"]
    lines.extend(
        [
            "## Held-out tasks",
            "",
            "| Task | n | Base exact | Adapter exact | Model transfer | Status |",
            "|---|---:|---:|---:|---|---|",
        ]
    )

    def exact(value: Any) -> str:
        number = _number(value)
        return "—" if number is None else f"{number:.3f}"

    rows = [
        (
            "city/river",
            held["copy_transfer"]["city_river"],
            "yes",
        ),
        (
            "3-4 arbitrary keys",
            held["copy_transfer"]["multi_key"],
            "yes",
        ),
        (
            "arithmetic",
            held["arithmetic_yaml"]["arithmetic"],
            "yes",
        ),
        (
            "city YAML",
            held["arithmetic_yaml"]["city_yaml"],
            "yes",
        ),
        (
            "runtime copy tool",
            held["runtime_copy_tool"],
            "no (tool use)",
        ),
    ]
    for name, entry, transfer in rows:
        lines.append(
            f"| {name} | {entry.get('examples') or '—'} | "
            f"{exact(entry.get('base_exact'))} | "
            f"{exact(entry.get('adapter_exact'))} | {transfer} | "
            f"{entry.get('status', 'unknown')} |"
        )
    lines.extend(
        [
            "",
            (
                "The runtime copy tool is a deterministic transcription tool. "
                "Its accuracy is **not** counted as model transfer."
            ),
        ]
    )
    capacity = card.get("capacity_probe")
    if capacity:
        lines.extend(
            [
                "",
                "## Capacity probe (not counted as 49M transfer)",
                "",
                f"Model: `{capacity.get('model', 'unknown')}`",
                "",
                "| Stage | Pairs | Rows | Baseline exact | Flat-prompt exact |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for row in capacity.get("stages", []):
            lines.append(
                f"| {row.get('stage', 'unknown')} | "
                f"{row.get('pairs_per_row', '?')} | {row.get('rows', '?')} | "
                f"{exact(row.get('baseline_exact'))} | "
                f"{exact(row.get('flat_prompt_exact'))} |"
            )
        lines.append("")
        lines.append(
            "This is capacity evidence about a larger pretrained model. "
            "It is not a from-scratch Atom Zero result and is excluded from "
            "the transfer score."
        )
    lines.extend(
        [
            "",
            "## Hardware",
            "",
            "| Resource | Value | Envelope |",
            "|---|---:|---:|",
        ]
    )
    hardware = card.get("hardware", {})
    envelope = hardware.get("envelope", {})
    for key, label in (
        ("cpu", "CPU"),
        ("ram_gb", "RAM (GB)"),
        ("disk_gb", "Disk (GB)"),
    ):
        value = hardware.get(key)
        limit = envelope.get(key)
        lines.append(
            f"| {label} | {value if value is not None else '—'} | "
            f"{'<=' + str(limit) if limit is not None else '—'} |"
        )
    for key, label in (
        ("model_params_m", "Base model (M params)"),
        ("adapter_params", "Adapter params"),
        ("train_seconds", "Recorded training (s)"),
        ("tests", "Tests recorded"),
    ):
        value = hardware.get(key)
        if value is not None:
            display = f"{value:,}" if isinstance(value, int) else value
            lines.append(f"| {label} | {display} | — |")

    sources = card.get("inputs", {}).get("sources", [])
    lines.extend(["", "## Input sources", ""])
    if sources:
        for source in sources:
            lines.append(f"- `{source}`")
    else:
        lines.append("- none")
    summary = card.get("summary", {})
    counts = summary.get("status_counts", {})
    lines.extend(
        [
            "",
            "## Status counts",
            "",
            (
                f"met {counts.get('met', 0)} · partial "
                f"{counts.get('partial', 0)} · not_met "
                f"{counts.get('not_met', 0)} · unknown "
                f"{counts.get('unknown', 0)}"
            ),
            "",
            "No aggregate score is reported.",
            "",
        ]
    )
    return "\n".join(lines)


def _load(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return data


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a content-free six-criterion capability scorecard from "
            "zero or more JSON result files."
        )
    )
    parser.add_argument(
        "--result",
        action="append",
        default=[],
        metavar="PATH",
        help="content-free result JSON; repeat for multiple files",
    )
    parser.add_argument(
        "--answerability",
        action="append",
        default=[],
        metavar="PATH",
        help="answerability profile JSON; repeat for multiple files",
    )
    parser.add_argument("--cpu", type=float, help="CPU cores of the test machine")
    parser.add_argument("--ram-gb", type=float, help="RAM in GB")
    parser.add_argument("--disk-gb", type=float, help="disk in GB")
    parser.add_argument(
        "--model-params-m", type=float, help="base model parameters in millions"
    )
    parser.add_argument("--adapter-params", type=int, help="adapter parameter count")
    parser.add_argument(
        "--train-seconds", type=float, help="longest recorded adapter training time"
    )
    parser.add_argument("--tests", type=int, help="recorded test count")
    parser.add_argument("--brier", type=float, help="answerability Brier score")
    parser.add_argument("--ece", type=float, help="answerability ECE")
    parser.add_argument("--refusal-recall", type=float)
    parser.add_argument("--refusal-precision", type=float)
    parser.add_argument("--answerable-coverage", type=float)
    parser.add_argument("--answerability-specs-failed", type=int)
    parser.add_argument("--answerability-specs-total", type=int)
    parser.add_argument("--city-river-exact", type=float)
    parser.add_argument("--city-river-examples", type=int)
    parser.add_argument("--multi-key-exact", type=float)
    parser.add_argument("--multi-key-examples", type=int)
    parser.add_argument("--arithmetic-exact", type=float)
    parser.add_argument("--city-yaml-exact", type=float)
    parser.add_argument("--arith-yaml-examples", type=int)
    parser.add_argument("--json-out", metavar="PATH")
    parser.add_argument("--md-out", metavar="PATH")
    parser.add_argument(
        "--capacity-probe",
        metavar="PATH",
        help="content-free large-base probe JSON (capacity evidence only)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="print JSON instead of the Markdown table",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    documents: list[tuple[str, dict[str, Any]]] = []
    try:
        for raw in args.result:
            path = Path(raw)
            documents.append((_source_label(path), _load(str(path))))
        for raw in args.answerability:
            path = Path(raw)
            documents.append((_source_label(path), _load(str(path))))
        capacity_probe = (
            _load(args.capacity_probe) if args.capacity_probe else None
        )
        card = build_scorecard(
            documents,
            answerability_flags={
                "brier": args.brier,
                "ece": args.ece,
                "refusal_recall": args.refusal_recall,
                "refusal_precision": args.refusal_precision,
                "answerable_coverage": args.answerable_coverage,
                "answerability_specs_failed": args.answerability_specs_failed,
                "answerability_specs_total": args.answerability_specs_total,
            },
            hardware={
                "cpu": args.cpu,
                "ram_gb": args.ram_gb,
                "disk_gb": args.disk_gb,
                "model_params_m": args.model_params_m,
                "adapter_params": args.adapter_params,
                "train_seconds": args.train_seconds,
                "tests": args.tests,
            },
            held_out_flags={
                "city_river_exact": args.city_river_exact,
                "city_river_examples": args.city_river_examples,
                "multi_key_exact": args.multi_key_exact,
                "multi_key_examples": args.multi_key_examples,
                "arithmetic_exact": args.arithmetic_exact,
                "city_yaml_exact": args.city_yaml_exact,
                "arith_yaml_examples": args.arith_yaml_examples,
            },
            capacity_probe=capacity_probe,
        )
        markdown = render_markdown(card)
        if args.json_out:
            Path(args.json_out).write_text(
                json.dumps(card, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n",
                encoding="utf-8",
            )
        if args.md_out:
            Path(args.md_out).write_text(markdown, encoding="utf-8")
        if args.json:
            sys.stdout.write(
                json.dumps(card, indent=2, sort_keys=True, ensure_ascii=False)
                + "\n"
            )
        else:
            sys.stdout.write(markdown + "\n")
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        sys.stderr.write(f"capability_scorecard: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
