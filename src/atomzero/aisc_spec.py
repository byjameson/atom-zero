"""Deterministic AISC spec generator from measured Atom Zero evidence.

This module is deliberately independent of AISC: it reads a measured
answerability profile and renders the AISC TOML text itself. AISC remains an
optional, external checker.

The generator copies measured metrics without rounding them into a pass and
keeps the refusal-policy targets fixed at their required values. A failing
measurement therefore stays visible in the generated spec.

    python -m atomzero.aisc_spec
    python -m atomzero.aisc_spec --check
    python -m atomzero.aisc_spec --generated-on 2026-11-23
"""
from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROFILE = Path(
    "benchmarks/results/answerability_hybrid/answerability_profile.json")
DEFAULT_OUTPUT = Path("docs/aisc/atomzero-answerability-generated.toml")

REQUIRED_KEYS = (
    "n",
    "brier",
    "ece",
    "threshold",
    "refusal_precision",
    "refusal_recall",
    "answerable_coverage",
    "commit",
    "dataset_hash",
    "calibration_date",
    "recalibration_due",
)
METRIC_KEYS = (
    "brier",
    "ece",
    "threshold",
    "refusal_precision",
    "refusal_recall",
    "answerable_coverage",
)
TEXT_KEYS = ("commit", "dataset_hash")
DATE_KEYS = ("calibration_date", "recalibration_due")

SPEC_NAME = "AtomZeroAnswerabilityGenerated"
SPEC_VERSION = "0.1"
SPEC_DESCRIPTION = (
    "Generated from measured held-out answerability and refusal-policy evidence")
MIN_ASSURANCE = 0.80
MAX_CONTEXT_TOKENS = 50_000
COMPONENT_ASSURANCE = 0.80
COMPONENT_CONTEXT_TOKENS = 2_000
MIN_REFUSAL_PRECISION = 0.90
MIN_REFUSAL_RECALL = 0.90
MIN_ANSWERABLE_COVERAGE = 0.80


def _iso_date(value, where: str) -> str:
    text = str(value).strip()
    try:
        parsed = date.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{where} must be YYYY-MM-DD: {value!r}") from exc
    return parsed.isoformat()


def _validated_evidence(data, where: str) -> dict:
    """Return the required evidence fields, or raise without inventing any."""
    if not isinstance(data, dict):
        raise ValueError(f"{where}: evidence must be a mapping")
    missing = [key for key in REQUIRED_KEYS if key not in data]
    if missing:
        raise ValueError(
            f"{where}: missing required keys: {', '.join(missing)}")

    n = data["n"]
    if isinstance(n, bool) or not isinstance(n, int) or n < 0:
        raise ValueError(f"{where}: 'n' must be a non-negative integer")

    values = {"n": n}
    for key in METRIC_KEYS:
        raw = data[key]
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"{where}: '{key}' must be a number")
        number = float(raw)
        if not math.isfinite(number) or not 0.0 <= number <= 1.0:
            raise ValueError(f"{where}: '{key}' must be within [0, 1]")
        values[key] = number

    for key in TEXT_KEYS:
        raw = data[key]
        if not isinstance(raw, str) or not raw.strip():
            raise ValueError(f"{where}: '{key}' must be a non-empty string")
        values[key] = raw.strip()

    for key in DATE_KEYS:
        values[key] = _iso_date(data[key], f"{where}: '{key}'")

    return values


def _source_path(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def _default_generated_on(raw: dict, values: dict) -> str:
    """Use the evidence profile's own date so the artifact is reproducible."""
    created = raw.get("created")
    if isinstance(created, str):
        candidate = created.strip()[:10]
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError:
            pass
    return values["calibration_date"]


def collect_evidence(root, profile_path=None, *, generated_on=None) -> dict:
    """Read measured answerability evidence without inventing values.

    ``source`` is stored relative to ``root`` when possible so the generated
    artifact does not embed a machine-specific checkout path.
    """
    base = Path(root).expanduser()
    if profile_path is None:
        path = base / DEFAULT_PROFILE
    else:
        path = Path(profile_path).expanduser()
        if not path.is_absolute():
            path = base / path

    raw = json.loads(path.read_text(encoding="utf-8"))
    values = _validated_evidence(raw, str(path))
    stamp = (_iso_date(generated_on, "generated_on")
             if generated_on is not None
             else _default_generated_on(raw, values))

    evidence = {
        "source": _source_path(path, base),
        "generated_on": stamp,
    }
    evidence.update(values)
    return evidence


def _resolve_generated_on(evidence: dict, generated_on) -> str:
    if generated_on is not None:
        return _iso_date(generated_on, "generated_on")
    if evidence.get("generated_on") is not None:
        return _iso_date(evidence["generated_on"], "generated_on")
    return _iso_date(evidence["calibration_date"], "calibration_date")


def _toml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def build_spec(evidence, generated_on=None) -> str:
    """Render canonical AISC TOML from measured evidence.

    Field order, numeric formatting and line endings are fixed. Targets are
    constants: evidence cannot lower them.
    """
    values = _validated_evidence(evidence, "evidence")
    stamp = _resolve_generated_on(evidence, generated_on)

    parts = ["held-out answerability experiment"]
    source = evidence.get("source")
    if isinstance(source, str) and source.strip():
        parts.append(f"source={source.strip()}")
    parts.extend([
        f"generated_on={stamp}",
        f"n={values['n']}",
        f"commit={values['commit']}",
        f"dataset_hash={values['dataset_hash']}",
        f"calibration_date={values['calibration_date']}",
        f"recalibration_due={values['recalibration_due']}",
    ])
    evidence_text = "; ".join(parts)

    lines = [
        f"name = {_toml_string(SPEC_NAME)}",
        f"version = {_toml_string(SPEC_VERSION)}",
        f"description = {_toml_string(SPEC_DESCRIPTION)}",
        f"min_assurance = {MIN_ASSURANCE:.2f}",
        f"max_context_tokens = {MAX_CONTEXT_TOKENS}",
        f"answerability_n = {values['n']}",
        f"answerability_brier = {values['brier']:.6f}",
        f"answerability_ece = {values['ece']:.6f}",
        f"refusal_precision = {values['refusal_precision']:.6f}",
        f"refusal_recall = {values['refusal_recall']:.6f}",
        f"answerable_coverage = {values['answerable_coverage']:.6f}",
        f"refusal_threshold = {values['threshold']:.6f}",
        f"min_refusal_precision = {MIN_REFUSAL_PRECISION:.2f}",
        f"min_refusal_recall = {MIN_REFUSAL_RECALL:.2f}",
        f"min_answerable_coverage = {MIN_ANSWERABLE_COVERAGE:.2f}",
        "",
        "[[components]]",
        f"name = {_toml_string('AnswerabilitySignal')}",
        f"input_type = {_toml_string('Query')}",
        f"output_type = {_toml_string('Decision')}",
        f"kind = {_toml_string('model')}",
        f"assurance = {COMPONENT_ASSURANCE:.2f}",
        "verified = true",
        f"evidence = {_toml_string(evidence_text)}",
        f"guarantees = [{_toml_string('calibrated answerability probability')}]",
        f"context_tokens = {COMPONENT_CONTEXT_TOKENS}",
    ]
    return "\n".join(lines) + "\n"


def _write_atomic(target: Path, text: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            fd = -1
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
    finally:
        if fd != -1:
            try:
                os.close(fd)
            except OSError:
                pass
        try:
            if temp.exists():
                temp.unlink()
        except OSError:
            pass


def write_spec(path, evidence, check=False, generated_on=None) -> dict:
    """Compare or atomically write the canonical spec.

    ``check=True`` never writes. ``ok`` means the file already matches the
    canonical text; ``changed`` is true when it does not exist or differs.
    """
    target = Path(path).expanduser()
    text = build_spec(evidence, generated_on=generated_on)
    payload = text.encode("utf-8")
    try:
        existing = target.read_bytes()
    except FileNotFoundError:
        existing = None
    changed = existing != payload

    if check:
        return {"ok": not changed, "changed": changed, "path": str(target)}

    if changed:
        _write_atomic(target, text)
    return {"ok": True, "changed": changed, "path": str(target)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", default=None,
        help="answerability profile JSON (default: the hybrid profile)")
    parser.add_argument(
        "--out", default=str(ROOT / DEFAULT_OUTPUT),
        help="generated AISC TOML path")
    parser.add_argument(
        "--check", action="store_true",
        help="compare only; exit 1 when the spec is missing or drifted")
    parser.add_argument(
        "--generated-on", default=None,
        help="YYYY-MM-DD stamp (default: the evidence profile date)")
    parser.add_argument(
        "--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    try:
        evidence = collect_evidence(
            ROOT, args.profile, generated_on=args.generated_on)
        result = write_spec(
            args.out, evidence, check=args.check,
            generated_on=args.generated_on)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
        return 2

    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    elif args.check:
        state = "ok" if result["ok"] else "drift"
        print(f"{state}: {result['path']}")
    elif result["changed"]:
        print(f"wrote {result['path']}")
    else:
        print(f"unchanged {result['path']}")

    if args.check and not result["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
