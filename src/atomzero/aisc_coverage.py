"""Evidence and verification coverage for AISC specs.

A passing AISC check is not the same as a complete spec. This module reports
whether components are verified, whether their evidence strings point at real
files, whether calibration evidence exists and how large the context budget is.
It imports AISC only when a report is requested.

    atomzero-coverage
    atomzero-coverage --json
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from atomzero.audit import discover_specs, resolve_aisc_path, resolve_spec_path

ROOT = Path(__file__).resolve().parents[2]
_EVIDENCE_RE = re.compile(r"[A-Za-z0-9_./\\-]+\.py")


def evidence_files(evidence: str) -> list[str]:
    """Evidence metnindeki .py dosya yollarını deterministik olarak çıkarır."""
    if not evidence:
        return []
    found = []
    seen = set()
    for match in _EVIDENCE_RE.findall(str(evidence)):
        normalized = match.replace("\\", "/").lstrip("./")
        if normalized not in seen:
            seen.add(normalized)
            found.append(normalized)
    return found


def component_coverage(component, project_root: str | Path = ROOT) -> dict:
    """Tek bir bileşenin kanıt ve doğrulama durumunu döndürür."""
    root = Path(project_root)
    contract = getattr(component, "contract", None)
    evidence = str(getattr(contract, "evidence", "") or "")
    files = evidence_files(evidence)
    missing = [name for name in files if not (root / name).is_file()]
    verified = bool(getattr(contract, "verified", False))
    calibration_n = int(getattr(contract, "calibration_n", 0) or 0)
    return {
        "name": getattr(component, "name", ""),
        "kind": getattr(component, "kind", ""),
        "verified": verified,
        "has_evidence": bool(evidence.strip()),
        "evidence_files": files,
        "missing_evidence_files": missing,
        "calibration_n": calibration_n,
        "reversible": bool(getattr(component, "reversible", True)),
    }


def spec_coverage(definition, project_root: str | Path = ROOT) -> dict:
    """Bir AISC spec'inin bileşen kapsamını özetler."""
    components = list(definition.composition.components)
    entries = [component_coverage(item, project_root) for item in components]
    missing = sorted({
        name for entry in entries for name in entry["missing_evidence_files"]})
    unverified = [entry["name"] for entry in entries if not entry["verified"]]
    uncalibrated = [
        entry["name"] for entry in entries if entry["calibration_n"] == 0]
    irreversible = [
        entry["name"] for entry in entries if not entry["reversible"]]
    data = definition.to_dict()
    return {
        "source": definition.source,
        "name": definition.name,
        "version": definition.version,
        "topology": data.get("topology", "unknown"),
        "components": entries,
        "summary": {
            "components": len(entries),
            "verified": sum(1 for entry in entries if entry["verified"]),
            "with_evidence": sum(
                1 for entry in entries if entry["has_evidence"]),
            "missing_evidence_files": missing,
            "unverified_components": unverified,
            "uncalibrated_components": uncalibrated,
            "irreversible_components": irreversible,
            "context_tokens": int(data.get("context_tokens", 0) or 0),
        },
        "ok": not missing and not unverified,
    }


def coverage_report(
    specs,
    aisc_path: str = "",
    project_root: str | Path = ROOT,
) -> dict:
    """Birden fazla spec için kapsam raporu üretir."""
    aisc = resolve_aisc_path(aisc_path)
    if aisc is not None:
        sys.path.insert(0, str(aisc))
    try:
        from aisc.spec import load_spec
    except ImportError as exc:
        raise RuntimeError(
            "AISC is not installed. Pass --aisc-path or set AISC_PATH.") from exc
    entries = []
    for raw in specs:
        path = resolve_spec_path(raw, root=project_root)
        try:
            definition = load_spec(path)
            entries.append(spec_coverage(definition, project_root))
        except (OSError, ValueError, TypeError, KeyError) as exc:
            # Expected bad-spec failures are reported per spec; unexpected
            # programming errors are allowed to propagate.
            entries.append({
                "source": str(path),
                "name": None,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "summary": {},
            })
    errors = sum(1 for entry in entries if "error" in entry)
    missing = sum(
        1 for entry in entries
        if entry.get("summary", {}).get("missing_evidence_files"))
    unverified = sum(
        1 for entry in entries
        if entry.get("summary", {}).get("unverified_components"))
    return {
        "specs": entries,
        "summary": {
            "specs": len(entries),
            "ok": sum(1 for entry in entries if entry.get("ok")),
            "errors": errors,
            "specs_with_missing_evidence": missing,
            "specs_with_unverified_components": unverified,
        },
        "ok": errors == 0 and missing == 0 and unverified == 0,
    }


def _render(report: dict) -> str:
    summary = report["summary"]
    lines = [
        f"AISC coverage — {summary['specs']} specs, {summary['ok']} clean, "
        f"{summary['errors']} errors, "
        f"{summary['specs_with_missing_evidence']} with missing evidence, "
        f"{summary['specs_with_unverified_components']} with unverified components",
    ]
    for entry in report["specs"]:
        if "error" in entry:
            lines.append(f"- ERROR {entry['source']}: {entry['error']}")
            continue
        spec_summary = entry["summary"]
        flags = []
        if spec_summary["missing_evidence_files"]:
            flags.append("missing:" + ",".join(spec_summary["missing_evidence_files"]))
        if spec_summary["unverified_components"]:
            flags.append("unverified:" + ",".join(spec_summary["unverified_components"]))
        lines.append(
            f"- {'PASS' if entry['ok'] else 'FAIL'} {entry['name']} "
            f"({spec_summary['verified']}/{spec_summary['components']} verified, "
            f"{spec_summary['with_evidence']} with evidence"
            + (f", {'; '.join(flags)}" if flags else "") + ")")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("specs", nargs="*", default=[])
    parser.add_argument(
        "--aisc-path", default=os.environ.get("AISC_PATH", ""))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    specs = args.specs or [str(path) for path in discover_specs()]
    if not specs:
        print("No AISC specs found under docs/aisc.", file=sys.stderr)
        return 2
    try:
        report = coverage_report(specs, args.aisc_path)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(_render(report))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
