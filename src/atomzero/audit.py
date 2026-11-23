r"""Optional AISC architecture audit for Atom Zero.

AISC is a separate local prototype; this module imports it only when the
command runs. It is the packaged version of ``scripts/aisc_audit.py`` and is
available as ``atomzero-audit`` after installation.

    atomzero-audit --aisc-path "C:\path\to\ai-systems-algebra" --strict
    python -m atomzero.audit --json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC_GLOB = "docs/aisc/*.toml"


def discover_specs(root: str | Path | None = None) -> list[Path]:
    bases = [Path(root) if root is not None else ROOT, Path.cwd()]
    for base in bases:
        specs = sorted(base.glob(SPEC_GLOB))
        if specs:
            return specs
    return []


def resolve_aisc_path(value: str | None) -> Path | None:
    if not value:
        return None
    base = Path(value).expanduser().resolve()
    for candidate in (base, base / "ai-systems-algebra"):
        if (candidate / "aisc").is_dir():
            return candidate
    raise SystemExit(f"AISC package not found under: {base}")


def resolve_spec_path(value: str | Path, root: str | Path = ROOT) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(root) / path


def audit_definition(definition, strict: bool = False,
                     require_calibration: bool = False) -> dict:
    from aisc.core import Severity

    spec = definition.spec
    if require_calibration:
        spec = replace(spec, require_calibration=True)
    issues = definition.system.check(spec)
    critical = sum(1 for issue in issues
                   if issue.severity is Severity.CRITICAL)
    warnings = sum(1 for issue in issues
                   if issue.severity is Severity.WARNING)
    failed = critical > 0 or (strict and warnings > 0)
    return {
        "source": definition.source,
        "name": definition.name,
        "version": definition.version,
        "ok": not failed,
        "critical_count": critical,
        "warning_count": warnings,
        "issues": [issue.to_dict() for issue in issues],
        "report": definition.system.report(spec),
        "data": definition.to_dict(),
    }


def summarize_results(results: list[dict]) -> dict:
    """Toplu denetim sonuçlarını tek bir sayıya indirger."""
    total = len(results)
    errors = sum(1 for result in results if "error" in result)
    passed = sum(1 for result in results if result.get("ok") is True)
    failed = total - passed
    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "ok": failed == 0 and total > 0,
    }


def run_audit(specs, aisc_path: str = "", strict: bool = False,
              require_calibration: bool = False) -> list[dict]:
    aisc = resolve_aisc_path(aisc_path)
    if aisc is not None:
        sys.path.insert(0, str(aisc))
    try:
        from aisc.spec import load_spec
    except ImportError as exc:
        raise RuntimeError(
            "AISC is not installed. Pass --aisc-path or set AISC_PATH.") from exc
    results = []
    for raw in specs:
        path = resolve_spec_path(raw)
        try:
            definition = load_spec(path)
            results.append(audit_definition(
                definition, strict, require_calibration))
        except Exception as exc:
            results.append({
                "source": str(path),
                "ok": False,
                "error": str(exc),
            })
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "specs", nargs="*", default=[],
        help="AISC TOML specs (default: docs/aisc/*.toml)")
    parser.add_argument(
        "--aisc-path", default=os.environ.get("AISC_PATH", ""),
        help="path to the ai-systems-algebra checkout")
    parser.add_argument("--strict", action="store_true",
                        help="treat warnings as failures")
    parser.add_argument("--require-calibration", action="store_true",
                        help="require AISC high-assurance calibration evidence")
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--summary", action="store_true",
        help="print only the aggregate pass/fail/error summary")
    parser.add_argument(
        "--coverage", action="store_true",
        help="report evidence/verification coverage instead of checking specs")
    args = parser.parse_args(argv)
    specs = args.specs or [str(path) for path in discover_specs()]
    if not specs:
        print("No AISC specs found under docs/aisc.", file=sys.stderr)
        return 2
    if args.coverage:
        try:
            from atomzero.aisc_coverage import coverage_report
            report = coverage_report(specs, args.aisc_path)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 2
        if args.json:
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            from atomzero.aisc_coverage import _render
            print(_render(report))
        return 0 if report["ok"] else 1
    try:
        results = run_audit(
            specs, args.aisc_path, args.strict, args.require_calibration)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 2
    summary = summarize_results(results)
    if args.summary:
        if args.json:
            print(json.dumps(summary, indent=2, ensure_ascii=False))
        else:
            print(
                f"summary: {summary['total']} specs, "
                f"{summary['passed']} pass, {summary['failed']} fail, "
                f"{summary['errors']} error")
        return 0 if summary["ok"] else 1
    if args.json:
        print(json.dumps(results, indent=2, ensure_ascii=False))
    else:
        for index, result in enumerate(results):
            if index:
                print("\n" + "=" * 72 + "\n")
            if "error" in result:
                print(f"[FAIL] {result['source']}: {result['error']}")
            else:
                print(result["report"])
                if args.strict:
                    print(
                        f"Strict: {'PASS' if result['ok'] else 'FAIL'} "
                        f"({result['critical_count']} critical, "
                        f"{result['warning_count']} warnings)")
                else:
                    note = ("; warnings are informational, use --strict"
                            if result["warning_count"] else "")
                    print(
                        f"Result: {'PASS' if result['ok'] else 'FAIL'} "
                        f"({result['critical_count']} critical, "
                        f"{result['warning_count']} warnings{note})")
    return 0 if all(result["ok"] for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
