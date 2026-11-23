"""Deterministic self-code audit and bounded patch proposals.

This is the first layer of self-development that looks at Atom Zero's own
repository instead of its documents. It does not edit code, call a model or
apply a patch. It scans for a small set of concrete risks, classifies findings,
and emits a human-gated proposal with allowed files and checks.

    atomzero-self-code --root . --json
    atomzero-self-code --root . --propose --json
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import json
import re
import sys
import tokenize
from pathlib import Path

__all__ = ["audit_report", "audit_tree", "propose_patch", "main"]

_SKIP_DIRS = {
    ".git", ".hg", ".svn", "__pycache__", "node_modules", "dist", "build",
    ".pytest_cache", ".mypy_cache", "results", "checkpoints",
}
_SCAN_SUFFIXES = {".py", ".md", ".toml", ".json", ".yml", ".yaml"}
_MARKER_RE = re.compile(r"\b(TODO|FIXME|XXX)\b", re.IGNORECASE)
_SHELL_CALLS = {"run", "call", "check_call", "check_output", "Popen"}
_BROAD_EXCEPTIONS = {"Exception", "BaseException"}


def _iter_files(root: Path):
    for path in sorted(root.rglob("*"), key=lambda item: str(item)):
        if not path.is_file() or path.suffix.lower() not in _SCAN_SUFFIXES:
            continue
        relative = path.relative_to(root)
        if any(part in _SKIP_DIRS for part in relative.parts):
            continue
        yield path


def _finding(kind, severity, path, line, message, evidence=""):
    key = f"{kind}|{path}|{line}|{message}"
    identifier = hashlib.sha256(key.encode("utf-8")).hexdigest()[:12]
    return {
        "finding_id": f"sc-{identifier}",
        "kind": kind,
        "severity": severity,
        "path": str(path),
        "line": int(line),
        "message": message,
        "evidence": str(evidence)[:200],
    }


def _comment_markers(source: str):
    """Return TODO/FIXME/XXX markers in real Python comments only."""
    markers = []
    try:
        tokens = tokenize.generate_tokens(io.StringIO(source).readline)
        for token in tokens:
            if token.type == tokenize.COMMENT and _MARKER_RE.search(token.string):
                markers.append((token.start[0], token.string.strip()))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        return []
    return markers


def _ast_code_findings(rel: str, source: str, is_test: bool = False):
    """Return code-boundary findings from the Python AST, not raw text."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    findings = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else (
                func.attr if isinstance(func, ast.Attribute) else "")
            if name in {"eval", "exec"} and isinstance(func, ast.Name):
                findings.append(_finding(
                    "dangerous_builtin", "high", rel, node.lineno,
                    "eval/exec is an arbitrary-code boundary",
                    ast.get_source_segment(source, node) or ""))
            if name in _SHELL_CALLS:
                for keyword in node.keywords:
                    if (keyword.arg == "shell"
                            and isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is True):
                        findings.append(_finding(
                            "shell_true", "high", rel, node.lineno,
                            "shell=True executes through the platform shell",
                            ast.get_source_segment(source, node) or ""))
            if is_test and name in {"now", "today"}:
                owner = func.value
                owner_name = owner.id if isinstance(owner, ast.Name) else (
                    owner.attr if isinstance(owner, ast.Attribute) else "")
                if owner_name in {"datetime", "date"}:
                    findings.append(_finding(
                        "wall_clock_test", "low", rel, node.lineno,
                        "test reads the wall clock; derive fixtures from a "
                        "fixed time",
                        ast.get_source_segment(source, node) or ""))
        if isinstance(node, ast.ExceptHandler):
            handler = node.type
            name = getattr(handler, "id", None)
            if name in _BROAD_EXCEPTIONS:
                findings.append(_finding(
                    "broad_exception", "medium", rel, node.lineno,
                    "broad exception may hide a failure; narrow or re-raise",
                    ast.get_source_segment(source, node) or ""))
    return findings


def audit_tree(root=".", max_per_kind=20) -> list[dict]:
    """Scan the repository and return deterministic, bounded findings."""
    base = Path(root).expanduser().resolve()
    if not base.is_dir():
        raise ValueError(f"root is not a directory: {base}")
    if max_per_kind < 1:
        raise ValueError("max_per_kind must be positive")
    findings: list[dict] = []
    seen = {kind: 0 for kind in (
        "wall_clock_test", "broad_exception", "shell_true",
        "dangerous_builtin", "todo_marker")}

    def add(finding: dict) -> None:
        kind = finding["kind"]
        if seen.get(kind, 0) >= max_per_kind:
            return
        findings.append(finding)
        seen[kind] = seen.get(kind, 0) + 1

    for path in _iter_files(base):
        rel = path.relative_to(base).as_posix()
        is_test = rel.startswith("tests/")
        is_python = path.suffix.lower() == ".py"
        try:
            source = path.read_text(encoding="utf-8-sig")
        except (OSError, UnicodeDecodeError):
            continue
        lines = source.splitlines()

        if is_python:
            for finding in _ast_code_findings(rel, source, is_test=is_test):
                add(finding)
            for number, marker in _comment_markers(source):
                add(_finding(
                    "todo_marker", "low", rel, number,
                    "unresolved task marker", marker))
        else:
            for number, line in enumerate(lines, start=1):
                if _MARKER_RE.search(line):
                    add(_finding(
                        "todo_marker", "low", rel, number,
                        "unresolved task marker", line.strip()))

    severity_order = {"high": 0, "medium": 1, "low": 2}

    def rank(item):
        path = item["path"]
        if path.startswith("src/"):
            location = 0
        elif path.startswith("tests/"):
            location = 1
        elif path.startswith("docs/"):
            location = 2
        else:
            location = 3
        return (severity_order.get(item["severity"], 3), location,
                item["kind"], path, item["line"])

    findings.sort(key=rank)
    return findings


def _tests_for(path: str) -> str:
    stem = Path(path).stem
    return f"tests/test_{stem}.py"


def _allowed_with_test(path: str) -> list[str]:
    allowed = [path]
    test = _tests_for(path)
    if path.startswith("src/") and test != path:
        allowed.append(test)
    return allowed


def propose_patch(finding: dict) -> dict:
    """Turn one finding into a bounded, human-gated patch proposal."""
    if not isinstance(finding, dict):
        raise ValueError("finding must be an object")
    kind = str(finding.get("kind") or "")
    path = str(finding.get("path") or "")
    if not kind or not path:
        raise ValueError("finding.kind and finding.path are required")
    if kind == "wall_clock_test":
        allowed = [path]
        checks = [f"pytest:{path}"]
        goal = "replace the wall-clock fixture with a fixed timestamp"
        risk = "low"
    elif kind == "broad_exception":
        allowed = _allowed_with_test(path)
        checks = [f"pytest:{_tests_for(path)}"]
        goal = "narrow the broad exception or re-raise after logging"
        risk = "medium"
    elif kind == "todo_marker":
        allowed = _allowed_with_test(path)
        checks = [f"pytest:{_tests_for(path)}"]
        goal = "resolve or explicitly schedule the task marker"
        risk = "medium"
    elif kind in {"shell_true", "dangerous_builtin"}:
        allowed = [path]
        checks = []
        goal = "manually review the arbitrary-code boundary before any edit"
        risk = "high"
    else:
        allowed = [path]
        checks = []
        goal = "review the finding manually"
        risk = "medium"
    identifier = hashlib.sha256(
        f"{finding.get('finding_id')}|{kind}|{path}".encode("utf-8")
    ).hexdigest()[:12]
    return {
        "proposal_id": f"prop-{identifier}",
        "finding_id": finding.get("finding_id", ""),
        "kind": kind,
        "severity": finding.get("severity", "medium"),
        "path": path,
        "goal": goal,
        "risk": risk,
        "allowed_files": allowed,
        "checks": checks,
        "requires_human": True,
        "auto_apply": False,
        "weights_changed": False,
        "method": "deterministic_self_code_proposal_v1",
    }


def audit_report(root=".", max_per_kind=20, propose=False) -> dict:
    findings = audit_tree(root, max_per_kind=max_per_kind)
    counts = {}
    for finding in findings:
        counts[finding["kind"]] = counts.get(finding["kind"], 0) + 1
    proposals = [propose_patch(item) for item in findings] if propose else []
    return {
        "root": str(Path(root).expanduser().resolve()),
        "findings": findings,
        "counts": dict(sorted(counts.items())),
        "proposals": proposals,
        "summary": {
            "findings": len(findings),
            "high": sum(1 for item in findings if item["severity"] == "high"),
            "medium": sum(
                1 for item in findings if item["severity"] == "medium"),
            "low": sum(1 for item in findings if item["severity"] == "low"),
            "proposals": len(proposals),
        },
        "auto_apply": False,
        "weights_changed": False,
        "method": "deterministic_self_code_audit_v1",
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--max-per-kind", type=int, default=20)
    parser.add_argument("--propose", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    try:
        report = audit_report(
            args.root, max_per_kind=args.max_per_kind,
            propose=args.propose)
    except (OSError, ValueError) as exc:
        print(f"input error: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(
            f"Self-code audit: {report['summary']['findings']} findings "
            f"({report['summary']['high']} high, "
            f"{report['summary']['medium']} medium, "
            f"{report['summary']['low']} low)")
        for finding in report["findings"][:20]:
            print(f"- {finding['severity']} {finding['kind']} "
                  f"{finding['path']}:{finding['line']} "
                  f"{finding['message']}")
        if report["proposals"]:
            print(f"proposals: {len(report['proposals'])} (human-gated)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
