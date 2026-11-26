"""Preflight diagnostics for an Atom Zero install.

    atomzero-doctor
    atomzero-doctor --json
    atomzero-doctor --cache ~/.cache/atomzero/atom-zero-49m --port 8013

The doctor separates hard failures from optional warnings. It never downloads
the model; it checks what is already on disk.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import socket
import sys
import tempfile
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path

from atomzero import __version__

EXPECTED_CACHE = {
    "model.pt": 278_746_911,
    "config.json": 1327,
    "chat_en_bpe8k.json": 551_291,
}
DEFAULT_CACHE = Path.home() / ".cache/atomzero/atom-zero-63m-balanced"


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    hint: str = ""


def check_python() -> Check:
    version = platform.python_version()
    if sys.version_info >= (3, 10):
        return Check("python", "ok", f"{version}")
    return Check("python", "fail", f"{version}", "Atom Zero requires Python 3.10+")


def check_package() -> Check:
    return Check("atomzero", "ok", __version__)


def check_deps() -> list[Check]:
    checks = []
    for module, label in (
        ("tokenizers", "tokenizers"),
        ("numpy", "numpy"),
        ("pypdf", "pypdf"),
        ("docx", "python-docx"),
    ):
        found = importlib.util.find_spec(module) is not None
        checks.append(
            Check(
                f"dependency:{label}",
                "ok" if found else "fail",
                "installed" if found else "missing",
                "" if found else f"pip install {label}",
            )
        )
    return checks


def check_torch() -> Check:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - environment dependent
        return Check("torch", "fail", f"import failed: {exc}", "pip install torch")
    detail = f"{torch.__version__}"
    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            detail += f" | CUDA {torch.version.cuda} | {name} ({total:.1f} GB)"
        except Exception:
            detail += " | CUDA available"
        return Check("torch", "ok", detail)
    return Check(
        "torch",
        "warn",
        f"{detail} | CPU only",
        "CPU works, but generation is slower; install a CUDA build for speed",
    )


def check_cache(cache_dir: Path, expected: dict[str, int] | None = None) -> list[Check]:
    cache_dir = Path(cache_dir).expanduser()
    expected = EXPECTED_CACHE if expected is None else expected
    checks = []
    for name, size in expected.items():
        path = cache_dir / name
        if not path.exists():
            checks.append(
                Check(
                    f"cache:{name}",
                    "fail",
                    f"missing: {path}",
                    "run `atomzero-quickstart` to download the model",
                )
            )
        elif path.stat().st_size != size:
            checks.append(
                Check(
                    f"cache:{name}",
                    "fail",
                    f"{path.stat().st_size} bytes, expected {size}",
                    "delete the file and run `atomzero-quickstart` again",
                )
            )
        else:
            checks.append(Check(f"cache:{name}", "ok", f"{size} bytes"))
    return checks


def check_writable(path: Path, label: str) -> Check:
    path = Path(path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".atomzero-", delete=True):
            pass
        return Check(label, "ok", f"writable: {path.parent}")
    except Exception as exc:
        return Check(label, "fail", f"not writable: {path.parent} ({exc})", "choose another path")


def check_docs_index(path: Path) -> Check:
    """Validate an existing document index without creating one."""
    path = Path(path).expanduser()
    if not path.exists():
        return Check(
            "docs-index",
            "ok",
            f"no index yet: {path}",
            "created by `atomzero-quickstart` or `atomzero-docs add`",
        )
    try:
        from atomzero.doc_memory import DocMemory

        memory = DocMemory(path)
        try:
            result = memory.integrity()
        finally:
            memory.close()
    except Exception as exc:
        return Check(
            "docs-index",
            "fail",
            f"could not open {path}: {exc}",
            "check file permissions or move/rebuild the index",
        )
    if not result["ok"]:
        return Check(
            "docs-index",
            "fail",
            f"{path} | " + "; ".join(result["issues"]),
            "run `atomzero-docs doctor --db <path>` for details",
        )
    return Check(
        "docs-index",
        "ok",
        f"{path} | {result['chunks']} chunks, {result['sources']} sources",
    )


def check_port(host: str, port: int) -> Check:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError as exc:
            return Check(
                "port",
                "fail",
                f"{host}:{port} is not available ({exc})",
                "use `atomzero-quickstart --port 8014`",
            )
    return Check("port", "ok", f"{host}:{port} is free")


def check_provider(url: str) -> Check:
    if not url:
        return Check("provider", "warn", "not configured", "optional: `atomzero-quickstart --provider`")
    base = url.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    tags_url = base + "/api/tags"
    try:
        with urllib.request.urlopen(tags_url, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8", "replace"))
        models = [m.get("name", "?") for m in payload.get("models", [])]
        if models:
            return Check("provider", "ok", f"{tags_url} | {', '.join(models[:4])}")
        return Check("provider", "warn", f"{tags_url} reachable, no models", "ollama pull llama3.2:3b")
    except Exception as exc:
        return Check(
            "provider",
            "warn",
            f"{tags_url} not reachable ({exc})",
            "optional: start Ollama with `ollama serve`",
        )


def check_aisc(aisc_path: str = "", strict: bool = True,
               require_calibration: bool = True) -> Check:
    """Run the optional AISC architecture audit without making it a dependency."""
    if not aisc_path:
        return Check(
            "aisc",
            "warn",
            "not configured",
            "optional: `atomzero-doctor --aisc --aisc-path <checkout>`",
        )
    try:
        from atomzero.audit import discover_specs, run_audit
        results = run_audit(
            [str(path) for path in discover_specs()], aisc_path,
            strict=strict, require_calibration=require_calibration)
    except Exception as exc:
        return Check(
            "aisc",
            "warn",
            f"unavailable: {exc}",
            "optional: install AISC or pass a valid --aisc-path",
        )
    critical = sum(int(item.get("critical_count", 0)) for item in results)
    warnings = sum(int(item.get("warning_count", 0)) for item in results)
    failed = [item for item in results if not item.get("ok")]
    status = "ok" if not failed else ("fail" if critical or strict else "warn")
    detail = (f"specs={len(results)} critical={critical} "
              f"warnings={warnings} failed={len(failed)}")
    return Check(
        "aisc", status, detail,
        "" if not failed else "run `atomzero-audit --strict` for details")


def check_disk(path: Path) -> Check:
    path = Path(path).expanduser()
    try:
        path.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(path).free / 1e9
        return Check("disk", "ok" if free >= 1.0 else "warn", f"{free:.1f} GB free at {path}")
    except Exception as exc:
        return Check("disk", "warn", f"could not read {path}: {exc}")


def run_checks(args) -> list[Check]:
    checks = [check_python(), check_package()]
    checks.extend(check_deps())
    checks.append(check_torch())
    checks.extend(check_cache(Path(args.cache)))
    checks.append(check_writable(Path(args.docs), "docs-path"))
    checks.append(check_docs_index(Path(args.docs)))
    checks.append(check_writable(Path(args.sessions) / "session.json", "sessions"))
    checks.append(check_port(args.host, args.port))
    checks.append(check_provider(args.provider_url if args.provider else ""))
    if getattr(args, "aisc", False) or getattr(args, "aisc_path", ""):
        checks.append(check_aisc(
            args.aisc_path or os.environ.get("AISC_PATH", ""),
            strict=getattr(args, "aisc_strict", True),
            require_calibration=True))
    checks.append(check_disk(Path(args.cache).expanduser().parent))
    return checks


def render(checks: list[Check]) -> str:
    lines = [f"Atom Zero doctor {__version__}"]
    for check in checks:
        lines.append(f"[{check.status.upper():4}] {check.name:24} {check.detail}")
        if check.hint:
            lines.append(f"       hint: {check.hint}")
    failures = sum(1 for check in checks if check.status == "fail")
    warnings = sum(1 for check in checks if check.status == "warn")
    lines.append(f"summary: {failures} fail, {warnings} warn")
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-doctor", description="Check an Atom Zero install.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--docs", default="data/atomzero_demo.db")
    parser.add_argument("--sessions", default="data/sessions")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--provider", action="store_true",
                        help="check the optional Ollama provider")
    parser.add_argument("--provider-url", default="http://127.0.0.1:11434/v1")
    parser.add_argument("--aisc", action="store_true",
                        help="run the optional AISC architecture audit")
    parser.add_argument("--aisc-path", default="",
                        help="AISC checkout (or set AISC_PATH)")
    parser.add_argument("--aisc-strict", action=argparse.BooleanOptionalAction,
                        default=True,
                        help="treat AISC warnings as failures")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    checks = run_checks(args)
    failures = sum(1 for check in checks if check.status == "fail")
    if args.json:
        print(json.dumps({
            "version": __version__,
            "ok": failures == 0,
            "checks": [asdict(check) for check in checks],
        }, indent=2))
    else:
        print(render(checks))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
