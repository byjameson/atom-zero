"""Verify that the public Atom Zero release surfaces agree on a version.

    python scripts/verify_release.py --version 0.1.8
    python scripts/verify_release.py --version 0.1.8 --devto-url https://...

Public endpoints only; no tokens are required.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYPI_PACKAGE = "atomzero"
HF_REPO = "byjameson/atom-zero-49m"
MODELSCOPE_REPO = "byjameson/atom-zero-49m"
GITHUB_REPO = "byjameson/atom-zero"


def fetch(url: str) -> str:
    request = urllib.request.Request(
        url, headers={"User-Agent": "atomzero-release-check"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode("utf-8", "replace")


def check(ok: bool, label: str, detail: str) -> bool:
    print(f"[{'OK' if ok else 'FAIL'}] {label}: {detail}")
    return ok


def version_from_pyproject() -> str:
    import tomllib
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    return data["project"]["version"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", default=version_from_pyproject())
    parser.add_argument("--devto-url", default="")
    args = parser.parse_args(argv)
    version = args.version
    expected = f"v{version}"
    failures = 0

    # PyPI
    try:
        data = json.loads(fetch(
            f"https://pypi.org/pypi/{PYPI_PACKAGE}/{version}/json"))
        info = data["info"]
        description = info.get("description") or ""
        failures += not check(info.get("version") == version, "PyPI version", info.get("version", "?"))
        failures += not check("stream-finetune" not in description, "PyPI old links", "clean")
        failures += not check("atomzero-doctor" in description or version < "0.1.8", "PyPI doctor", "documented")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] PyPI: {exc}")

    # Hugging Face
    try:
        text = fetch(f"https://huggingface.co/{HF_REPO}/raw/main/README.md")
        failures += not check(version in text, "HF card version", version)
        failures += not check("stream-finetune" not in text, "HF old links", "clean")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] HF: {exc}")

    # ModelScope
    try:
        data = json.loads(fetch(
            f"https://modelscope.cn/api/v1/models/{MODELSCOPE_REPO}"))
        text = (data.get("Data") or {}).get("ReadMeContent") or ""
        failures += not check(version in text, "ModelScope card version", version)
        failures += not check("stream-finetune" not in text, "ModelScope old links", "clean")
    except Exception as exc:
        failures += 1
        print(f"[FAIL] ModelScope: {exc}")

    # GitHub release and source
    try:
        release = json.loads(fetch(
            f"https://api.github.com/repos/{GITHUB_REPO}/releases/tags/{expected}"))
        assets = [asset.get("name", "") for asset in release.get("assets", [])]
        failures += not check(expected in release.get("tag_name", ""), "GitHub release", expected)
        failures += not check(
            any(name.endswith(".whl") for name in assets)
            and any(name.endswith(".tar.gz") for name in assets),
            "GitHub assets", ", ".join(assets) or "none")
        readme = fetch(f"https://raw.githubusercontent.com/{GITHUB_REPO}/refs/heads/main/README.md")
        failures += not check(version in readme, "GitHub README version", version)
    except Exception as exc:
        failures += 1
        print(f"[FAIL] GitHub: {exc}")

    if args.devto_url:
        try:
            text = fetch(args.devto_url)
            failures += not check("Atom Zero" in text, "dev.to article", "reachable")
            failures += not check(version in text, "dev.to version", version)
        except Exception as exc:
            failures += 1
            print(f"[FAIL] dev.to: {exc}")

    print(f"summary: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
