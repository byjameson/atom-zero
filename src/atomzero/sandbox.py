"""Bounded local execution sandbox for the autonomous loop.

This machine has no Docker, WSL or VM backend installed, so the default
backend is a disposable Git worktree with a sanitized environment, a hard
timeout and no credentials. It is **not** an OS-level sandbox and does not
claim to be one. The backend report says this explicitly.

    from atomzero.sandbox import backend_info, run_tests
    backend_info()
    run_tests(worktree, timeout=900)
"""
from __future__ import annotations

import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from atomzero.scoreboard import parse_pytest_summary

__all__ = [
    "backend_info",
    "cleaned_env",
    "run_command",
    "run_tests",
    "sanitized_env",
]

_SENSITIVE_KEY_RE = re.compile(
    r"(TOKEN|SECRET|PASSWORD|PASSWD|API_KEY|APIKEY|OPENAI|ANTHROPIC|"
    r"HUGGING|HF_|PYPI|TWINE|AWS_|AZURE_|GITHUB_|GH_|COHERE|GEMINI)",
    re.IGNORECASE,
)
_OUTPUT_LIMIT = 20_000


def backend_info() -> dict:
    """Report what isolation is actually available on this machine."""
    docker = shutil.which("docker")
    wsl = shutil.which("wsl")
    return {
        "backend": "worktree",
        "os_vm": False,
        "docker_available": bool(docker),
        "wsl_available": bool(wsl),
        "isolation": "disposable worktree + sanitized env + timeout",
        "network_isolated": False,
        "guarantees": [
            "runs in the agent worktree only",
            "credentials and token-like environment variables are removed",
            "a hard timeout bounds the process",
            "no push, tag or publish is performed",
        ],
        "limits": [
            "not an OS-level VM; the process can still read host files",
            "network is not blocked at the OS level",
            "a child process can outlive a simple timeout on Windows",
        ],
    }


def cleaned_env(base=None) -> dict:
    """Return ``base`` without token-like keys and without proxy settings."""
    source = os.environ if base is None else base
    cleaned = {
        key: value for key, value in source.items()
        if not _SENSITIVE_KEY_RE.search(str(key))
    }
    for key in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY",
                "http_proxy", "https_proxy", "all_proxy"):
        cleaned.pop(key, None)
    return cleaned


def sanitized_env(worktree=None, base=None) -> dict:
    """Return a scrubbed environment rooted in a temporary home directory."""
    env = cleaned_env(base)
    home = tempfile.mkdtemp(prefix="atomzero-sandbox-home-")
    env.update({
        "HOME": home,
        "USERPROFILE": home,
        "TMP": home,
        "TEMP": home,
        "TMPDIR": home,
        "PYTHONDONTWRITEBYTECODE": "1",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
    })
    if worktree is not None:
        env["PYTHONPATH"] = str(Path(worktree).resolve() / "src")
    return env


def _truncate(value: str) -> str:
    text = str(value or "")
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[-_OUTPUT_LIMIT:]


def run_command(command, *, cwd, timeout: int = 900, env=None) -> dict:
    """Run one bounded command and return a content-light result."""
    if isinstance(command, str):
        command = shlex.split(command, posix=False)
    if not isinstance(command, (list, tuple)) or not command:
        raise ValueError("command must be a non-empty list or string")
    try:
        timeout = max(1, min(int(timeout), 3600))
    except (TypeError, ValueError) as exc:
        raise ValueError("timeout must be an integer") from exc
    workdir = Path(cwd).expanduser().resolve()
    if not workdir.is_dir():
        raise ValueError(f"cwd is not a directory: {workdir}")
    if env is None:
        env = sanitized_env(workdir)
    started = time.monotonic()
    try:
        completed = subprocess.run(
            [str(part) for part in command], cwd=str(workdir), env=env,
            capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=timeout)
        elapsed = time.monotonic() - started
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "timed_out": False,
            "elapsed_seconds": round(elapsed, 3),
            "stdout": _truncate(completed.stdout),
            "stderr": _truncate(completed.stderr),
            "command": [str(part) for part in command],
            "cwd": str(workdir),
        }
    except subprocess.TimeoutExpired as exc:
        elapsed = time.monotonic() - started
        stdout = exc.stdout.decode("utf-8", "replace") if isinstance(
            exc.stdout, bytes) else (exc.stdout or "")
        stderr = exc.stderr.decode("utf-8", "replace") if isinstance(
            exc.stderr, bytes) else (exc.stderr or "")
        return {
            "ok": False,
            "returncode": None,
            "timed_out": True,
            "elapsed_seconds": round(elapsed, 3),
            "stdout": _truncate(stdout),
            "stderr": _truncate(stderr),
            "command": [str(part) for part in command],
            "cwd": str(workdir),
        }


def run_tests(worktree, *, command=None, timeout: int = 900,
              env=None) -> dict:
    """Run the project test command inside the sandbox and parse totals."""
    workdir = Path(worktree).expanduser().resolve()
    if command is None:
        command = [sys.executable, "-m", "pytest", "tests", "-q"]
    result = run_command(
        command, cwd=workdir, timeout=timeout,
        env=env if env is not None else sanitized_env(workdir))
    counts = {"tests_passed": 0, "tests_failed": 0}
    try:
        counts = parse_pytest_summary(result["stdout"])
    except ValueError:
        counts = {"tests_passed": 0, "tests_failed": 0}
    result.update(counts)
    result.update({
        "backend": "worktree",
        "isolation": "disposable worktree + sanitized env + timeout",
        "os_vm": False,
        "network_isolated": False,
    })
    result["stdout_tail"] = str(result.pop("stdout", ""))[-1000:]
    result["stderr_tail"] = str(result.pop("stderr", ""))[-1000:]
    return result
