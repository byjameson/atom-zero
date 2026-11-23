import json
import sys

import pytest

from atomzero.sandbox import (
    backend_info,
    cleaned_env,
    run_command,
    run_tests,
    sanitized_env,
)


def test_backend_is_explicitly_not_an_os_vm():
    info = backend_info()
    assert info["backend"] == "worktree"
    assert info["os_vm"] is False
    assert info["network_isolated"] is False
    assert "disposable worktree" in info["isolation"]


def test_cleaned_env_removes_tokens_and_proxies():
    env = cleaned_env({
        "PATH": "p",
        "OPENAI_API_KEY": "secret",
        "PYPI_TOKEN": "secret",
        "HF_TOKEN": "secret",
        "HTTPS_PROXY": "http://proxy",
        "NORMAL": "ok",
    })
    assert env["PATH"] == "p"
    assert env["NORMAL"] == "ok"
    assert "OPENAI_API_KEY" not in env
    assert "PYPI_TOKEN" not in env
    assert "HF_TOKEN" not in env
    assert "HTTPS_PROXY" not in env


def test_sanitized_env_uses_temp_home_and_worktree_pythonpath(tmp_path):
    env = sanitized_env(tmp_path, base={"OPENAI_API_KEY": "secret",
                                        "PATH": "p"})
    assert env["PYTHONPATH"] == str(tmp_path.resolve() / "src")
    assert env["HOME"] == env["USERPROFILE"] == env["TMP"]
    assert "OPENAI_API_KEY" not in env


def test_run_command_success_and_failure(tmp_path):
    ok = run_command([sys.executable, "-c", "print('hello')"],
                     cwd=tmp_path, timeout=30)
    assert ok["ok"] is True
    assert ok["returncode"] == 0
    assert "hello" in ok["stdout"]
    bad = run_command([sys.executable, "-c", "raise SystemExit(3)"],
                      cwd=tmp_path, timeout=30)
    assert bad["ok"] is False
    assert bad["returncode"] == 3


def test_run_tests_parses_pytest_summary(tmp_path):
    result = run_tests(
        tmp_path, command=[sys.executable, "-c",
                           "print('2 passed, 1 warning in 0.01s')"],
        timeout=30)
    assert result["ok"] is True
    assert result["tests_passed"] == 2
    assert result["tests_failed"] == 0
    assert result["backend"] == "worktree"
    assert result["os_vm"] is False
    assert "stdout_tail" in result


def test_run_tests_failed_summary(tmp_path):
    result = run_tests(
        tmp_path, command=[sys.executable, "-c",
                           "print('1 failed, 2 passed in 0.01s'); "
                           "raise SystemExit(1)"],
        timeout=30)
    assert result["ok"] is False
    assert result["tests_passed"] == 2
    assert result["tests_failed"] == 1


def test_run_tests_timeout(tmp_path):
    result = run_tests(
        tmp_path, command=[sys.executable, "-c",
                           "import time; time.sleep(5)"],
        timeout=1)
    assert result["ok"] is False
    assert result["timed_out"] is True
    assert result["tests_passed"] == 0


def test_invalid_command_and_cwd(tmp_path):
    with pytest.raises(ValueError):
        run_command([], cwd=tmp_path)
    with pytest.raises(ValueError):
        run_command([sys.executable, "-c", "pass"],
                    cwd=tmp_path / "missing")
