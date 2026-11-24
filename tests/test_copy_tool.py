from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from atomzero.copy_tool import (
    CopyTool,
    copy_pairs,
    extract_pairs,
    parse_rendered,
    render_pairs,
)
from research import benchmark_copy_tool as benchmark

ROOT = Path(__file__).resolve().parents[1]


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(ROOT), str(ROOT / "src")])
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _run_module(
    module: str, *args: str, input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env=_env(),
        input=input_text,
        check=False,
    )


def _prompt(pairs: list[tuple[str, str]]) -> str:
    body = "; ".join(f"{key}={value}" for key, value in pairs)
    return f"Copy these pairs exactly: [{body}]. Return only the result."


def _row(fmt: str, pairs: list[tuple[str, str]]) -> dict:
    return {
        "user": _prompt(pairs),
        "assistant": copy_pairs(_prompt(pairs), fmt=fmt),
        "format": fmt,
        "pairs": [[key, value] for key, value in pairs],
        "keys": [key for key, _ in pairs],
        "values": [value for _, value in pairs],
    }


@pytest.mark.parametrize("pair_count", [2, 3, 4])
@pytest.mark.parametrize("fmt", ["json", "kv", "yaml"])
def test_pair_counts_round_trip(pair_count: int, fmt: str):
    pairs = []
    for index in range(pair_count):
        key = f"key{index}"
        value = f"multi word value {index}" if index % 2 else f"value{index}"
        pairs.append((key, value))

    output = copy_pairs(_prompt(pairs), fmt=fmt)
    parsed = parse_rendered(output, fmt=fmt)

    assert list(parsed.items()) == pairs
    if fmt == "kv":
        assert "\n" not in output
    if fmt == "yaml":
        assert len(output.splitlines()) == pair_count
    if fmt == "json":
        assert ": " not in output and ", " not in output


def test_quoted_values_with_separators_round_trip():
    text = (
        '[first="hello world"; second=\'a; b: c\'; '
        'third="say \\"hi\\""]')
    expected = [
        ("first", "hello world"),
        ("second", "a; b: c"),
        ("third", 'say "hi"'),
    ]
    assert extract_pairs(text) == expected

    for fmt in ("json", "kv", "yaml"):
        output = copy_pairs(text, fmt=fmt)
        assert list(parse_rendered(output, fmt=fmt).items()) == expected


def test_colon_and_newline_syntax():
    assert extract_pairs("a: 1\nb: two words") == [
        ("a", "1"), ("b", "two words")]
    assert extract_pairs("a=1\nb=2\n") == [("a", "1"), ("b", "2")]


def test_multiple_bracket_blocks_combine():
    assert extract_pairs("Use [a=1; b=2] and [c=3]") == [
        ("a", "1"), ("b", "2"), ("c", "3")]


def test_copy_tool_wrapper():
    tool = CopyTool("yml")
    text = "[a=1; b=hello world]"
    assert tool.extract(text) == [("a", "1"), ("b", "hello world")]
    assert tool(text) == 'a: "1"\nb: hello world'
    assert tool.render([("x", "1")]) == 'x: "1"'


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("a=1;broken", "malformed pair"),
        ("=1", "missing key"),
        ("a=", "missing value"),
        ("a=1;a=2", "duplicate key"),
        ("[a=1; b=2] [a=3]", "duplicate key"),
        ("hello", "malformed pair"),
        ("[a=1", "unclosed"),
        ("[nothing here]", "inside brackets"),
        ("[=1]", "missing key"),
        ('a="unterminated', "unterminated"),
        ("", "empty"),
    ],
)
def test_extract_pairs_rejects_bad_input(text: str, message: str):
    with pytest.raises(ValueError, match=message):
        extract_pairs(text)


def test_render_and_format_errors():
    with pytest.raises(ValueError, match="unknown format"):
        copy_pairs("a=1", fmt="xml")
    with pytest.raises(ValueError, match="no pairs"):
        render_pairs([])
    with pytest.raises(ValueError, match="duplicate key"):
        render_pairs([("a", "1"), ("a", "2")])
    with pytest.raises(ValueError, match="missing value"):
        render_pairs([("a", "")])
    with pytest.raises(ValueError, match="duplicate JSON key"):
        parse_rendered('{"a":1,"a":2}', fmt="json")


def test_cli_main_json_and_kv(capsys):
    from atomzero.copy_tool import main

    assert main(
        ["--text", "a=1;b=hello world", "--format", "json"]) == 0
    captured = capsys.readouterr()
    assert captured.out.strip() == '{"a":"1","b":"hello world"}'
    assert captured.err == ""

    assert main(["--text", "a=1;b=2", "--format", "kv"]) == 0
    assert capsys.readouterr().out.strip() == "a=1;b=2"


def test_cli_module_subprocess_success_and_error():
    ok = _run_module(
        "atomzero.copy_tool",
        "--text", "a=1;b=hello world", "--format", "yaml")
    assert ok.returncode == 0
    assert ok.stdout.strip() == 'a: "1"\nb: hello world'

    bad = _run_module(
        "atomzero.copy_tool", "--text", "a=1;broken", "--format", "json")
    assert bad.returncode == 2
    assert bad.stdout == ""
    assert "error:" in bad.stderr

    unknown = _run_module(
        "atomzero.copy_tool", "--text", "a=1", "--format", "xml")
    assert unknown.returncode == 2


def test_cli_stdin_and_text_file(tmp_path):
    piped = _run_module(
        "atomzero.copy_tool", "--format", "kv", input_text="a=1;b=2")
    assert piped.returncode == 0
    assert piped.stdout.strip() == "a=1;b=2"

    path = tmp_path / "pairs.txt"
    path.write_text("[a=1; b=2]", encoding="utf-8")
    from_file = _run_module(
        "atomzero.copy_tool", "--text-file", str(path),
        "--format", "json")
    assert from_file.returncode == 0
    assert json.loads(from_file.stdout) == {"a": "1", "b": "2"}


def test_pyproject_declares_entry_point():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'atomzero-copy = "atomzero.copy_tool:main"' in pyproject


def _write_jsonl(path: Path, rows: list[dict | None]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8")


def test_benchmark_exact_rate_on_fixture(tmp_path):
    rows = [
        _row("json", [
            ("zqxkey1", "zqxvalueone"),
            ("zqxkey2", "zqx value two"),
        ]),
        _row("kv", [
            ("zqxkey3", "zqxvalue3"),
            ("zqxkey4", "zqx value four"),
            ("zqxkey5", "zqxvalue5"),
        ]),
        _row("yaml", [
            ("zqxkey6", "zqxvalue6"),
            ("zqxkey7", "zqx value seven"),
            ("zqxkey8", "zqxvalue8"),
            ("zqxkey9", "zqx value nine"),
        ]),
    ]
    path = tmp_path / "eval.jsonl"
    _write_jsonl(path, rows)

    summary = benchmark.run_benchmark(path)

    assert summary["rows_total"] == 3
    assert summary["rows_valid"] == 3
    assert summary["rows_invalid"] == 0
    assert summary["rows_exact"] == 3
    assert summary["exact_rate"] == 1.0
    assert summary["key_copy_rate"] == 1.0
    assert summary["status"] == "exact"
    assert summary["model_used"] is False
    assert summary["weights_changed"] is False

    body = json.dumps(summary)
    assert "zqxkey" not in body
    assert "zqx" not in body


def test_benchmark_counts_invalid_rows_and_keeps_valid_rate():
    good = _row("json", [("alpha", "one"), ("beta", "two words")])
    malformed_prompt = {
        "user": "there are no pairs here",
        "format": "json",
        "pairs": [["alpha", "one"]],
    }
    unsupported_format = {
        "user": "[alpha=one]",
        "format": "xml",
        "pairs": [["alpha", "one"]],
    }
    prompt_mismatch = {
        "user": "[alpha=one]",
        "format": "json",
        "pairs": [["alpha", "different"]],
    }
    summary = benchmark.summarize_rows(
        [good, malformed_prompt, unsupported_format, prompt_mismatch])

    assert summary["rows_total"] == 4
    assert summary["rows_valid"] == 1
    assert summary["rows_invalid"] == 3
    assert summary["exact_rate"] == 1.0
    assert summary["key_copy_rate"] == 1.0
    assert summary["invalid_reasons"] == {
        "malformed_prompt": 1,
        "prompt_metadata_mismatch": 1,
        "unsupported_format": 1,
    }


def test_benchmark_cli_outputs_content_free_json(tmp_path):
    pairs = [("qkey", "qvalue"), ("qkey2", "q value two")]
    path = tmp_path / "eval.jsonl"
    _write_jsonl(path, [_row("json", pairs)])

    result = _run_module(
        "research.benchmark_copy_tool", "--dataset", str(path))

    assert result.returncode == 0
    summary = json.loads(result.stdout)
    assert summary["exact_rate"] == 1.0
    assert "qkey" not in result.stdout
    assert "qvalue" not in result.stdout


def test_benchmark_missing_dataset_exit_code(tmp_path):
    missing = tmp_path / "missing.jsonl"
    result = _run_module(
        "research.benchmark_copy_tool", "--dataset", str(missing))
    assert result.returncode == 2
    assert "error:" in result.stderr
