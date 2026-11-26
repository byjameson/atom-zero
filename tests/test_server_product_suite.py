from __future__ import annotations

import json

import pytest

import research.server_product_suite as suite


PROMPT_SENTINEL = "SERVER_PROMPT_SENTINEL_ALPHA"
ANSWER_SENTINEL = "SERVER_ANSWER_SENTINEL_BETA"
SOURCE_SENTINEL = "launch_policy.txt"


class FakeProcess:
    def __init__(self, exit_code=None):
        self.returncode = exit_code
        self.terminated = False
        self.killed = False
        self.waits = 0

    def poll(self):
        return self.returncode

    def terminate(self):
        self.terminated = True
        self.returncode = -15

    def wait(self, timeout=None):
        self.waits += 1
        return self.returncode

    def kill(self):
        self.killed = True
        self.returncode = -9


class RecordingStopper:
    def __init__(self):
        self.calls = []

    def __call__(self, process):
        self.calls.append(process)
        process.terminate()
        process.wait(timeout=1)


def _doc_response(text, source=SOURCE_SENTINEL):
    return {
        "text": text,
        "sources": [{"source": source, "title": "Demo"}],
        "confidence": {"label": "high", "score": 0.9},
    }


def _chat_response(text, sources=None):
    return {
        "choices": [{
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
        "sources": list(sources or []),
    }


def _passing_http(base_url, path, payload, timeout):
    if path == "/v1/documents/answer":
        query = payload["query"]
        for case in suite.RAG_CASES:
            if query == case["prompt"]:
                return _doc_response(
                    f"The demo note says {case['expect']}.", case["source"])
        raise AssertionError(f"unexpected RAG query: {query}")
    content = payload["messages"][-1]["content"]
    for case in suite.COPY_CASES:
        if content == case["prompt"]:
            return _chat_response(case["expect"])
    return _chat_response(
        f"{ANSWER_SENTINEL} This is a useful answer with enough words.")


BENIGN_CASES = [
    {
        "prompt": f"{PROMPT_SENTINEL} ordinary question {index}",
        "category": "general",
        "forbidden_refusal_markers": [],
    }
    for index in range(suite.BENIGN_LIMIT)
]


def _run(tmp_path, http=_passing_http, *, keep_server=False, waiter=None,
         stopper=None, benign=None, docs_path="default"):
    process = FakeProcess()
    started = {}

    def starter(**kwargs):
        started.update(kwargs)
        return process

    def wait(*args, **kwargs):
        return None

    stop = stopper or RecordingStopper()
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    kwargs = {
        "port": 8031,
        "timeout": 5,
        "keep_server": keep_server,
        "http": http,
        "starter": starter,
        "waiter": waiter or wait,
        "stopper": stop,
        "benign_loader": lambda: (BENIGN_CASES if benign is None else benign),
    }
    if docs_path != "default":
        kwargs["docs_path"] = docs_path
    report = suite.run_server_suite(model, **kwargs)
    return report, process, stop, started


def test_parse_response_handles_chat_content_parts_and_sources():
    payload = {
        "choices": [{
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hello "},
                    {"type": "text", "text": "world"},
                ],
                "tool_calls": [{"name": "copy_tool"}],
            },
        }],
        "sources": {"first": {"source": "docs/launch_policy.txt"}},
    }
    parsed = suite.parse_response(payload)
    assert parsed["text"] == "hello world"
    assert parsed["sources"] == [{"source": "docs/launch_policy.txt"}]
    assert parsed["tool_calls"] == [{"name": "copy_tool"}]


def test_parse_response_handles_document_answer_and_malformed_input():
    parsed = suite.parse_response({
        "text": "The launch code is L-7781.",
        "sources": [{"source": SOURCE_SENTINEL}],
    })
    assert parsed["text"] == "The launch code is L-7781."
    assert parsed["sources"] == [{"source": SOURCE_SENTINEL}]
    assert suite.parse_response(None)["text"] == ""
    assert suite.parse_response("not a mapping")["sources"] == []


def test_source_citation_detection_accepts_paths_and_text_markers():
    assert suite.has_source_citation(
        _doc_response("L-7781", "notes/launch_policy.txt"),
        "launch_policy.txt")
    assert suite.has_source_citation(
        {"text": "L-7781 (source: launch_policy.txt)", "sources": []},
        "launch_policy.txt")
    assert suite.has_source_citation(
        {"text": "L-7781", "sources": ["launch_policy.txt"]})
    assert not suite.has_source_citation(
        _doc_response("L-7781", "warranty.txt"), "launch_policy.txt")
    assert not suite.has_source_citation(
        {"text": "L-7781", "sources": []}, "launch_policy.txt")


def test_rag_classification_needs_substring_and_citation():
    good = suite.classify_rag(
        _doc_response("The launch code is L-7781."),
        expected="L-7781", source=SOURCE_SENTINEL)
    assert good["passed"] is True
    assert good["match"] is True
    assert good["source_citation"] is True

    wrong_source = suite.classify_rag(
        _doc_response("The launch code is L-7781.", "warranty.txt"),
        expected="L-7781", source=SOURCE_SENTINEL)
    assert wrong_source["passed"] is False
    assert wrong_source["source_citation"] is False

    no_match = suite.classify_rag(
        _doc_response("The launch code is W-9900."),
        expected="L-7781", source=SOURCE_SENTINEL)
    assert no_match["passed"] is False
    assert no_match["match"] is False


def test_copy_classification_requires_exact_answer():
    expected = '{"project":"atlas"}'
    assert suite.classify_copy(
        _chat_response(expected), expected=expected)["passed"] is True
    assert suite.classify_copy(
        _chat_response(" " + expected + "\n"), expected=expected)["passed"]
    assert suite.classify_copy(
        _chat_response('{"project": "atlas"}'),
        expected=expected)["passed"] is False


def test_useful_classification_matches_product_suite_rules():
    assert suite.classify_useful(
        _chat_response("This is a normal helpful answer."))["passed"] is True
    assert suite.classify_useful(
        _chat_response("I'm sorry, but I can't help with that."))[
            "refusal"] is True
    assert suite.classify_useful(
        _chat_response("Too short."))["passed"] is False
    assert suite.classify_useful(
        _chat_response(""))["passed"] is False
    assert suite.classify_useful(
        _chat_response("A plain answer with enough words."),
        ["plain answer"])["refusal"] is True


def test_authored_cases_have_the_required_shape():
    assert len(suite.RAG_CASES) == 5
    assert all(case["expect"] and case["source"] for case in suite.RAG_CASES)
    assert len(suite.COPY_CASES) == 3
    assert {case["format"] for case in suite.COPY_CASES} == {"json", "kv"}
    assert len(suite.MULTI_TURN_CASES) == 3
    assert all(len(case["turns"]) == 2 for case in suite.MULTI_TURN_CASES)
    assert len(suite.LOCAL_BENIGN_PROMPTS) == suite.BENIGN_LIMIT


def test_benign_loader_uses_builder_and_falls_back_locally():
    class FakeBuilder:
        @staticmethod
        def build_rows(seed):
            assert seed == suite.BENIGN_SEED
            return [
                {"user": f"builder prompt {index}",
                 "category": "general"}
                for index in range(20)
            ]

    built = suite.load_benign_cases(module=FakeBuilder)
    assert len(built) == suite.BENIGN_LIMIT
    assert all(row["prompt"].startswith("builder prompt") for row in built)
    assert len({row["prompt"] for row in built}) == suite.BENIGN_LIMIT

    fallback = suite.load_benign_cases(module=object())
    assert len(fallback) == suite.BENIGN_LIMIT
    assert all(row["prompt"] for row in fallback)


def _area(examples, passed):
    return suite._area(examples, passed, 0, 0, 0)


def test_server_ok_boundaries_are_inclusive():
    assert suite.passes_server_bar(0.8, 0.8, 0.66, 0.8) is True
    assert suite.passes_server_bar(0.8, 0.8, 2 / 3, 0.8) is True
    assert suite.passes_server_bar(0.79, 0.8, 0.66, 0.8) is False
    assert suite.passes_server_bar(0.8, 0.79, 0.66, 0.8) is False
    assert suite.passes_server_bar(0.8, 0.8, 0.65, 0.8) is False
    assert suite.passes_server_bar(0.8, 0.8, 0.66, 0.79) is False
    assert suite.passes_server_bar(None, 1.0, 1.0, 1.0) is False


def test_build_report_scores_each_area_and_threshold():
    areas = {
        "rag": _area(5, 4),
        "copy": _area(3, 3),
        "multi_turn": _area(3, 2),
        "benign": _area(10, 8),
    }
    report = suite.build_report(
        model_dir="checkpoints/demo", host="127.0.0.1", port=8031,
        started=True, kept=False, areas=areas)
    assert report["server_ok"] is True
    assert report["status"] == "pass"
    assert report["decision"] == "server_ok"
    assert report["areas"]["rag"]["pass_rate"] == 0.8
    assert report["areas"]["copy"]["pass_rate"] == 1.0
    assert report["areas"]["multi_turn"]["pass_rate"] == 0.666667
    assert report["areas"]["benign"]["pass_rate"] == 0.8
    assert report["totals"] == {
        "examples": 21, "passed": 17, "refusals": 0,
        "source_citations": 0, "errors": 0,
    }
    assert report["passed_checks"] == list(suite.CHECK_ORDER)
    assert report["failed_checks"] == []

    areas["copy"] = _area(3, 2)
    failed = suite.build_report(
        model_dir="checkpoints/demo", host="127.0.0.1", port=8031,
        started=True, kept=False, areas=areas)
    assert failed["server_ok"] is False
    assert failed["decision"] == "not_server_ok"
    assert failed["failed_checks"] == ["copy_ok"]
    assert failed["checks"]["copy_ok"]["status"] == "fail"


def test_passing_run_is_content_free_and_stops_server(tmp_path):
    report, process, stopper, _ = _run(tmp_path)
    assert report["server_ok"] is True
    assert report["status"] == "pass"
    assert report["areas"]["rag"]["examples"] == 5
    assert report["areas"]["rag"]["source_citations"] == 5
    assert report["areas"]["copy"]["examples"] == 3
    assert report["areas"]["multi_turn"]["examples"] == 3
    assert report["areas"]["benign"]["examples"] == 10
    assert report["totals"]["examples"] == 21
    assert report["totals"]["passed"] == 21
    assert report["totals"]["source_citations"] == 5
    assert process.terminated is True
    assert stopper.calls == [process]
    assert report["content_free"] is True
    assert report["prompts_included"] is False
    assert report["answers_included"] is False
    text = json.dumps(report, sort_keys=True)
    for sentinel in (
            PROMPT_SENTINEL, ANSWER_SENTINEL, SOURCE_SENTINEL,
            "What is the launch code?", "L-7781", "TR-42",
            '{"project":"atlas"'):
        assert sentinel not in text


def test_keep_server_skips_cleanup(tmp_path):
    report, process, stopper, _ = _run(
        tmp_path, keep_server=True, docs_path=tmp_path / "demo.db")
    assert report["server_ok"] is True
    assert report["server"]["kept"] is True
    assert process.terminated is False
    assert stopper.calls == []


def test_request_errors_still_stop_server_and_are_counted(tmp_path):
    def failing_http(base_url, path, payload, timeout):
        raise RuntimeError("request failed")

    report, process, stopper, _ = _run(tmp_path, http=failing_http)
    assert report["server_ok"] is False
    assert report["status"] == "fail"
    assert report["areas"]["rag"]["errors"] == 5
    assert report["areas"]["copy"]["errors"] == 3
    assert report["areas"]["multi_turn"]["errors"] == 6
    assert report["areas"]["benign"]["errors"] == 10
    assert process.terminated is True
    assert stopper.calls == [process]


def test_startup_failure_is_reported_and_process_is_stopped(tmp_path):
    def failing_waiter(*args, **kwargs):
        raise TimeoutError("never listened")

    report, process, stopper, _ = _run(
        tmp_path, waiter=failing_waiter,
        docs_path=tmp_path / "demo.db")
    assert report["server_ok"] is False
    assert report["decision"] == "suite_error"
    assert report["error_type"] == "TimeoutError"
    assert report["server"]["started"] is False
    assert process.terminated is True
    assert stopper.calls == [process]


def test_default_temp_docs_are_removed_after_the_run(
        tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    monkeypatch.setattr(
        suite.tempfile, "mkdtemp",
        lambda prefix="": str(run_dir))
    report, _, _, started = _run(tmp_path)
    assert report["server_ok"] is True
    assert started["docs_path"].endswith("demo.db")
    assert not (run_dir / "server.log").exists()
    assert not run_dir.exists()


def test_stop_process_handles_running_and_exited_processes():
    running = FakeProcess()
    assert suite.stop_process(running) is True
    assert running.terminated is True
    assert running.waits == 1

    exited = FakeProcess(exit_code=0)
    assert suite.stop_process(exited) is False
    assert exited.terminated is False


def test_wait_for_server_detects_an_exited_process():
    with pytest.raises(RuntimeError):
        suite.wait_for_server(
            "127.0.0.1", 9, FakeProcess(exit_code=3), 0.1,
            sleep=lambda seconds: None)


def test_wait_for_server_accepts_a_connection():
    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    result = suite.wait_for_server(
        "127.0.0.1", 9, None, 0.1,
        connector=lambda address, timeout: Connection())
    assert result is None


def test_start_quickstart_server_builds_the_required_command(
        tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    docs = tmp_path / "demo.db"
    log = tmp_path / "server.log"
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(suite.subprocess, "Popen", fake_popen)
    suite.start_quickstart_server(
        model_dir=model, port=8123, docs_path=docs, log_path=log)
    command = captured["command"]
    assert command[0].endswith("python.exe") or "python" in command[0]
    assert command[1:4] == ["-m", "atomzero.quickstart", "--no-browser"]
    assert "--model-dir" in command
    assert command[command.index("--model-dir") + 1] == str(model)
    assert command[command.index("--port") + 1] == "8123"
    assert command[command.index("--docs") + 1] == str(docs)
    assert "--cache" not in command
    assert str(suite.ROOT / "src") in captured["kwargs"]["env"]["PYTHONPATH"]
    assert docs.parent.exists()


def test_cli_prints_json_and_exit_status(monkeypatch, capsys):
    monkeypatch.setattr(
        suite, "run_server_suite",
        lambda *args, **kwargs: {"server_ok": True, "status": "pass"})
    assert suite.main(["--model-dir", "checkpoints/demo", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["server_ok"] is True

    monkeypatch.setattr(
        suite, "run_server_suite",
        lambda *args, **kwargs: {"server_ok": False, "status": "fail"})
    assert suite.main(["--model-dir", "checkpoints/demo", "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["server_ok"] is False
