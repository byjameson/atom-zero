import json
import threading
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from atomzero import chat_server
from atomzero.doc_memory import DocMemory
from atomzero.feedback import FeedbackStore
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.research_queue import ResearchQueue
from atomzero.safety import SafetyFilter


def _post(url, payload):
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_raw(url, body):
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={"Content-Type": "text/plain"}, method="POST")
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def test_auto_reranker_falls_back_when_model_missing(tmp_path, capsys):
    docs = DocMemory(tmp_path / "docs.db")
    try:
        active = chat_server._attach_reranker(
            docs, "auto", str(tmp_path / "missing"))
        assert active == "hashed"
        assert docs.reranker is None
        assert "note: neural reranker disabled" in capsys.readouterr().out
    finally:
        docs.close()


def test_demo_page_uses_daily_agent_endpoint():
    page = chat_server.PAGE.lower()
    assert "atom zero" in page
    assert "/v1/agent/chat" in page
    assert "/v1/documents" in page
    assert "/v1/research/queue" in page
    assert "/v1/teacher/notes" in page
    assert "/v1/teacher/notes/verify" in page
    assert "/v1/learning/feedback" in page
    assert "/v1/learning/gaps" in page
    assert "/v1/learning/adapters" in page
    assert "adapter versions" in page
    assert "/v1/documents/forget" in page
    assert "/v1/documents/evidence" in page
    assert "/v1/documents/answer" in page
    assert "/v1/documents/summary" in page
    assert "/v1/hungry/run" in page
    assert "hungry mode" in page
    assert "grounded answer" in page
    assert "summary" in page
    assert 'id="docsrc"' in page
    assert 'id="docprefix"' in page
    assert "docsumcollection" in page
    assert "summarize collection" in page
    assert "source_prefix" in page
    assert "all sources" in page
    assert "copy citation" in page
    assert "evidence" in page
    assert "/v1/memory/history" in page
    assert "/v1/memory/export" in page
    assert "/v1/memory/import" in page
    assert "session backup" in page
    assert "atomzero_session" in page
    assert "add a document" in page
    assert "indexed documents" in page
    assert "search indexed documents" in page
    assert "new session" in page
    assert "add a teacher answer" in page
    assert "learning queue" in page
    assert "knowledge gaps" in page
    assert "export approved" in page
    assert "unverified" in page
    assert "reviewer" in page
    assert "verify" in page
    assert "research queue" in page
    assert "run approved" in page
    assert "teach a fact" in page
    assert "short story" not in page
    assert "/generate" not in page


def test_doc_answer_returns_sources(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("law.txt", "The capital of France is Paris.")
    answer = chat_server.doc_answer("capital of France", docs)
    assert answer is not None
    assert "Paris" in answer["text"]
    assert answer["sources"][0]["source"] == "law.txt"
    assert chat_server.doc_answer("nothing here", docs) is None
    docs.close()


def test_teacher_note_validation(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    with pytest.raises(ValueError):
        chat_server.add_teacher_note(docs, "chatgpt", "", "topic", "")
    with pytest.raises(ValueError):
        chat_server.add_teacher_note(
            docs, "chatgpt", "", "topic", "answer", "javascript:alert(1)")
    with pytest.raises(ValueError):
        chat_server.add_teacher_note(
            docs, "chatgpt", "", "x" * 301, "answer")
    docs.close()


def test_doc_answer_dedupes_repeated_sources(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text(
        "ankara.txt",
        ("Ankara is a city. " * 30) + "\n\n" + ("Ankara is in Turkey. " * 30),
        title="Ankara")
    answer = chat_server.doc_answer("Ankara", docs)
    assert answer is not None
    assert len(answer["sources"]) == 1
    assert answer["sources"][0]["source"] == "ankara.txt"
    docs.close()


def test_ingest_document_validates_source_and_text(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    with pytest.raises(ValueError):
        chat_server.ingest_document(docs, "", "some text")
    with pytest.raises(ValueError):
        chat_server.ingest_document(docs, "bad\nsource", "some text")
    with pytest.raises(ValueError):
        chat_server.ingest_document(docs, "note.txt", "   ")
    docs.close()


def test_http_document_endpoints_and_grounded_chat(tmp_path, monkeypatch):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("law.txt", "The capital of France is Paris.",
                  title="Notes")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "user_id": 0, "assistant_id": 1,
        "queue_path": str(tmp_path / "queue.jsonl"),
        "allow_url_fetch": True, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    import atomzero.web_research as wr
    monkeypatch.setattr(wr, "index_url", lambda docs, url: {
        "source": url, "title": "T", "chunks": 1, "preview": "p"})
    monkeypatch.setattr(wr, "wikipedia_research", lambda docs, q, lang, limit: [
        {"source": "https://wiki/x", "title": "X", "chunks": 1,
         "preview": "p"}])
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        with urllib.request.urlopen(base + "/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
        assert health["documents"]["sources"] == 1
        with urllib.request.urlopen(base + "/v1/documents", timeout=10) as r:
            sources = json.loads(r.read().decode("utf-8"))
        assert sources["data"][0]["source"] == "law.txt"
        search = _post(base + "/v1/documents/search",
                       {"query": "capital France"})
        assert "Paris" in search["data"][0]["chunk"]
        summary = json.loads(urllib.request.urlopen(
            base + "/v1/documents/summary?source=law.txt&sentences=2",
            timeout=10).read().decode("utf-8"))
        assert "Paris" in summary["text"]
        assert summary["method"] == "extractive"
        collection = json.loads(urllib.request.urlopen(
            base + "/v1/documents/summary?kind=document&sentences=2",
            timeout=10).read().decode("utf-8"))
        assert "Paris" in collection["text"]
        assert collection["sources"] >= 1
        chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the capital of France?"}],
            "max_tokens": 20,
        })
        assert "Paris" in chat["choices"][0]["message"]["content"]
        assert chat["sources"][0]["source"] == "law.txt"
        taught = _post(base + "/v1/teach",
                       {"fact": "The sky is blue."})
        assert taught["fact_chunks"] == 1
        assert "blue" in _post(
            base + "/v1/documents/search",
            {"query": "sky"})["data"][0]["chunk"]
        queued = _post(base + "/v1/teach",
                       {"user": "good morning",
                        "assistant": "Good morning!"})
        assert queued["skill_queued"] is True
        queue = (tmp_path / "queue.jsonl").read_text(encoding="utf-8")
        assert "good morning" in queue
        ingested = _post(base + "/v1/documents", {
            "source": "warranty.txt", "title": "Warranty",
            "text": "The warranty code is W-7781. Keep the receipt.",
        })
        assert ingested["chunks"] == 1
        assert ingested["title"] == "Warranty"
        assert len(ingested["sha256"]) == 64
        assert ingested["content_sha256"] == ingested["sha256"]
        assert "W-7781" in _post(
            base + "/v1/documents/search",
            {"query": "warranty code"})["data"][0]["chunk"]
        warranty_chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the warranty code?"}],
            "max_tokens": 20,
        })
        assert "W-7781" in warranty_chat["choices"][0]["message"]["content"]
        assert warranty_chat["sources"][0]["source"] == "warranty.txt"
        replaced = _post(base + "/v1/documents", {
            "source": "warranty.txt",
            "text": "The warranty code is W-9900.",
        })
        assert replaced["chunks"] == 1
        assert _post(base + "/v1/documents/search",
                     {"query": "7781"})["data"] == []
        assert "W-9900" in _post(
            base + "/v1/documents/search",
            {"query": "warranty"})["data"][0]["chunk"]
        raw = _post_raw(
            base + "/v1/documents/upload?" + urllib.parse.urlencode({
                "source": "raw.txt", "title": "Raw upload"}),
            "The raw upload code is R-42.")
        assert raw["source"] == "raw.txt"
        assert raw["chunks"] == 1
        assert "R-42" in _post(
            base + "/v1/documents/search",
            {"query": "raw upload"})["data"][0]["chunk"]

        old_body = chat_server.MAX_BODY_BYTES
        chat_server.MAX_BODY_BYTES = 5
        try:
            raw_over_body = _post_raw(
                base + "/v1/documents/upload?" + urllib.parse.urlencode({
                    "source": "raw-over-body.txt"}),
                "1234567890")
            assert raw_over_body["source"] == "raw-over-body.txt"
            with pytest.raises(urllib.error.HTTPError) as json_over_body:
                _post(base + "/v1/documents", {
                    "source": "json-over-body.txt", "text": "1234567890"})
            assert json_over_body.value.code == 413
        finally:
            chat_server.MAX_BODY_BYTES = old_body

        old_doc = chat_server.MAX_DOC_BYTES
        chat_server.MAX_DOC_BYTES = 5
        try:
            with pytest.raises(urllib.error.HTTPError) as raw_over_doc:
                _post_raw(
                    base + "/v1/documents/upload?" + urllib.parse.urlencode({
                        "source": "raw-over-doc.txt"}),
                    "1234567890")
            assert raw_over_doc.value.code == 413
        finally:
            chat_server.MAX_DOC_BYTES = old_doc
        old_max = chat_server.MAX_DOC_BYTES
        chat_server.MAX_DOC_BYTES = 5
        try:
            with pytest.raises(urllib.error.HTTPError) as too_big:
                _post(base + "/v1/documents",
                      {"source": "big.txt", "text": "123456"})
            assert too_big.value.code == 413
        finally:
            chat_server.MAX_DOC_BYTES = old_max
        with pytest.raises(urllib.error.HTTPError) as missing_source:
            _post(base + "/v1/documents", {"text": "no source"})
        assert missing_source.value.code == 400
        teacher = _post(base + "/v1/teacher/notes", {
            "teacher": "chatgpt-web", "model": "gpt-5",
            "topic": "Teacher code",
            "answer": "The teacher code is T-9000.",
            "source_url": "https://example.com/chat",
        })
        assert teacher["kind"] == "teacher"
        assert teacher["verified"] is False
        assert teacher["source"].startswith("teacher://chatgpt-web/")
        listed_docs = json.loads(urllib.request.urlopen(
            base + "/v1/documents", timeout=10).read().decode("utf-8"))
        teacher_listed = next(
            item for item in listed_docs["data"]
            if item["source"].startswith("teacher://chatgpt-web/"))
        assert teacher_listed["kind"] == "teacher"
        assert teacher_listed["verified"] is False
        assert teacher_listed["source_url"] == "https://example.com/chat"
        teacher_hit = _post(
            base + "/v1/documents/search",
            {"query": "teacher code"})["data"][0]
        assert teacher_hit["kind"] == "teacher"
        assert teacher_hit["verified"] is False
        assert teacher_hit["source_url"] == "https://example.com/chat"
        teacher_chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the teacher code?"}],
            "max_tokens": 20,
        })
        teacher_text = teacher_chat["choices"][0]["message"]["content"]
        assert "T-9000" in teacher_text
        assert "unverified teacher-derived note" in teacher_text
        assert teacher_chat["sources"][0]["verified"] is False
        assert teacher_chat["sources"][0]["source_url"] == "https://example.com/chat"
        verified_note = _post(base + "/v1/teacher/notes/verify", {
            "source": teacher["source"], "reviewer": "reviewer-1"})
        assert verified_note["ok"] is True
        assert verified_note["verified"] is True
        assert verified_note["reviewer"] == "reviewer-1"
        assert verified_note["verified_at"]
        notes = json.loads(urllib.request.urlopen(
            base + "/v1/teacher/notes", timeout=10).read().decode("utf-8"))
        verified_listed = next(
            item for item in notes["data"]
            if item["source"] == teacher["source"])
        assert verified_listed["verified"] is True
        assert verified_listed["reviewer"] == "reviewer-1"
        assert notes["stats"]["verified"] == 1
        verified_chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the teacher code?"}],
            "max_tokens": 20,
        })
        verified_text = verified_chat["choices"][0]["message"]["content"]
        assert "teacher note, verified by reviewer-1" in verified_text
        assert "unverified teacher-derived note" not in verified_text
        assert verified_chat["sources"][0]["verified"] is True
        assert verified_chat["sources"][0]["reviewer"] == "reviewer-1"
        with pytest.raises(urllib.error.HTTPError) as missing_note:
            _post(base + "/v1/teacher/notes/verify",
                  {"source": "teacher://missing", "reviewer": "r"})
        assert missing_note.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as no_provider:
            _post(base + "/v1/teacher/ask", {"prompt": "hello"})
        assert no_provider.value.code == 400

        class FakeProvider:
            label = "fake"
            model = "fake-1"
            host = "fake"

            def ask(self, prompt, system="", max_tokens=512):
                return "The provider answer is P-777."

        chat_server._state["teacher"] = FakeProvider()
        provider_note = _post(base + "/v1/teacher/ask", {
            "prompt": "What is the provider answer?",
            "topic": "Provider code",
            "source_url": "https://example.com/provider",
        })
        assert provider_note["kind"] == "teacher"
        assert provider_note["verified"] is False
        assert provider_note["source"].startswith("teacher://fake/")
        assert "P-777" in _post(
            base + "/v1/documents/search",
            {"query": "provider answer"})["data"][0]["chunk"]
        provider_chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the provider answer?"}],
            "max_tokens": 20,
        })
        provider_text = provider_chat["choices"][0]["message"]["content"]
        assert "P-777" in provider_text
        assert "unverified teacher-derived note" in provider_text
        chat_server._state.pop("teacher", None)
        learned = _post(base + "/v1/learn_url",
                        {"url": "https://example.com/page"})
        assert learned["title"] == "T"
        research = _post(base + "/v1/research", {"query": "anything"})
        assert research["pages"][0]["title"] == "X"
        agent = _post(base + "/v1/agent/chat", {
            "session_id": "s1", "message": "add task buy milk"})
        assert "Added task 1" in agent["reply"]
        session_chat = _post(base + "/v1/chat/completions", {
            "session_id": "s1",
            "messages": [{"role": "user", "content": "list tasks"}],
            "max_tokens": 20,
        })
        content = session_chat["choices"][0]["message"]["content"]
        assert "buy milk" in content
        planned = _post(base + "/v1/agent/plan", {
            "session_id": "s1",
            "steps": [{"tool": "task_add",
                       "args": {"text": "plan task"}}],
        })
        assert planned["status"] == "confirmation_required"
        assert planned["results"] == []
        confirmed = _post(base + "/v1/agent/plan", {
            "session_id": "s1", "confirm": True,
            "steps": [{"tool": "task_add",
                       "args": {"text": "plan task"}}],
        })
        assert confirmed["status"] == "ok"
        listed = _post(base + "/v1/agent/plan", {
            "session_id": "s1",
            "steps": [{"tool": "task_list"}],
        })
        tasks = listed["results"][0]["result"]["tasks"]
        assert any(task["text"] == "plan task" for task in tasks)
        with pytest.raises(urllib.error.HTTPError) as bad_plan:
            _post(base + "/v1/agent/plan", {
                "session_id": "s1",
                "steps": [{"tool": "shell"}],
            })
        assert bad_plan.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as bad_confirm:
            _post(base + "/v1/agent/plan", {
                "session_id": "s1", "confirm": "true",
                "steps": [{"tool": "time"}],
            })
        assert bad_confirm.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_resolve_adapter_auto_uses_promoted_path(tmp_path):
    assert chat_server.resolve_adapter("") == ""
    assert chat_server.resolve_adapter("checkpoints/plain") == "checkpoints/plain"
    active = tmp_path / "active.json"
    active.write_text(
        '{"path": "checkpoints/adapters_feedback/v0001"}',
        encoding="utf-8")
    assert chat_server.resolve_adapter("auto", str(active)) == (
        "checkpoints/adapters_feedback/v0001")
    active.write_text("not json", encoding="utf-8")
    assert chat_server.resolve_adapter("auto", str(active)) == ""


def test_http_learning_feedback_review_and_export(tmp_path):
    store = FeedbackStore(tmp_path / "feedback.json")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "feedback": store,
        "feedback_path": str(tmp_path / "feedback.json"),
        "feedback_export_path": str(tmp_path / "training.jsonl"),
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        first = _post(base + "/v1/learning/feedback", {
            "user": "hello", "assistant": "Hi!", "rating": 1,
            "session_id": "s1"})
        assert first["status"] == "pending"
        second = _post(base + "/v1/learning/feedback", {
            "user": "what is 2+2?", "assistant": "Five.",
            "corrected": "Four.", "rating": -1, "session_id": "s1"})
        bad = _post(base + "/v1/learning/feedback", {
            "user": "bad", "assistant": "wrong", "rating": -1})
        listed = json.loads(urllib.request.urlopen(
            base + "/v1/learning/queue", timeout=10).read().decode("utf-8"))
        assert listed["stats"] == {"total": 3, "pending": 3}
        with pytest.raises(urllib.error.HTTPError) as no_correction:
            _post(base + "/v1/learning/queue/approve", {"id": bad["id"]})
        assert no_correction.value.code == 400
        for item in (first, second):
            approved = _post(base + "/v1/learning/queue/approve",
                             {"id": item["id"]})
            assert approved["status"] == "approved"
        rejected = _post(base + "/v1/learning/queue/reject",
                         {"id": bad["id"]})
        assert rejected["status"] == "rejected"
        exported = _post(base + "/v1/learning/queue/export", {})
        assert exported["examples"] == 2
        rows = [json.loads(line) for line in
                (tmp_path / "training.jsonl").read_text().splitlines()]
        assert rows == [
            {"user": "hello", "assistant": "Hi!"},
            {"user": "what is 2+2?", "assistant": "Four."},
        ]
        health = json.loads(urllib.request.urlopen(
            base + "/health", timeout=10).read().decode("utf-8"))
        assert health["learning"]["approved"] == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_knowledge_gaps_propose_and_dismiss(tmp_path):
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    research = ResearchQueue(tmp_path / "research.json")
    gap = gaps.add("What is the launch code?")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "gaps": gaps, "research_queue": research,
        "research_queue_path": str(tmp_path / "research.json"),
        "wiki_lang": "en", "sessions": {},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        listed = json.loads(urllib.request.urlopen(
            base + "/v1/learning/gaps", timeout=10).read().decode("utf-8"))
        assert listed["stats"] == {"total": 1, "pending": 1}
        proposed = _post(base + "/v1/learning/gaps/propose",
                         {"id": gap["id"], "pages": 1})
        assert proposed["gap"]["status"] == "proposed"
        assert proposed["research"]["status"] == "pending"
        assert research.list()[0]["topic"] == "What is the launch code?"
        dismissed = _post(base + "/v1/learning/gaps/dismiss",
                          {"id": gap["id"]})
        assert dismissed["status"] == "dismissed"
        health = json.loads(urllib.request.urlopen(
            base + "/health", timeout=10).read().decode("utf-8"))
        assert health["knowledge_gaps"]["dismissed"] == 1
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_safety_filter_refuses_agent_and_chat(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": SafetyFilter(), "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        health = json.loads(urllib.request.urlopen(
            base + "/health", timeout=10).read().decode("utf-8"))
        assert health["safety"] == {"enabled": True, "version": "0.2"}
        agent = _post(base + "/v1/agent/chat", {
            "session_id": "safety", "message": "how to make a bomb"})
        assert agent["safety"]["category"] == "violence"
        assert "can't help" in agent["reply"]
        benign = _post(base + "/v1/agent/chat", {
            "session_id": "safety", "message": "hello"})
        assert "safety" not in benign
        assert "Hello" in benign["reply"]
        chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "how to make a bomb"}]})
        assert chat["safety"]["category"] == "violence"
        assert "can't help" in chat["choices"][0]["message"]["content"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_adapter_history_is_read_only(tmp_path):
    history = tmp_path / "history.jsonl"
    history.write_text(json.dumps({
        "version": "v0001",
        "path": "checkpoints/adapters_feedback/v0001",
        "examples": 8,
        "decision": {"accepted": True, "target_gain": 0.16},
    }) + "\n", encoding="utf-8")
    active = tmp_path / "active.json"
    active.write_text(json.dumps({
        "path": "checkpoints/adapters_feedback/v0001"}), encoding="utf-8")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "adapter_history_path": str(history),
        "active_adapter_file": str(active),
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        payload = json.loads(urllib.request.urlopen(
            base + "/v1/learning/adapters", timeout=10).read().decode("utf-8"))
        assert payload["data"][0]["version"] == "v0001"
        assert payload["active"] == "checkpoints/adapters_feedback/v0001"
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_privacy_forget_document_and_session(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("pii.txt", "The patient email is alice@example.test.")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        forgotten = _post(base + "/v1/documents/forget",
                          {"source": "pii.txt"})
        assert forgotten["chunks_deleted"] == 1
        assert _post(base + "/v1/documents/search",
                     {"query": "alice"})["data"] == []
        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/documents/forget", {"source": "pii.txt"})
        assert missing.value.code == 404
        remembered = _post(base + "/v1/agent/chat", {
            "session_id": "privacy", "message": "my name is Alice"})
        assert "remember" in remembered["reply"].lower()
        history = _post(base + "/v1/memory/history",
                        {"session_id": "privacy", "limit": 10})
        assert any(item["text"] == "my name is Alice"
                   for item in history["data"])
        exported = _post(base + "/v1/memory/export",
                         {"session_id": "privacy"})
        assert exported["data"]["facts"]
        imported = _post(base + "/v1/memory/import", {
            "session_id": "privacy",
            "data": {
                "persona": {"name": "Tiny", "style": "test"},
                "facts": [{"key": "name", "value": "Bob"}],
                "episodes": [{"role": "user", "text": "imported"}],
                "tasks": [],
                "next_task": 1,
            },
        })
        assert imported["facts"] == 1
        recalled = _post(base + "/v1/agent/chat", {
            "session_id": "privacy", "message": "what is my name?"})
        assert "Bob" in recalled["reply"]
        removed = _post(base + "/v1/memory/forget", {
            "session_id": "privacy", "text": "Bob"})
        assert removed["facts_deleted"] == 1
        recalled = _post(base + "/v1/agent/chat", {
            "session_id": "privacy", "message": "what is my name?"})
        assert "do not know your name" in recalled["reply"]
        cleared = _post(base + "/v1/memory/forget", {
            "session_id": "privacy", "all": True})
        assert cleared["cleared_all"] is True
    finally:
        httpd.shutdown()
        httpd.server_close()
        docs.close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_research_queue_requires_approval_and_cites(tmp_path, monkeypatch):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    queue = ResearchQueue(tmp_path / "research.json")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "user_id": 0, "assistant_id": 1,
        "queue_path": str(tmp_path / "queue.jsonl"),
        "allow_url_fetch": False, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "research_queue": queue,
        "research_queue_path": str(tmp_path / "research.json"),
    })
    calls = []

    def fake_research(memory, query, lang, limit):
        calls.append(query)
        source = "https://en.wikipedia.org/?curid=9"
        chunks = memory.add_text(
            source, "Ankara is the capital of Turkey.", title="Ankara")
        return [{"source": source, "title": "Ankara", "chunks": chunks,
                 "preview": "Ankara is the capital of Turkey.",
                 "context": "Ankara is the capital of Turkey. It is in Anatolia."}]

    import atomzero.web_research as wr
    monkeypatch.setattr(wr, "wikipedia_research", fake_research)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        item = _post(base + "/v1/research/queue",
                     {"topic": "Ankara", "pages": 2})
        assert item["status"] == "pending"
        listed = json.loads(urllib.request.urlopen(
            base + "/v1/research/queue", timeout=10).read().decode("utf-8"))
        assert listed["data"][0]["id"] == item["id"]
        unapproved = _post(base + "/v1/research/queue/run", {"limit": 1})
        assert unapproved["ran"] == []
        assert calls == []
        approved = _post(base + "/v1/research/queue/approve",
                         {"id": item["id"]})
        assert approved["status"] == "approved"

        class FakeTeacher:
            label = "fake"
            model = "fake-1"
            host = "fake"

            def ask(self, prompt, system="", max_tokens=512):
                assert "https://en.wikipedia.org/?curid=9" in prompt
                return "Teacher synthesis: Ankara is the capital of Turkey."

        chat_server._state["teacher"] = FakeTeacher()
        ran = _post(base + "/v1/research/queue/run", {"limit": 1})
        assert ran["ran"][0]["status"] == "done"
        assert calls == ["Ankara"]
        assert ran["ran"][0]["sources"][0]["source"].startswith("https://")
        assert ran["ran"][0]["teacher"]["verified"] is False
        assert ran["ran"][0]["teacher"]["source"].startswith("teacher://fake/")
        assert "Ankara" in _post(
            base + "/v1/documents/search",
            {"query": "capital Turkey"})["data"][0]["chunk"]
        chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the capital of Turkey?"}],
            "max_tokens": 20,
        })
        assert "Ankara" in chat["choices"][0]["message"]["content"]
        assert any(source["source"].startswith("https://")
                   for source in chat["sources"])
        teacher_chat = _post(base + "/v1/chat/completions", {
            "messages": [{"role": "user",
                          "content": "what is the teacher synthesis?"}],
            "max_tokens": 20,
        })
        assert "Teacher synthesis" in teacher_chat["choices"][0]["message"]["content"]
        assert "unverified teacher-derived note" in teacher_chat["choices"][0]["message"]["content"]
        health = json.loads(urllib.request.urlopen(
            base + "/health", timeout=10).read().decode("utf-8"))
        assert health["research_queue"]["done"] == 1
        queue_note = ran["ran"][0]["teacher"]["source"]
        _post(base + "/v1/teacher/notes/verify", {
            "source": queue_note, "reviewer": "queue-reviewer"})
        queue_view = json.loads(urllib.request.urlopen(
            base + "/v1/research/queue", timeout=10).read().decode("utf-8"))
        assert queue_view["data"][0]["teacher"]["verified"] is True
        assert queue_view["data"][0]["teacher"]["reviewer"] == "queue-reviewer"
        second = _post(base + "/v1/research/queue", {"topic": "Izmir"})
        rejected = _post(base + "/v1/research/queue/reject",
                         {"id": second["id"]})
        assert rejected["status"] == "rejected"
        ran_again = _post(base + "/v1/research/queue/run", {"limit": 1})
        assert ran_again["ran"] == []
        assert calls == ["Ankara"]
        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/research/queue/approve", {"id": "r9999"})
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_ui_collection_scope_contract():
    page = chat_server.PAGE
    assert 'id="docprefix"' in page
    assert 'id="docsumcollection"' in page
    assert "source_prefix: prefix" in page
    assert "diverse: true" in page
    assert "summarizeCollection" in page
    assert "collection summary" in page
    assert "/v1/graph" in page
    assert "/v1/graph/path" in page
    assert "Second brain graph" in page
    assert 'id="graphrefresh"' in page
    assert 'id="graphpathbtn"' in page
    assert "showEntity" in page
    assert "showGraphPath" in page
    assert "/v1/skills" in page
    assert "/v1/skills/run" in page
    assert "/v1/skills/chain" in page
    assert "/v1/skills/chains" in page
    assert "Custom skills" in page
    assert 'id="skrefresh"' in page
    assert 'id="skillrun"' in page
    assert 'id="skillchainrun"' in page
    assert 'id="chainreciperun"' in page
    assert "refreshSkills" in page
    assert "runSkill" in page
    assert "runSkillChain" in page
    assert "runRecipe" in page
    assert "/v1/briefing" in page
    assert "/v1/weekly-review" in page
    assert "/v1/goals/plan" in page
    assert "Daily brief" in page
    assert 'id="briefrefresh"' in page
    assert 'id="briefweekly"' in page
    assert "getWeeklyReview" in page
    assert "planGoal" in page
    assert 'data-q="plan goal prepare the warranty page"' in page
    assert 'data-q="brief me"' in page


def test_doc_answer_returns_evidence_confidence_and_source_scope(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.add_text("warranty.txt", "The warranty period is 24 months.",
                  title="Warranty")
    docs.add_text("travel.txt",
                  "Book business class. The travel portal code is TR-42.",
                  title="Travel")
    answer = chat_server.doc_answer("what is the launch code?", docs)
    assert answer is not None
    assert "L-7781" in answer["text"]
    assert answer["sources"][0]["source"] == "launch.txt"
    assert answer["evidence"][0]["source"] == "launch.txt"
    assert [item["source"] for item in answer["evidence"]] == ["launch.txt"]
    assert "L-7781" in answer["evidence"][0]["snippet"]
    assert 0.0 <= answer["confidence"]["score"] <= 1.0
    assert answer["confidence"]["label"] in {"low", "medium", "high"}
    scoped = chat_server.doc_answer(
        "warranty period", docs, source="warranty.txt")
    assert scoped is not None
    assert all(item["source"] == "warranty.txt"
               for item in scoped["evidence"])
    assert "24 months" in scoped["text"]
    by_prefix = chat_server.doc_answer(
        "launch code", docs, source_prefix="launch")
    assert by_prefix is not None
    assert "L-7781" in by_prefix["text"]
    docs.add_text("teacher://chatgpt/1",
                  "The warranty period is 36 months.",
                  title="Teacher", meta={"kind": "teacher"})
    by_kind = chat_server.doc_answer("warranty period", docs, kind="teacher")
    assert by_kind is not None
    assert "36 months" in by_kind["text"]
    assert all(item["kind"] == "teacher" for item in by_kind["evidence"])
    assert chat_server.doc_answer("population of Paris", docs) is None
    docs.close()


def test_http_document_answer_and_scoped_search(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("launch.txt", "The launch code is L-7781.",
                  title="Launch")
    docs.add_text("warranty.txt", "The warranty period is 24 months.",
                  title="Warranty")
    docs.add_text("notes/extra.txt", "The visitor badge code is VB-42.",
                  title="Visitor")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        answer = _post(base + "/v1/documents/answer",
                       {"query": "what is the launch code?"})
        assert "L-7781" in answer["text"]
        assert answer["confidence"]["label"] in {"low", "medium", "high"}
        assert answer["evidence"][0]["source"] == "launch.txt"
        scoped = _post(base + "/v1/documents/answer", {
            "query": "warranty period", "source": "warranty.txt"})
        assert "24 months" in scoped["text"]
        assert all(item["source"] == "warranty.txt"
                   for item in scoped["evidence"])
        filtered = _post(base + "/v1/documents/search", {
            "query": "launch code", "source": "warranty.txt"})
        assert filtered["data"] == []
        prefixed = _post(base + "/v1/documents/answer", {
            "query": "launch code", "source_prefix": "launch"})
        assert "L-7781" in prefixed["text"]
        collection = json.loads(urllib.request.urlopen(
            base + "/v1/documents/summary?source_prefix=notes/&sentences=2",
            timeout=10).read().decode("utf-8"))
        assert "VB-42" in collection["text"]
        assert collection["sources"] == 1
        assert collection["citations"][0]["source"] == "notes/extra.txt"
        missing = _post(base + "/v1/documents/answer",
                        {"query": "population of Paris"})
        assert missing["sources"] == []
        assert missing["confidence"]["label"] == "none"
        with pytest.raises(urllib.error.HTTPError) as empty:
            _post(base + "/v1/documents/answer", {"query": "  "})
        assert empty.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_hungry_mode_dry_run_and_learn(tmp_path):
    class FakeTeacher:
        label = "fake"
        model = "fake-1"

        def ask(self, prompt, system="", max_tokens=512):
            if prompt.startswith("Propose"):
                return "What is the capital of France?"
            return "Paris is the capital of France."

    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("launch.txt", "The launch code is L-7781.")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gap = gaps.add("What is the capital of France?")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "gaps": gaps, "teacher": FakeTeacher(),
        "hungry_ledger_path": str(tmp_path / "hungry.jsonl"),
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        dry = _post(base + "/v1/hungry/run", {
            "dry_run": True, "max_questions": 1})
        assert dry["dry_run"] is True
        assert dry["provider_calls"] == 0
        assert gaps.get(gap["id"])["status"] == "pending"
        learned = _post(base + "/v1/hungry/run", {
            "max_questions": 1, "max_provider_calls": 2})
        assert learned["learned"] == 1
        assert learned["weights_changed"] is False
        assert gaps.get(gap["id"])["status"] == "learned"
        hits = docs.search("capital of France")
        assert hits[0]["kind"] == "teacher"
        assert hits[0]["verified"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_skills_endpoint(tmp_path):
    from atomzero.skills import SkillRegistry

    skill = {
        "name": "demo-skill",
        "triggers": ["demo skill"],
        "prompt": "Demo: {context}",
        "provider": False,
    }
    provider_skill = {
        "name": "provider-skill",
        "triggers": ["provider skill"],
        "prompt": "Provider: {context}",
        "provider": True,
    }
    chain_skill = {
        "name": "chain-skill",
        "triggers": ["chain skill"],
        "prompt": "Chain previous: {previous}",
        "provider": False,
    }
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    (skills_dir / "demo.json").write_text(
        json.dumps(skill), encoding="utf-8")
    (skills_dir / "provider.json").write_text(
        json.dumps(provider_skill), encoding="utf-8")
    (skills_dir / "chain.json").write_text(
        json.dumps(chain_skill), encoding="utf-8")
    chain_dir = skills_dir / "chains"
    chain_dir.mkdir()
    (chain_dir / "demo-recipe.json").write_text(json.dumps({
        "name": "demo-recipe",
        "steps": ["demo-skill", "chain-skill"],
        "carry_context": True,
    }), encoding="utf-8")
    registry = SkillRegistry.from_dir(skills_dir)
    from atomzero.skills import ChainRecipeRegistry
    chains = ChainRecipeRegistry.from_skills_dir(skills_dir)
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.")
    calls = []

    def generator(prompt):
        calls.append(prompt)
        return "Provider answer."

    old = dict(chat_server._state)
    chat_server._state.update({
        "skills": registry, "skills_dir": str(skills_dir),
        "chains": chains, "chains_dir": chains.directory,
        "docs": docs, "generator": generator,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        data = json.loads(urllib.request.urlopen(
            base + "/v1/skills", timeout=10).read().decode("utf-8"))
        assert data["dir"] == str(skills_dir)
        assert {item["name"] for item in data["data"]} == {
            "demo-skill", "provider-skill", "chain-skill"}
        recipes = json.loads(urllib.request.urlopen(
            base + "/v1/skills/chains", timeout=10).read().decode("utf-8"))
        assert recipes["data"][0]["name"] == "demo-recipe"
        recipe_run = _post(base + "/v1/skills/chains/run", {
            "name": "demo-recipe", "query": "launch code"})
        assert recipe_run["recipe"] == "demo-recipe"
        assert recipe_run["reply"].startswith("Chain previous:")
        assert "Demo:" in recipe_run["reply"]
        assert recipe_run["provider_calls"] == 0
        template = _post(base + "/v1/skills/run", {
            "name": "demo-skill", "query": "launch code"})
        assert template["reply"].startswith("Demo:")
        assert "L-7781" in template["reply"]
        assert template["provider_called"] is False
        provider = _post(base + "/v1/skills/run", {
            "name": "provider-skill", "query": "launch code"})
        assert provider["reply"] == "Provider answer."
        assert provider["provider_called"] is True
        assert provider["sources"][0]["source"] == "notes/launch.txt"
        chat_server._state["generator"] = None
        fallback = _post(base + "/v1/skills/run", {
            "name": "provider-skill", "query": "launch code"})
        assert "needs a local provider" in fallback["reply"]
        assert fallback["provider_called"] is False
        assert fallback["sources"][0]["source"] == "notes/launch.txt"
        chain = _post(base + "/v1/skills/chain", {
            "skills": ["demo-skill", "chain-skill"],
            "query": "launch code"})
        assert chain["reply"].startswith("Chain previous:")
        assert "Demo:" in chain["reply"]
        assert "L-7781" in chain["reply"]
        assert chain["provider_calls"] == 0
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_graph_and_entity(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text(
        "geo.txt",
        "Ankara is the capital of Turkey. Istanbul is in Turkey.",
        title="Geography")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        graph = json.loads(urllib.request.urlopen(
            base + "/v1/graph", timeout=10).read().decode("utf-8"))
        names = {node["name"] for node in graph["nodes"]}
        assert {"Ankara", "Istanbul", "Turkey"} <= names
        edges = {(edge["source"], edge["target"])
                 for edge in graph["edges"]}
        assert ("Ankara", "Turkey") in edges
        entity = json.loads(urllib.request.urlopen(
            base + "/v1/graph/entity?name=Ankara",
            timeout=10).read().decode("utf-8"))
        assert entity["sources"][0]["source"] == "geo.txt"
        assert "Ankara" in entity["sources"][0]["snippet"]
        assert any(item["target"] == "Turkey"
                   for item in entity["connections"])
        path = json.loads(urllib.request.urlopen(
            base + "/v1/graph/path?source=Ankara&target=Istanbul",
            timeout=10).read().decode("utf-8"))
        assert path["found"] is True
        assert path["path"][0] == "Ankara"
        assert path["path"][-1] == "Istanbul"
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(
                base + "/v1/graph/entity?name=", timeout=10)
        assert missing.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_weekly_review(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                  title="Launch")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the warranty claim deadline?")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en", "gaps": gaps,
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(base + "/v1/agent/chat", {
            "session_id": "week-1", "message": "add task buy milk"})
        _post(base + "/v1/agent/chat", {
            "session_id": "week-1", "message": "add task write report"})
        _post(base + "/v1/agent/chat", {
            "session_id": "week-1", "message": "complete task 1"})
        review = _post(base + "/v1/weekly-review", {
            "session_id": "week-1", "days": 7})
        assert review["task_counts"]["open"] == 1
        assert review["task_counts"]["completed_recent"] == 1
        assert review["task_counts"]["completed_total"] == 1
        assert review["pending_gap_count"] == 1
        assert review["recent_source_count"] == 1
        assert "L-7781" in {node["name"]
                             for node in review["graph"]["entities"]}
        assert "Weekly review" in review["text"]
        assert "write report" in review["text"]
        assert "buy milk" in review["text"]
        assert review["weights_changed"] is False
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_briefing_and_goal_plan(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("notes/warranty.txt",
                  "The warranty period is 24 months. Claims need a receipt.",
                  title="Warranty")
    gaps = KnowledgeGapStore(tmp_path / "gaps.json")
    gaps.add("What is the warranty claim deadline?")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en", "gaps": gaps,
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        brief = _post(base + "/v1/briefing", {"session_id": "brief-1"})
        assert brief["open_task_count"] == 0
        assert "Daily brief" in brief["text"]
        _post(base + "/v1/agent/chat", {
            "session_id": "brief-1", "message": "add task buy milk"})
        brief = _post(base + "/v1/briefing", {"session_id": "brief-1"})
        assert brief["open_task_count"] == 1
        assert "buy milk" in brief["text"]
        plan = _post(base + "/v1/goals/plan", {
            "session_id": "brief-1", "goal": "warranty page"})
        assert plan["context"][0]["source"] == "notes/warranty.txt"
        assert plan["suggested_tasks"]
        assert plan["created_tasks"] == []
        confirmed = _post(base + "/v1/goals/plan", {
            "session_id": "brief-1", "goal": "warranty page",
            "confirm": True})
        assert len(confirmed["created_tasks"]) == 3
        after = _post(base + "/v1/briefing", {"session_id": "brief-1"})
        assert after["open_task_count"] == 4
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()
