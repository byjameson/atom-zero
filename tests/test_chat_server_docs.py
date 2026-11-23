import csv
import io
import json
import threading
from datetime import date, timedelta
import urllib.error
import urllib.parse
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

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
    assert "/v1/documents/compare" in page
    assert 'id="cmpbtn"' in page
    assert "comparesources" in page
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


def test_http_document_timeline(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("history.txt",
                  "The launch was on 5 January 2020. "
                  "A follow-up happened in 2021.", title="History")
    docs.add_text("toplanti.txt",
                  "Toplantı 6 Ocak 2020 tarihinde yapıldı.",
                  title="Toplantı")
    docs.add_text("notes.txt", "Bananas are yellow.", title="Notes")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "timeline-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        timeline = json.loads(urllib.request.urlopen(
            base + "/v1/documents/timeline",
            timeout=10).read().decode("utf-8"))
        assert [event["date"] for event in timeline["events"]] == [
            "2020-01-05", "2020-01-06", "2021"]
        assert timeline["events"][0]["source"] == "history.txt"
        assert timeline["events"][-1]["precision"] == "year"
        filtered = json.loads(urllib.request.urlopen(
            base + "/v1/documents/timeline?source=toplanti.txt",
            timeout=10).read().decode("utf-8"))
        assert filtered["count"] == 1
        assert filtered["events"][0]["date"] == "2020-01-06"
        limited = json.loads(urllib.request.urlopen(
            base + "/v1/documents/timeline?limit=1",
            timeout=10).read().decode("utf-8"))
        assert len(limited["events"]) == 1
        try:
            urllib.request.urlopen(
                base + "/v1/documents/timeline?source=missing.txt",
                timeout=10)
            raise AssertionError("missing source should return 404")
        except urllib.error.HTTPError as exc:
            assert exc.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


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
    assert "/v1/weekly-review/export" in page
    assert "/v1/weekly-review/reports" in page
    assert "/v1/goals/plan" in page
    assert "Daily brief" in page
    assert 'id="briefrefresh"' in page
    assert "getBriefing" in page
    assert 'id="briefres"' in page
    assert 'id="briefweekly"' in page
    assert 'id="brieftag"' in page
    assert 'id="weeklytag"' in page
    assert 'id="weeklyauditaction"' in page
    assert "brieftag" in page
    assert "weeklytag" in page
    assert 'id="overduebtn"' in page
    assert 'id="overdueres"' in page
    assert 'id="duesoonbtn"' in page
    assert 'id="duedays"' in page
    assert 'data-q="due soon"' in page
    assert 'data-q="search tasks report"' in page
    assert 'id="tasksearch"' in page
    assert 'id="tasksearchstatus"' in page
    assert 'id="tasksearchdue"' in page
    assert 'id="tasksearchpriority"' in page
    assert 'id="tasksearchtag"' in page
    assert 'id="tasksort"' in page
    assert 'id="tasksearchbtn"' in page
    assert 'id="tasksearchres"' in page
    assert 'id="tasksearchmsg"' in page
    assert '<option value="overdue">' in page
    assert '<option value="week">' in page
    assert '<option value="none">' in page
    assert '<option value="high">' in page
    assert '<option value="normal">' in page
    assert '<option value="priority">Priority first</option>' in page
    assert "searchTasks" in page
    assert "tasksort" in page
    assert 'id="taskliststatus"' in page
    assert 'id="tasklistdue"' in page
    assert 'id="tasklistpriority"' in page
    assert 'id="tasklisttag"' in page
    assert 'id="tasklistsort"' in page
    assert 'id="tasklistbtn"' in page
    assert 'id="tasklistcsv"' in page
    assert 'id="tasklistmd"' in page
    assert 'id="tasklistmsg"' in page
    assert 'id="tasklistres"' in page
    assert 'id="tasktagsbtn"' in page
    assert 'id="tasktagsmsg"' in page
    assert 'id="tasktagsres"' in page
    assert 'id="renameoldtag"' in page
    assert 'id="renamenewtag"' in page
    assert 'id="renametagconfirm"' in page
    assert 'id="renametagbtn"' in page
    assert 'id="renametagmsg"' in page
    assert 'id="deletetag"' in page
    assert 'id="deletetagconfirm"' in page
    assert 'id="deletetagbtn"' in page
    assert 'id="deletetagmsg"' in page
    assert 'id="undotagdeletebtn"' in page
    assert 'id="tagauditbtn"' in page
    assert 'id="tagaudittag"' in page
    assert 'id="tagauditaction"' in page
    assert 'id="tagauditcsv"' in page
    assert 'id="tagauditmd"' in page
    assert "downloadTagAudit" in page
    assert "/v1/tasks/tags/audit/export" in page
    assert 'id="tagauditmsg"' in page
    assert 'id="tagauditres"' in page
    assert "deleteTag" in page
    assert "undoTagDelete" in page
    assert "getTagAudit" in page
    assert '<option value="open" selected>Open</option>' in page
    assert "getTaskList" in page
    assert "getTaskTags" in page
    assert "renameTag" in page
    assert "downloadTaskList" in page
    assert "task.tags" in page
    assert "tasklisttag" in page
    assert "/v1/tasks/list" in page
    assert "/v1/tasks/tags" in page
    assert "/v1/tasks/export" in page
    assert page.count("document.getElementById('tasksort').value") >= 4
    assert "session_id: sessionId, sort" in page
    assert "session_id: sessionId, days: 7, sort" in page
    assert "&sort=" in page
    assert "/v1/tasks/search" in page
    assert "tasksearchstatus" in page
    assert "tasksearchdue" in page
    assert "tasksearchpriority" in page
    assert "task.priority" in page
    assert 'id="rescheduletaskid"' in page
    assert 'id="rescheduledue"' in page
    assert 'id="rescheduletaskbtn"' in page
    assert 'id="reschedulemsg"' in page
    assert "rescheduleTask" in page
    assert 'id="taskpriorityid"' in page
    assert 'id="taskprioritylevel"' in page
    assert 'id="taskprioritybtn"' in page
    assert 'id="taskprioritymsg"' in page
    assert "setTaskPriority" in page
    assert "/v1/tasks/priority" in page
    assert 'id="deletetaskid"' in page
    assert 'id="deletetaskconfirm"' in page
    assert 'id="deletetaskbtn"' in page
    assert 'id="deletetaskmsg"' in page
    assert "deleteTask" in page
    assert "/v1/tasks/delete" in page
    assert 'id="dueres"' in page
    assert "getOverdueTasks" in page
    assert "getDueSoonTasks" in page
    assert "/v1/tasks/overdue" in page
    assert "/v1/tasks/due-soon" in page
    assert 'id="duetag"' in page
    assert "duetag" in page
    assert 'id="weeklymd"' in page
    assert 'id="weeklyreportsrefresh"' in page
    assert 'id="weeklyreportstag"' in page
    assert "weeklyreportstag" in page
    assert "item.tag" in page
    assert 'id="weeklyreportsprune"' in page
    assert 'id="weeklyreportsverify"' in page
    assert 'id="weeklyreportsrepair"' in page
    assert 'id="weeklyreportscheck"' in page
    assert 'id="weeklycompareleft"' in page
    assert 'id="weeklycompareright"' in page
    assert 'id="weeklyreportscompare"' in page
    assert 'id="weeklyreportscompareout"' in page
    assert "compareWeeklyReports" in page
    assert "downloadComparisonMarkdown" in page
    assert "copyComparisonMarkdown" in page
    assert "/v1/weekly-review/reports/compare/export" in page
    assert "/v1/weekly-review/reports/verify" in page
    assert "/v1/weekly-review/reports/repair" in page
    assert "verifyWeeklyReports" in page
    assert "repairWeeklyReports" in page
    assert 'id="weeklyreportskeep"' in page
    assert 'id="weeklykeep"' in page
    assert 'id="weeklykeepdry"' in page
    assert 'id="weeklyreportspreview"' in page
    assert "/v1/weekly-review/reports/keep" in page
    assert "/v1/weekly-review/reports/delete" in page
    assert 'id="weeklyreportsearch"' in page
    assert 'id="weeklyreportssearch"' in page
    assert 'id="weeklyreportsort"' in page
    assert 'value="oldest"' in page
    assert 'value="largest"' in page
    assert 'data-q="find report weekly"' in page
    assert 'data-q="list reports"' in page
    assert 'data-q="list reports oldest"' in page
    assert 'data-q="verify report archive"' in page
    assert 'data-q="repair report archive"' in page
    assert 'data-q="keep reports 2 dry-run"' in page
    assert "show report w0001" in page
    assert "compare reports w0001 w0002" in page
    assert "repair report w0001" in page
    assert "getWeeklyReview" in page
    assert "downloadWeeklyMarkdown" in page
    assert "refreshWeeklyReports" in page
    assert "downloadArchivedReport" in page
    assert "deleteArchivedReport" in page
    assert "pruneWeeklyReports" in page
    assert "keepWeeklyReports" in page
    assert "params.set('query'" in page
    assert "dry_run" in page
    assert "would_remove" in page
    assert "planGoal" in page
    assert 'data-q="plan goal prepare the warranty page"' in page
    assert 'data-q="brief me"' in page
    assert 'data-q="brief me priority"' in page
    assert 'data-q="list tasks priority"' in page
    assert 'data-q="list tasks done"' in page
    assert 'data-q="list tasks due week priority high"' in page
    assert 'data-q="weekly review priority"' in page
    assert 'data-q="due soon priority"' in page
    assert 'data-q="overdue tasks priority"' in page
    assert 'data-q="search tasks report sort priority"' in page


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


def test_http_chat_source_comparison(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("policy_a.txt",
                  "The warranty period is 24 months. "
                  "The warranty covers water damage.", title="Policy A")
    docs.add_text("policy_b.txt",
                  "The warranty period is 36 months. "
                  "The warranty does not cover water damage.",
                  title="Policy B")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": docs, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        result = _post(base + "/v1/agent/chat", {
            "session_id": "compare-1",
            "message": "compare sources policy_a.txt and policy_b.txt"})
        assert "numeric_mismatch" in result["reply"]
        assert "24 months" in result["reply"]
        assert "36 months" in result["reply"]
        assert result["tool_calls"][0]["name"] == "compare_sources"
        assert result["comparison"]["summary"]["conflict_count"] >= 2
        assert len(result["sources"]) == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_source_comparison(tmp_path):
    try:
        docs = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    docs.add_text("policy_a.txt",
                  "The warranty period is 24 months. "
                  "The warranty covers water damage.", title="Policy A")
    docs.add_text("policy_b.txt",
                  "The warranty period is 36 months. "
                  "The warranty does not cover water damage.",
                  title="Policy B")
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
        result = json.loads(urllib.request.urlopen(
            base + "/v1/documents/compare?source=policy_a.txt"
                   "&against=policy_b.txt", timeout=10).read().decode())
        reasons = {reason for conflict in result["conflicts"]
                   for reason in conflict["reasons"]}
        assert "numeric_mismatch" in reasons
        assert "negation_mismatch" in reasons
        assert result["summary"]["conflict_count"] >= 2
        assert result["weights_changed"] is False
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(
                base + "/v1/documents/compare?source=missing.txt",
                timeout=10)
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
            "session_id": "week-1", "message": "add task buy milk tags home"})
        _post(base + "/v1/agent/chat", {
            "session_id": "week-1",
            "message": "add task write report tags work,home"})
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
        filtered = _post(base + "/v1/weekly-review", {
            "session_id": "week-1", "days": 7, "tag": "work"})
        assert filtered["tag"] == "work"
        assert filtered["task_counts"] == {
            "open": 1, "completed_recent": 0, "completed_total": 0}
        assert [task["id"] for task in filtered["open_tasks"]] == [2]
        assert filtered["tags"] == [
            {"tag": "home", "open": 1, "done": 0, "total": 1},
            {"tag": "work", "open": 1, "done": 0, "total": 1},
        ]
        assert "| tag=work" in filtered["text"]
        assert "Tags for #work: 2 distinct" in filtered["text"]
        assert "write report" in filtered["text"]
        assert "buy milk" not in filtered["text"]
        with pytest.raises(urllib.error.HTTPError) as invalid_tag:
            _post(base + "/v1/weekly-review", {
                "session_id": "week-1", "days": 7, "tag": "bad tag!"})
        assert invalid_tag.value.code == 400
        with urllib.request.urlopen(
                base + "/v1/weekly-review/export"
                       "?session_id=week-1&days=7", timeout=10) as response:
            assert response.headers.get_content_type() == "text/markdown"
            assert "attachment" in response.headers[
                "Content-Disposition"]
            assert "atomzero-weekly-review-" in response.headers[
                "Content-Disposition"]
            markdown = response.read().decode("utf-8")
        assert markdown.startswith("# Weekly review - ")
        assert "Period: last 7 days" in markdown
        assert "- [ ] **#2** write report" in markdown
        assert "- [x] **#1** buy milk" in markdown
        assert "What is the warranty claim deadline?" in markdown
        assert "`notes/launch.txt`" in markdown
        assert "L-7781" in markdown
        with urllib.request.urlopen(
                base + "/v1/weekly-review/export"
                       "?session_id=week-1&days=7&tag=work",
                timeout=10) as response:
            filtered_markdown = response.read().decode("utf-8")
        assert filtered_markdown.splitlines()[0].endswith("| tag=work")
        assert "## Tags for #work (2 tags)" in filtered_markdown
        assert "- [ ] **#2** write report" in filtered_markdown
        assert "buy milk" not in filtered_markdown
        with pytest.raises(urllib.error.HTTPError) as invalid_tag:
            urllib.request.urlopen(
                base + "/v1/weekly-review/export?"
                       "session_id=week-1&tag=bad%20tag%21",
                timeout=10)
        assert invalid_tag.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid:
            urllib.request.urlopen(
                base + "/v1/weekly-review/export?days=not-a-number",
                timeout=10)
        assert invalid.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()


def test_http_weekly_reports_archive(tmp_path):
    from atomzero.weekly_reports import WeeklyReportArchive

    report = tmp_path / "weekly.md"
    report.write_text("# Weekly review\n\n- [ ] archived task\n",
                      encoding="utf-8")
    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    entry = archive.record(report, {
        "generated": "2026-11-22 09:00",
        "period_days": 7,
        "tag": "work",
        "task_counts": {"open": 1, "completed_recent": 0,
                        "completed_total": 1},
        "pending_gap_count": 0,
        "recent_source_count": 1,
        "graph": {"entities": [{"name": "L-7781"}], "connections": []},
    }, exported_at="2026-11-22T09:00:00")
    missing_report = tmp_path / "gone.md"
    missing_report.write_text("# Gone\n", encoding="utf-8")
    missing_entry = archive.record(missing_report, {
        "generated": "2026-11-22 10:00", "period_days": 14,
        "tag": "home",
        "task_counts": {"open": 0}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    })
    missing_report.unlink()
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "weekly_reports_path": str(index_path),
        "weekly_reports": archive,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        listing = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5",
            timeout=10).read().decode("utf-8"))
        assert listing["total"] == 2
        by_id = {item["id"]: item for item in listing["data"]}
        assert by_id[entry["id"]]["exists"] is True
        assert by_id[missing_entry["id"]]["exists"] is False
        work = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&tag=work",
            timeout=10).read().decode("utf-8"))
        assert work["tag"] == "work"
        assert work["matched"] == 1
        assert work["total"] == 2
        assert [item["id"] for item in work["data"]] == [entry["id"]]
        assert work["data"][0]["tag"] == "work"
        home = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&tag=home",
            timeout=10).read().decode("utf-8"))
        assert [item["id"] for item in home["data"]] == [
            missing_entry["id"]]
        missing_tag = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&tag=missing",
            timeout=10).read().decode("utf-8"))
        assert missing_tag["data"] == []
        assert missing_tag["matched"] == 0
        with pytest.raises(urllib.error.HTTPError) as invalid_tag:
            urllib.request.urlopen(
                base + "/v1/weekly-review/reports?tag=bad%20tag%21",
                timeout=10)
        assert invalid_tag.value.code == 400
        oldest = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&sort=oldest",
            timeout=10).read().decode("utf-8"))
        newest = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&sort=newest",
            timeout=10).read().decode("utf-8"))
        largest = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&sort=largest",
            timeout=10).read().decode("utf-8"))
        assert oldest["sort"] == "oldest"
        assert oldest["data"][0]["id"] == entry["id"]
        assert newest["data"][0]["id"] == missing_entry["id"]
        assert largest["data"][0]["id"] == entry["id"]
        with pytest.raises(urllib.error.HTTPError) as invalid_sort:
            urllib.request.urlopen(
                base + "/v1/weekly-review/reports?sort=random", timeout=10)
        assert invalid_sort.value.code == 400
        chat = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat",
            "message": "find report weekly"})
        assert chat["tool_calls"][0]["name"] == "search_weekly_reports"
        assert chat["matched"] == 2
        assert entry["id"] in chat["reply"]
        assert missing_entry["id"] in chat["reply"]
        listed = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat", "message": "list reports 1"})
        assert listed["tool_calls"][0]["name"] == "list_weekly_reports"
        assert "Archived weekly reports (1 of 2)" in listed["reply"]
        assert len(listed["reports"]) == 1
        ordered = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat",
            "message": "list reports oldest"})
        assert ordered["sort"] == "oldest"
        assert ordered["reports"][0]["id"] == entry["id"]
        sorted_search = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat",
            "message": "find report weekly largest"})
        assert sorted_search["sort"] == "largest"
        assert sorted_search["reports"][0]["id"] == entry["id"]
        shown = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat",
            "message": f"show report {entry['id']}"})
        assert shown["tool_calls"][0]["name"] == "show_weekly_report"
        assert "archived task" in shown["report_markdown"]
        assert shown["report_id"] == entry["id"]
        verification = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports/verify",
            timeout=10).read().decode("utf-8"))
        assert verification["summary"]["total"] == 2
        assert verification["summary"]["ok"] == 1
        assert verification["summary"]["missing"] == 1
        verified_by_id = {
            item["id"]: item for item in verification["entries"]}
        assert verified_by_id[entry["id"]]["status"] == "ok"
        assert verified_by_id[missing_entry["id"]]["status"] == "missing"
        comparison = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports/compare"
                   f"?left={entry['id']}&right={missing_entry['id']}",
            timeout=10).read().decode("utf-8"))
        comparison_metrics = {
            item["key"]: item for item in comparison["metrics"]}
        assert comparison["left"]["id"] == entry["id"]
        assert comparison["right"]["id"] == missing_entry["id"]
        assert comparison["hash_equal"] is False
        assert comparison_metrics["recent_source_count"]["delta"] == -1
        assert comparison_metrics["size_bytes"]["delta"] != 0
        with pytest.raises(urllib.error.HTTPError) as missing_compare:
            urllib.request.urlopen(
                base + "/v1/weekly-review/reports/compare"
                       "?left=w0001&right=w9999", timeout=10)
        assert missing_compare.value.code == 404
        with urllib.request.urlopen(
                base + "/v1/weekly-review/reports/compare/export"
                       f"?left={entry['id']}&right={missing_entry['id']}",
                timeout=10) as export_response:
            assert export_response.headers.get_content_type() == (
                "text/markdown")
            assert "attachment" in export_response.headers[
                "Content-Disposition"]
            comparison_markdown = export_response.read().decode("utf-8")
        assert comparison_markdown.startswith(
            f"# Report comparison: {entry['id']} -> {missing_entry['id']}")
        assert "| Metric | Left | Right | Delta |" in comparison_markdown
        assert "## Content hash" in comparison_markdown
        chat_verify = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat",
            "message": "verify report archive"})
        assert chat_verify["verification"]["summary"]["missing"] == 1
        assert "1 ok, 1 failed" in chat_verify["reply"]
        chat_one = _post(base + "/v1/agent/chat", {
            "session_id": "report-chat",
            "message": f"verify report {entry['id']}"})
        assert chat_one["ok"] is True
        assert chat_one["report_id"] == entry["id"]
        pruned = _post(base + "/v1/weekly-review/reports/prune", {})
        assert pruned["removed"] == 1
        assert pruned["remaining"] == 1
        listing = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5",
            timeout=10).read().decode("utf-8"))
        assert listing["total"] == 1
        assert listing["data"][0]["id"] == entry["id"]
        filtered = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&query=weekly",
            timeout=10).read().decode("utf-8"))
        assert filtered["matched"] == 1
        assert filtered["data"][0]["id"] == entry["id"]
        empty_filter = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5&query=missing",
            timeout=10).read().decode("utf-8"))
        assert empty_filter["matched"] == 0
        assert empty_filter["data"] == []
        response = urllib.request.urlopen(
            base + f"/v1/weekly-review/reports?id={entry['id']}",
            timeout=10)
        assert response.headers.get_content_type() == "text/markdown"
        assert "weekly.md" in response.headers["Content-Disposition"]
        assert "archived task" in response.read().decode("utf-8")
        for number in range(1, 3):
            extra = tmp_path / f"extra-{number}.md"
            extra.write_text(f"# Extra {number}\n", encoding="utf-8")
            archive.record(extra, {
                "generated": f"2026-11-2{number} 09:00", "period_days": 7,
                "task_counts": {"open": 0}, "recent_source_count": 0,
                "graph": {"entities": [], "connections": []},
            }, exported_at=f"2026-11-2{number}T09:00:00")
        preview = _post(base + "/v1/weekly-review/reports/keep",
                        {"keep": 2, "dry_run": True})
        assert preview["dry_run"] is True
        assert preview["trimmed"] == 0
        assert preview["would_remove_count"] == 1
        assert preview["remaining"] == 3
        listing = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5",
            timeout=10).read().decode("utf-8"))
        assert listing["total"] == 3
        retention = _post(
            base + "/v1/weekly-review/reports/keep", {"keep": 2})
        assert retention["trimmed"] == 1
        assert retention["remaining"] == 2
        listing = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5",
            timeout=10).read().decode("utf-8"))
        assert listing["total"] == 2
        delete_target = listing["data"][0]
        deleted = _post(
            base + "/v1/weekly-review/reports/delete",
            {"id": delete_target["id"]})
        assert deleted["deleted"]["id"] == delete_target["id"]
        assert deleted["remaining"] == 1
        assert Path(delete_target["path"]).exists()
        listing = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports?limit=5",
            timeout=10).read().decode("utf-8"))
        assert listing["total"] == 1
        with pytest.raises(urllib.error.HTTPError) as missing:
            urllib.request.urlopen(
                base + "/v1/weekly-review/reports?id=w9999", timeout=10)
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_chat_report_maintenance_flow(tmp_path):
    from atomzero.weekly_reports import WeeklyReportArchive

    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    archive.record(missing, {
        "generated": "2026-11-20 09:00", "period_days": 7,
        "task_counts": {}, "recent_source_count": 0,
        "graph": {"entities": [], "connections": []},
    }, exported_at="2026-11-20T09:00:00")
    missing.unlink()
    for number in range(1, 5):
        report = tmp_path / f"report-{number}.md"
        report.write_text(f"# Report {number}\n", encoding="utf-8")
        archive.record(report, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {"open": number}, "recent_source_count": 1,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "weekly_reports_path": str(index_path),
        "weekly_reports": archive,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        prune = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat",
            "message": "prune report archive"})
        assert 'Reply "confirm reports"' in prune["reply"]
        assert archive.count() == 5
        applied = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat", "message": "confirm reports"})
        assert "Pruned 1 missing report(s)." in applied["reply"]
        assert archive.count() == 4

        keep = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat", "message": "keep reports 2"})
        assert 'Reply "confirm reports"' in keep["reply"]
        assert archive.count() == 4
        kept = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat", "message": "confirm reports"})
        assert "Kept the newest 2 report(s); removed 2" in kept["reply"]
        assert archive.count() == 2

        dry_run = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat",
            "message": "keep reports 1 dry-run"})
        assert dry_run["dry_run"] is True
        assert dry_run["would_remove_count"] == 1
        assert archive.count() == 2

        target = archive.list()[0]
        delete = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat",
            "message": f"delete report {target['id']}"})
        assert 'Reply "confirm reports"' in delete["reply"]
        assert archive.count() == 2
        deleted = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat", "message": "confirm reports"})
        assert f"Deleted archive entry {target['id']}." in deleted["reply"]
        assert archive.count() == 1
        assert Path(target["path"]).exists()
        cancelled = _post(base + "/v1/agent/chat", {
            "session_id": "maint-chat", "message": "cancel reports"})
        assert "no pending report action" in cancelled["reply"].lower()
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_weekly_reports_repair(tmp_path):
    from atomzero.weekly_reports import WeeklyReportArchive

    index_path = tmp_path / "weekly_reports.json"
    archive = WeeklyReportArchive(index_path)
    good = tmp_path / "good.md"
    good.write_text("# Good\n", encoding="utf-8")
    changed = tmp_path / "changed.md"
    changed.write_text("# Alpha\n", encoding="utf-8")
    missing = tmp_path / "missing.md"
    missing.write_text("# Missing\n", encoding="utf-8")
    for number, path in enumerate((good, changed, missing), start=1):
        archive.record(path, {
            "generated": f"2026-11-2{number} 09:00", "period_days": 7,
            "task_counts": {}, "recent_source_count": 0,
            "graph": {"entities": [], "connections": []},
        }, exported_at=f"2026-11-2{number}T09:00:00")
    changed.write_text("# Bravo changed\n", encoding="utf-8")
    missing.unlink()
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "weekly_reports_path": str(index_path),
        "weekly_reports": archive,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        repaired = _post(
            base + "/v1/weekly-review/reports/repair", {})
        assert repaired["summary"]["repaired"] == 1
        assert repaired["summary"]["ok"] == 1
        assert repaired["summary"]["missing"] == 1
        by_name = {item["name"]: item for item in repaired["entries"]}
        assert by_name["changed.md"]["status"] == "repaired"
        verification = json.loads(urllib.request.urlopen(
            base + "/v1/weekly-review/reports/verify",
            timeout=10).read().decode("utf-8"))
        assert verification["summary"]["ok"] == 2
        assert verification["summary"]["missing"] == 1

        good.write_text("# Good changed\n", encoding="utf-8")
        chat_preview = _post(base + "/v1/agent/chat", {
            "session_id": "repair-chat",
            "message": "repair report archive"})
        assert 'Reply "confirm reports"' in chat_preview["reply"]
        assert archive.verify()["summary"]["failed"] == 2
        chat_repair = _post(base + "/v1/agent/chat", {
            "session_id": "repair-chat", "message": "confirm reports"})
        assert "Repaired 1 report metadata" in chat_repair["reply"]
        assert archive.verify()["summary"]["ok"] == 2

        changed.write_text("# Changed single\n", encoding="utf-8")
        single_preview = _post(base + "/v1/agent/chat", {
            "session_id": "repair-chat", "message": "repair report w0002"})
        assert "Report w0002 metadata repair preview" in single_preview[
            "reply"]
        assert archive.verify()["summary"]["failed"] == 2
        single_repair = _post(base + "/v1/agent/chat", {
            "session_id": "repair-chat", "message": "confirm reports"})
        assert "Report w0002 metadata repair: 1 repaired" in single_repair[
            "reply"]
        assert archive.verify()["summary"]["ok"] == 2
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_overdue_tasks(tmp_path):
    overdue = (date.today() - timedelta(days=1)).isoformat()
    future = (date.today() + timedelta(days=1)).isoformat()
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(base + "/v1/agent/chat", {
            "session_id": "due-http",
            "message": f"add task submit report due {overdue} tags work"})
        _post(base + "/v1/agent/chat", {
            "session_id": "due-http",
            "message": f"add task plan launch due {future} tags home"})
        result = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/overdue?session_id=due-http"
                   f"&today={date.today().isoformat()}",
            timeout=10).read().decode("utf-8"))
        assert result["open_task_count"] == 2
        assert result["overdue_task_count"] == 1
        assert result["overdue_tasks"][0]["text"] == "submit report"
        assert result["overdue_tasks"][0]["due"] == overdue
        filtered = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/overdue?session_id=due-http&tag=work"
                   f"&today={date.today().isoformat()}",
            timeout=10).read().decode("utf-8"))
        assert filtered["tag"] == "work"
        assert filtered["open_task_count"] == 1
        assert filtered["overdue_task_count"] == 1
        assert filtered["overdue_tasks"][0]["text"] == "submit report"
        other = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/overdue?session_id=due-http&tag=home"
                   f"&today={date.today().isoformat()}",
            timeout=10).read().decode("utf-8"))
        assert other["open_task_count"] == 1
        assert other["overdue_task_count"] == 0
        with pytest.raises(urllib.error.HTTPError) as invalid_tag:
            urllib.request.urlopen(
                base + "/v1/tasks/overdue?session_id=due-http"
                       "&tag=bad%20tag%21",
                timeout=10)
        assert invalid_tag.value.code == 400
        empty = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/overdue?session_id=empty",
            timeout=10).read().decode("utf-8"))
        assert empty["open_task_count"] == 0
        assert empty["overdue_tasks"] == []
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_due_soon_tasks(tmp_path):
    today = date.today()
    due_soon = (today + timedelta(days=2)).isoformat()
    later = (today + timedelta(days=10)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        for text, due, tag in (("submit report", due_soon, "work"),
                               ("plan launch", later, "home"),
                               ("late task", overdue, "work")):
            _post(base + "/v1/agent/chat", {
                "session_id": "due-soon-http",
                "message": f"add task {text} due {due} tags {tag}"})
        result = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/due-soon?session_id=due-soon-http&days=7"
                   f"&today={today.isoformat()}",
            timeout=10).read().decode("utf-8"))
        assert result["days"] == 7
        assert result["open_task_count"] == 3
        assert result["due_soon_count"] == 1
        assert result["due_soon_tasks"][0]["text"] == "submit report"
        assert result["window_end"] == (
            today + timedelta(days=7)).isoformat()
        filtered = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/due-soon?session_id=due-soon-http&days=7"
                   f"&tag=work&today={today.isoformat()}",
            timeout=10).read().decode("utf-8"))
        assert filtered["tag"] == "work"
        assert filtered["open_task_count"] == 2
        assert filtered["due_soon_count"] == 1
        assert filtered["due_soon_tasks"][0]["text"] == "submit report"
        other = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/due-soon?session_id=due-soon-http&days=7"
                   f"&tag=home&today={today.isoformat()}",
            timeout=10).read().decode("utf-8"))
        assert other["open_task_count"] == 1
        assert other["due_soon_count"] == 0
        with pytest.raises(urllib.error.HTTPError) as invalid_tag:
            urllib.request.urlopen(
                base + "/v1/tasks/due-soon?session_id=due-soon-http"
                       "&tag=bad%20tag%21",
                timeout=10)
        assert invalid_tag.value.code == 400
        chat = _post(base + "/v1/agent/chat", {
            "session_id": "due-soon-http", "message": "due soon"})
        assert chat["tool_calls"][0]["name"] == "due_soon"
        assert chat["due_soon"]["due_soon_count"] == 1
        brief = _post(base + "/v1/briefing", {
            "session_id": "due-soon-http"})
        assert brief["due_soon_count"] == 1
        assert "1 due soon" in brief["text"]
        weekly = _post(base + "/v1/weekly-review", {
            "session_id": "due-soon-http", "days": 7})
        assert weekly["due_soon_count"] == 1
        assert "Due soon: 1 (next 7 days)" in weekly["text"]
        with pytest.raises(urllib.error.HTTPError) as invalid_days:
            urllib.request.urlopen(
                base + "/v1/tasks/due-soon?session_id=due-soon-http&days=0",
                timeout=10)
        assert invalid_days.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_search_tasks(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    today = date.today()
    week = (today + timedelta(days=3)).isoformat()
    overdue = (today - timedelta(days=1)).isoformat()
    try:
        for text, due, priority, tags in (
                ("submit quarterly report", week, "high", "work"),
                ("submit expenses", overdue, "low", "finance"),
                ("plan launch", "", "normal", "work"),
                ("submit archived report", "", "high", "work"),
                ("submit today report", today.isoformat(), "low",
                 "finance")):
            message = (f"add task {text} priority {priority} "
                       f"tags {tags}")
            if due:
                message += f" due {due}"
            _post(base + "/v1/agent/chat", {
                "session_id": "task-search-http", "message": message})
        _post(base + "/v1/agent/chat", {
            "session_id": "task-search-http", "message": "complete task 4"})

        def search(**filters):
            params = {
                "session_id": "task-search-http",
                "q": "submit",
                "today": today.isoformat(),
            }
            params.update(filters)
            return json.loads(urllib.request.urlopen(
                base + "/v1/tasks/search?" + urllib.parse.urlencode(params),
                timeout=10).read().decode("utf-8"))

        result = search(q="submit report")
        assert result["query"] == "submit report"
        assert result["status"] == "all"
        assert result["due"] == "all"
        assert result["priority"] == "all"
        assert result["tag"] == "all"
        assert result["count"] == 3
        assert [task["id"] for task in result["tasks"]] == [1, 4, 5]
        assert result["open_task_count"] == 4
        assert result["weights_changed"] is False
        open_only = search(status="open")
        assert open_only["status"] == "open"
        assert [task["id"] for task in open_only["tasks"]] == [1, 2, 5]
        week_only = search(due="week")
        assert week_only["due"] == "week"
        assert [task["id"] for task in week_only["tasks"]] == [1, 5]
        today_only = search(due="today")
        assert [task["id"] for task in today_only["tasks"]] == [5]
        overdue_only = search(due="overdue")
        assert [task["id"] for task in overdue_only["tasks"]] == [2]
        done_only = search(status="done")
        assert [task["id"] for task in done_only["tasks"]] == [4]
        undated = search(due="none")
        assert [task["id"] for task in undated["tasks"]] == [4]
        combined = search(status="open", due="week")
        assert [task["id"] for task in combined["tasks"]] == [1, 5]
        high = search(priority="high")
        assert high["priority"] == "high"
        assert [task["id"] for task in high["tasks"]] == [1, 4]
        low = search(priority="low")
        assert [task["id"] for task in low["tasks"]] == [2, 5]
        normal = search(q="plan", priority="normal")
        assert [task["id"] for task in normal["tasks"]] == [3]
        combined = search(status="open", due="week", priority="high")
        assert [task["id"] for task in combined["tasks"]] == [1]
        work_tag = search(tag="work")
        assert work_tag["tag"] == "work"
        assert [task["id"] for task in work_tag["tasks"]] == [1, 4]
        finance_tag = search(tag="finance")
        assert [task["id"] for task in finance_tag["tasks"]] == [2, 5]
        combined = search(
            status="open", due="week", priority="high", tag="work")
        assert [task["id"] for task in combined["tasks"]] == [1]
        empty = search(q="missing")
        assert empty["count"] == 0
        assert empty["tasks"] == []
        tag_view = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags?session_id=task-search-http&today="
            + urllib.parse.quote(today.isoformat()),
            timeout=10).read().decode("utf-8"))
        assert tag_view["tag_count"] == 2
        assert tag_view["task_count"] == 5
        assert tag_view["tagged_task_count"] == 5
        assert tag_view["untagged_task_count"] == 0
        assert tag_view["open_tagged"] == 4
        assert tag_view["done_tagged"] == 1
        assert tag_view["tags"] == [
            {"tag": "work", "open": 2, "done": 1, "total": 3},
            {"tag": "finance", "open": 2, "done": 0, "total": 2},
        ]
        assert "Task tags: 2 distinct" in tag_view["text"]
        assert "- work: 2 open, 1 done, 3 total" in tag_view["text"]
        _post(base + "/v1/agent/chat", {
            "session_id": "task-search-http",
            "message": "add task untagged note"})
        tag_view = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags?session_id=task-search-http&today="
            + urllib.parse.quote(today.isoformat()),
            timeout=10).read().decode("utf-8"))
        assert tag_view["untagged_task_count"] == 1
        assert tag_view["task_count"] == 6
        for bad in ({"status": "blocked"}, {"due": "month"},
                    {"priority": "urgent"}):
            with pytest.raises(urllib.error.HTTPError) as invalid:
                search(**bad)
            assert invalid.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as missing_query:
            urllib.request.urlopen(
                base + "/v1/tasks/search?session_id=task-search-http",
                timeout=10)
        assert missing_query.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_rename_tag(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    session_id = "task-rename-http"
    try:
        for message in (
                "add task write report tags work,home",
                "add task pay invoice tags work,finance",
                "add task read book",
                "complete task 2"):
            _post(base + "/v1/agent/chat", {
                "session_id": session_id, "message": message})

        def tags():
            return json.loads(urllib.request.urlopen(
                base + "/v1/tasks/tags?session_id=" + session_id,
                timeout=10).read().decode("utf-8"))

        before = tags()
        assert [item["tag"] for item in before["tags"]] == [
            "work", "finance", "home"]
        with pytest.raises(urllib.error.HTTPError) as need_confirm:
            _post(base + "/v1/tasks/tags/rename", {
                "session_id": session_id, "old": "work", "new": "job"})
        assert need_confirm.value.code == 400
        detail = json.loads(need_confirm.value.read().decode("utf-8"))
        assert detail["error"] == "confirmation required"
        assert detail["old"] == "work"
        assert detail["new"] == "job"
        assert detail["would_update"] == 2
        assert detail["open"] == 1
        assert detail["done"] == 1
        assert [item["tag"] for item in tags()["tags"]] == [
            "work", "finance", "home"]

        applied = _post(base + "/v1/tasks/tags/rename", {
            "session_id": session_id, "old": "work", "new": "job",
            "confirm": True})
        assert applied["applied"] is True
        assert applied["old"] == "work"
        assert applied["new"] == "job"
        assert applied["updated"] == 2
        assert applied["task_ids"] == [1, 2]
        after = tags()
        assert [item["tag"] for item in after["tags"]] == [
            "job", "finance", "home"]
        assert after["tag_count"] == 3
        assert after["tagged_task_count"] == 2
        assert after["untagged_task_count"] == 1

        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/tasks/tags/rename", {
                "session_id": session_id, "old": "missing", "new": "job",
                "confirm": True})
        assert missing.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as same:
            _post(base + "/v1/tasks/tags/rename", {
                "session_id": session_id, "old": "job", "new": "job",
                "confirm": True})
        assert same.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid:
            _post(base + "/v1/tasks/tags/rename", {
                "session_id": session_id, "old": "job", "new": "bad tag!",
                "confirm": True})
        assert invalid.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_delete_tag(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    session_id = "task-delete-tag-http"
    try:
        for message in (
                "add task write report tags work,home",
                "add task pay invoice tags work,finance",
                "add task read book",
                "complete task 2"):
            _post(base + "/v1/agent/chat", {
                "session_id": session_id, "message": message})

        def tags():
            return json.loads(urllib.request.urlopen(
                base + "/v1/tasks/tags?session_id=" + session_id,
                timeout=10).read().decode("utf-8"))

        with pytest.raises(urllib.error.HTTPError) as need_confirm:
            _post(base + "/v1/tasks/tags/delete", {
                "session_id": session_id, "tag": "work"})
        assert need_confirm.value.code == 400
        detail = json.loads(need_confirm.value.read().decode("utf-8"))
        assert detail["error"] == "confirmation required"
        assert detail["tag"] == "work"
        assert detail["would_update"] == 2
        assert detail["open"] == 1
        assert detail["done"] == 1
        before = tags()
        assert before["tag_count"] == 3

        applied = _post(base + "/v1/tasks/tags/delete", {
            "session_id": session_id, "tag": "work", "confirm": True})
        assert applied["applied"] is True
        assert applied["tag"] == "work"
        assert applied["updated"] == 2
        assert applied["task_ids"] == [1, 2]
        after = tags()
        assert [item["tag"] for item in after["tags"]] == [
            "finance", "home"]
        assert after["tag_count"] == 2
        assert after["tagged_task_count"] == 2
        assert after["untagged_task_count"] == 1

        undo = _post(base + "/v1/tasks/tags/undo-delete", {
            "session_id": session_id})
        assert undo["tag"] == "work"
        assert undo["restored"] == 2
        assert undo["task_ids"] == [1, 2]
        assert undo["already"] == []
        assert undo["skipped"] == []
        restored = tags()
        assert [item["tag"] for item in restored["tags"]] == [
            "work", "finance", "home"]
        assert restored["tag_count"] == 3
        assert restored["tagged_task_count"] == 2
        with pytest.raises(urllib.error.HTTPError) as no_undo:
            _post(base + "/v1/tasks/tags/undo-delete", {
                "session_id": session_id})
        assert no_undo.value.code == 404

        audit = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags/audit?session_id=" + session_id,
            timeout=10).read().decode("utf-8"))
        assert audit["count"] == 2
        assert audit["total"] == 2
        assert [event["action"] for event in audit["events"]] == [
            "undo_tag_delete", "delete_tag"]
        assert audit["events"][0]["restored"] == 2
        assert audit["events"][1]["updated"] == 2
        limited_audit = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags/audit?session_id=" + session_id
                   + "&limit=1",
            timeout=10).read().decode("utf-8"))
        assert limited_audit["count"] == 1
        assert limited_audit["total"] == 2
        filtered_audit = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags/audit?session_id=" + session_id
                   + "&tag=work&action=delete_tag",
            timeout=10).read().decode("utf-8"))
        assert filtered_audit["tag"] == "work"
        assert filtered_audit["action"] == "delete_tag"
        assert filtered_audit["count"] == 1
        assert filtered_audit["total"] == 1
        assert filtered_audit["events"][0]["action"] == "delete_tag"
        with urllib.request.urlopen(
                base + "/v1/tasks/tags/audit/export?session_id=" + session_id
                       + "&tag=work&action=delete_tag&format=csv",
                timeout=10) as response:
            assert response.headers.get_content_type() == "text/csv"
            assert "atomzero-tag-audit-" in response.headers[
                "Content-Disposition"]
            csv_audit = response.read().decode("utf-8")
        assert csv_audit.startswith("ts,action,tag,task_ids")
        assert "delete_tag,work" in csv_audit
        assert "undo_tag_delete" not in csv_audit
        with urllib.request.urlopen(
                base + "/v1/tasks/tags/audit/export?session_id=" + session_id
                       + "&tag=work&action=delete_tag&format=markdown",
                timeout=10) as response:
            assert response.headers.get_content_type() == "text/markdown"
            md_audit = response.read().decode("utf-8")
        assert md_audit.startswith("# Tag audit - ")
        assert "| delete_tag | work |" in md_audit
        assert "| undo_tag_delete | work |" not in md_audit
        with pytest.raises(urllib.error.HTTPError) as invalid_format:
            urllib.request.urlopen(
                base + "/v1/tasks/tags/audit/export?session_id=" + session_id
                       + "&format=xml",
                timeout=10)
        assert invalid_format.value.code == 400
        undo_audit = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags/audit?session_id=" + session_id
                   + "&action=undo",
            timeout=10).read().decode("utf-8"))
        assert undo_audit["action"] == "undo_tag_delete"
        assert undo_audit["count"] == 1
        missing_audit = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/tags/audit?session_id=" + session_id
                   + "&tag=missing",
            timeout=10).read().decode("utf-8"))
        assert missing_audit["count"] == 0
        assert missing_audit["total"] == 0
        with pytest.raises(urllib.error.HTTPError) as invalid_audit_tag:
            urllib.request.urlopen(
                base + "/v1/tasks/tags/audit?session_id=" + session_id
                       + "&tag=bad%20tag%21",
                timeout=10)
        assert invalid_audit_tag.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid_audit_action:
            urllib.request.urlopen(
                base + "/v1/tasks/tags/audit?session_id=" + session_id
                       + "&action=random",
                timeout=10)
        assert invalid_audit_action.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid_limit:
            urllib.request.urlopen(
                base + "/v1/tasks/tags/audit?session_id=" + session_id
                       + "&limit=0",
                timeout=10)
        assert invalid_limit.value.code == 400

        brief = _post(base + "/v1/briefing", {
            "session_id": session_id})
        assert brief["tag_audit_total"] == 2
        assert [event["action"] for event in brief["tag_audit"]] == [
            "undo_tag_delete", "delete_tag"]
        assert "Tag audit: 2 event(s)" in brief["text"]
        fetched_brief = json.loads(urllib.request.urlopen(
            base + "/v1/briefing?session_id=" + session_id,
            timeout=10).read().decode("utf-8"))
        assert fetched_brief["tag_audit_total"] == 2
        home_brief = _post(base + "/v1/briefing", {
            "session_id": session_id, "tag": "home"})
        assert home_brief["tag_audit_total"] == 0
        assert "Tag audit" not in home_brief["text"]

        weekly = _post(base + "/v1/weekly-review", {
            "session_id": session_id, "days": 7, "tag": "work",
            "audit_action": "delete_tag"})
        assert weekly["tag_audit_total"] == 1
        assert weekly["tag_audit_action"] == "delete_tag"
        assert [event["action"] for event in weekly["tag_audit"]] == [
            "delete_tag"]
        assert ("Tag audit: 1 event(s) | tag=work | action=delete_tag"
                in weekly["text"])
        with urllib.request.urlopen(
                base + "/v1/weekly-review/export?session_id=" + session_id
                       + "&days=7&tag=work&audit_action=delete_tag",
                timeout=10) as response:
            weekly_markdown = response.read().decode("utf-8")
        assert ("## Tag audit (1) | tag=work | action=delete_tag"
                in weekly_markdown)
        assert "undo delete tag 'work'" not in weekly_markdown

        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/tasks/tags/delete", {
                "session_id": session_id, "tag": "missing",
                "confirm": True})
        assert missing.value.code == 404
        with pytest.raises(urllib.error.HTTPError) as invalid:
            _post(base + "/v1/tasks/tags/delete", {
                "session_id": session_id, "tag": "bad tag!",
                "confirm": True})
        assert invalid.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_update_task_priority(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(base + "/v1/agent/chat", {
            "session_id": "priority-http",
            "message": "add task submit report priority low"})
        updated = _post(base + "/v1/tasks/priority", {
            "session_id": "priority-http", "id": 1, "priority": "high"})
        assert updated["session_id"] == "priority-http"
        assert updated["task"]["id"] == 1
        assert updated["task"]["priority"] == "high"
        search = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/search?session_id=priority-http&q=submit"
                   "&priority=high", timeout=10).read().decode("utf-8"))
        assert search["count"] == 1
        assert search["tasks"][0]["priority"] == "high"
        listed = _post(base + "/v1/agent/chat", {
            "session_id": "priority-http", "message": "list tasks"})
        assert "1 [high] submit report" in listed["reply"]
        with pytest.raises(urllib.error.HTTPError) as invalid:
            _post(base + "/v1/tasks/priority", {
                "session_id": "priority-http", "id": 1,
                "priority": "urgent"})
        assert invalid.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as empty:
            _post(base + "/v1/tasks/priority", {
                "session_id": "priority-http", "id": 1})
        assert empty.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/tasks/priority", {
                "session_id": "priority-http", "id": 99,
                "priority": "high"})
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_priority_sort_views(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    today = date.today()
    try:
        for text, offset, priority, tags in (
                ("low overdue task", -1, "low", "home"),
                ("high overdue task", -2, "high", "work"),
                ("normal soon task", 2, "normal", "home,work"),
                ("high soon task", 1, "high", "work")):
            due = (today + timedelta(days=offset)).isoformat()
            _post(base + "/v1/agent/chat", {
                "session_id": "priority-sort-http",
                "message": f"add task {text} due {due} priority {priority} "
                           f"tags {tags}"})

        def get(path, **params):
            params.update({
                "session_id": "priority-sort-http",
                "today": today.isoformat(),
            })
            return json.loads(urllib.request.urlopen(
                base + path + "?" + urllib.parse.urlencode(params),
                timeout=10).read().decode("utf-8"))

        overdue = get("/v1/tasks/overdue", sort="priority")
        assert overdue["sort"] == "priority"
        assert [task["id"] for task in overdue["overdue_tasks"]] == [2, 1]
        due_soon = get("/v1/tasks/due-soon", days=7, sort="priority")
        assert due_soon["sort"] == "priority"
        assert [task["id"] for task in due_soon["due_soon_tasks"]] == [4, 3]
        search = get("/v1/tasks/search", q="task", sort="priority")
        assert search["sort"] == "priority"
        assert [task["id"] for task in search["tasks"]] == [2, 4, 3, 1]
        task_list = get(
            "/v1/tasks/list", status="open", due="week",
            priority="high", tag="work")
        assert task_list["status"] == "open"
        assert task_list["due"] == "week"
        assert task_list["priority"] == "high"
        assert task_list["tag"] == "work"
        assert task_list["sort"] == "default"
        assert task_list["count"] == 1
        assert [task["id"] for task in task_list["tasks"]] == [4]
        all_priority = get("/v1/tasks/list", status="all", sort="priority")
        assert [task["id"] for task in all_priority["tasks"]] == [2, 4, 3, 1]
        tagged = get("/v1/tasks/list", tag="work")
        assert [task["id"] for task in tagged["tasks"]] == [2, 3, 4]
        chat_list = _post(base + "/v1/agent/chat", {
            "session_id": "priority-sort-http",
            "message": "list tasks due week priority high tag work"})
        assert chat_list["tool_calls"][0] == {
            "name": "list_tasks", "status": "open", "due": "week",
            "priority": "high", "tag": "work", "sort": "default",
            "count": 1}
        assert ([task["id"] for task in chat_list["task_list"]["tasks"]]
                == [task["id"] for task in task_list["tasks"]])
        export_params = urllib.parse.urlencode({
            "session_id": "priority-sort-http",
            "today": today.isoformat(),
            "status": "open", "due": "week", "priority": "high",
            "tag": "work", "sort": "priority", "format": "csv",
        })
        with urllib.request.urlopen(
                base + "/v1/tasks/export?" + export_params,
                timeout=10) as response:
            assert response.headers.get_content_type() == "text/csv"
            assert ".csv" in response.headers.get("Content-Disposition", "")
            csv_body = response.read().decode("utf-8")
        csv_rows = list(csv.reader(io.StringIO(csv_body)))
        assert csv_rows[0][:7] == [
            "id", "status", "priority", "tags", "text", "due", "overdue"]
        assert len(csv_rows) == 2
        assert csv_rows[1][:6] == [
            "4", "open", "high", "work", "high soon task",
            (today + timedelta(days=1)).isoformat()]
        export_params = urllib.parse.urlencode({
            "session_id": "priority-sort-http",
            "today": today.isoformat(),
            "status": "open", "due": "week", "priority": "high",
            "tag": "work", "sort": "priority", "format": "markdown",
        })
        with urllib.request.urlopen(
                base + "/v1/tasks/export?" + export_params,
                timeout=10) as response:
            assert response.headers.get_content_type() == "text/markdown"
            assert ".md" in response.headers.get("Content-Disposition", "")
            markdown_body = response.read().decode("utf-8")
        assert markdown_body.startswith("# Task list - ")
        assert "- Tag: work" in markdown_body
        assert "| 4 | open | high | work | high soon task |" in markdown_body
        brief = _post(base + "/v1/briefing", {
            "session_id": "priority-sort-http", "sort": "priority"})
        assert brief["sort"] == "priority"
        assert [task["id"] for task in brief["tasks"]] == [2, 4, 3, 1]
        assert brief["focus"].startswith("overdue task 2")
        assert "sort=priority" in brief["text"]
        weekly = _post(base + "/v1/weekly-review", {
            "session_id": "priority-sort-http", "days": 7,
            "sort": "priority"})
        assert weekly["sort"] == "priority"
        assert [task["id"] for task in weekly["open_tasks"]] == [2, 4, 3, 1]
        assert "sort=priority" in weekly["text"]
        export_params = urllib.parse.urlencode({
            "session_id": "priority-sort-http", "days": 7,
            "sort": "priority"})
        markdown = urllib.request.urlopen(
            base + "/v1/weekly-review/export?" + export_params,
            timeout=10).read().decode("utf-8")
        assert markdown.startswith("# Weekly review - ")
        assert "sort=priority" in markdown
        assert markdown.index("high overdue task") < \
            markdown.index("low overdue task")
        with pytest.raises(urllib.error.HTTPError) as invalid:
            get("/v1/tasks/overdue", sort="bogus")
        assert invalid.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid_brief:
            _post(base + "/v1/briefing", {
                "session_id": "priority-sort-http", "sort": "bogus"})
        assert invalid_brief.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid_list:
            get("/v1/tasks/list", status="blocked")
        assert invalid_list.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as invalid_export:
            get("/v1/tasks/export", format="json")
        assert invalid_export.value.code == 400
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_reschedule_task(tmp_path):
    overdue = (date.today() - timedelta(days=1)).isoformat()
    future = (date.today() + timedelta(days=3)).isoformat()
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(base + "/v1/agent/chat", {
            "session_id": "resched",
            "message": f"add task submit report due {overdue}"})
        before = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/overdue?session_id=resched",
            timeout=10).read().decode("utf-8"))
        assert before["overdue_task_count"] == 1
        result = _post(base + "/v1/tasks/reschedule", {
            "session_id": "resched", "id": 1, "due": future})
        assert result["task"]["due"] == future
        after = json.loads(urllib.request.urlopen(
            base + "/v1/tasks/overdue?session_id=resched",
            timeout=10).read().decode("utf-8"))
        assert after["overdue_task_count"] == 0
        with pytest.raises(urllib.error.HTTPError) as invalid:
            _post(base + "/v1/tasks/reschedule", {
                "session_id": "resched", "id": 1, "due": "next week"})
        assert invalid.value.code == 400
        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/tasks/reschedule", {
                "session_id": "resched", "id": 99, "due": future})
        assert missing.value.code == 404
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_delete_task_requires_confirmation(tmp_path):
    old = dict(chat_server._state)
    chat_server._state.update({
        "model": None, "tok": None, "docs": None, "adapter": "",
        "name": "tiny-test", "params": 1, "config": {}, "step": 0,
        "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
        "safety": None, "wiki_lang": "en",
        "session_dir": str(tmp_path / "sessions"), "sessions": {},
        "generator": None, "allow_general_chat": False, "skills": None,
    })
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        _post(base + "/v1/agent/chat", {
            "session_id": "delete-http", "message": "add task buy milk"})
        with pytest.raises(urllib.error.HTTPError) as unconfirmed:
            _post(base + "/v1/tasks/delete", {
                "session_id": "delete-http", "id": 1})
        assert unconfirmed.value.code == 400
        deleted = _post(base + "/v1/tasks/delete", {
            "session_id": "delete-http", "id": 1, "confirm": True})
        assert deleted["deleted"]["text"] == "buy milk"
        with pytest.raises(urllib.error.HTTPError) as missing:
            _post(base + "/v1/tasks/delete", {
                "session_id": "delete-http", "id": 1, "confirm": True})
        assert missing.value.code == 404
        chat = _post(base + "/v1/agent/chat", {
            "session_id": "delete-http", "message": "list tasks"})
        assert "buy milk" not in chat["reply"]
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)


def test_http_chat_weekly_review(tmp_path):
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
            "session_id": "week-chat", "message": "add task buy milk"})
        _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "add task write report"})
        yesterday = (date.today() - timedelta(days=1)).isoformat()
        two_days_ago = (date.today() - timedelta(days=2)).isoformat()
        _post(base + "/v1/agent/chat", {
            "session_id": "week-chat",
            "message": f"add task low priority task due {yesterday} "
                       "priority low"})
        _post(base + "/v1/agent/chat", {
            "session_id": "week-chat",
            "message": f"add task high priority task due {two_days_ago} "
                       "priority high"})
        _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "complete task 1"})
        result = _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "weekly review"})
        assert result["tool_calls"][0] == {
            "name": "weekly_review", "days": 7, "sort": "default"}
        assert result["review"]["task_counts"] == {
            "open": 3, "completed_recent": 1, "completed_total": 1}
        assert result["review"]["pending_gap_count"] == 1
        assert result["review"]["recent_source_count"] == 1
        assert "Weekly review" in result["reply"]
        assert "write report" in result["reply"]
        assert "buy milk" in result["reply"]
        assert result["sources"][0]["source"] == "notes/launch.txt"
        assert result["weights_changed"] is False
        priority_review = _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "weekly review priority"})
        assert priority_review["tool_calls"][0] == {
            "name": "weekly_review", "days": 7, "sort": "priority"}
        assert priority_review["review"]["sort"] == "priority"
        assert [task["id"] for task in
                priority_review["review"]["open_tasks"]] == [4, 2, 3]
        assert "sort=priority" in priority_review["reply"]
        priority_brief = _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "brief me priority"})
        assert priority_brief["tool_calls"][0] == {
            "name": "briefing", "sort": "priority"}
        assert priority_brief["brief"]["sort"] == "priority"
        assert [task["id"] for task in
                priority_brief["brief"]["tasks"]] == [4, 2, 3]
        assert "sort=priority" in priority_brief["reply"]
        chat_search = _post(base + "/v1/agent/chat", {
            "session_id": "week-chat",
            "message": "search tasks task sort priority"})
        assert chat_search["tool_calls"][0]["sort"] == "priority"
        assert [task["id"] for task in
                chat_search["task_search"]["tasks"]] == [4, 3]
        chat_overdue = _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "overdue tasks priority"})
        assert chat_overdue["tool_calls"][0] == {
            "name": "overdue", "sort": "priority", "count": 2}
        assert [task["id"] for task in
                chat_overdue["overdue"]["overdue_tasks"]] == [4, 3]
        turkish = _post(base + "/v1/agent/chat", {
            "session_id": "week-chat", "message": "haftalık özet 14 gün"})
        assert turkish["tool_calls"][0]["days"] == 14
        assert "last 14 days" in turkish["reply"]
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
            "session_id": "brief-1", "message": "add task buy milk tags home"})
        _post(base + "/v1/agent/chat", {
            "session_id": "brief-1",
            "message": "add task write report tags work,home"})
        brief = _post(base + "/v1/briefing", {"session_id": "brief-1"})
        assert brief["open_task_count"] == 2
        assert "buy milk" in brief["text"]
        assert "write report" in brief["text"]
        filtered = _post(base + "/v1/briefing", {
            "session_id": "brief-1", "tag": "work"})
        assert filtered["tag"] == "work"
        assert filtered["open_task_count"] == 1
        assert [task["id"] for task in filtered["tasks"]] == [2]
        assert filtered["tags"] == [
            {"tag": "home", "open": 1, "done": 0, "total": 1},
            {"tag": "work", "open": 1, "done": 0, "total": 1},
        ]
        assert "| tag=work" in filtered["text"]
        assert "Tags for #work: 2 distinct" in filtered["text"]
        assert "write report" in filtered["text"]
        assert "buy milk" not in filtered["text"]
        fetched = json.loads(urllib.request.urlopen(
            base + "/v1/briefing?session_id=brief-1&tag=work",
            timeout=10).read().decode("utf-8"))
        assert fetched["tag"] == "work"
        assert fetched["open_task_count"] == 1
        assert [task["id"] for task in fetched["tasks"]] == [2]
        with pytest.raises(urllib.error.HTTPError) as invalid_tag:
            urllib.request.urlopen(
                base + "/v1/briefing?session_id=brief-1&tag=bad%20tag%21",
                timeout=10)
        assert invalid_tag.value.code == 400
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
        assert after["open_task_count"] == 5
    finally:
        httpd.shutdown()
        httpd.server_close()
        chat_server._state.clear()
        chat_server._state.update(old)
        docs.close()
