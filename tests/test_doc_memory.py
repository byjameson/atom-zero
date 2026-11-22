import json

import pytest

from atomzero.doc_memory import (
    DocMemory, SourceNotFound, _chunks, _identifier_terms, _query_terms,
    ann_topk, best_snippet, diverse_hits, embed_text, extractive_summary,
    extractive_summary_units, main, retrieval_confidence)


def test_doc_memory_indexes_and_searches_with_source(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    added = mem.add_text(
        "law.txt", "The capital of France is Paris. Rent contracts need a date.",
        title="Notes")
    assert added >= 1
    hits = mem.search("capital France", k=3)
    assert hits
    assert "Paris" in hits[0]["chunk"]
    assert hits[0]["source"] == "law.txt"
    assert mem.stats()["sources"] == 1
    # Re-ingesting the same source replaces its chunks.
    mem.add_text("law.txt", "Only one fact.", title="Notes")
    assert len(mem.search("fact", k=3)) == 1
    mem.close()


def test_best_snippet_selects_query_sentence_and_keeps_teacher_value():
    text = (
        "Berlin was first documented in the 13th century. "
        "The city has a population of 3.7 million. "
        "It is the capital of Germany.")
    snippet = best_snippet("what is the population of Berlin?", text)
    assert "population of 3.7 million" in snippet
    teacher = ("Topic: Teacher code\n\nThe teacher code is T-9000.")
    assert "T-9000" in best_snippet("what is the teacher code?", teacher)


def test_identifier_terms_and_exact_code_ranking(tmp_path):
    assert "l-7781" in _identifier_terms("Use L-7781 here")
    assert "l-7781" in _query_terms("What is L-7781?")
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("launch.txt", "The launch code is L-7781.", title="Launch")
    mem.add_text("warranty.txt", "The warranty code is W-7781.",
                 title="Warranty")
    launch = mem.search("L-7781", k=2)
    warranty = mem.search("W-7781", k=2)
    assert launch[0]["source"] == "launch.txt"
    assert warranty[0]["source"] == "warranty.txt"
    assert "L-7781" in best_snippet("L-7781", launch[0]["chunk"])
    mem.close()


def test_chunks_split_long_text_on_word_boundaries():
    chunks = _chunks(("alpha " * 300).strip())
    assert len(chunks) > 1
    assert all(len(chunk) <= 900 for chunk in chunks)
    assert all(chunk.startswith("alpha") for chunk in chunks)


def test_search_ignores_common_question_words(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    mem.add_text("noise.txt", "is " * 300)
    mem.add_text("ankara.txt", "Ankara is the capital of Turkey.",
                 title="Ankara")
    hits = mem.search("What is Ankara?", k=2)
    assert hits
    assert "capital" in hits[0]["chunk"]
    mem.close()


def test_hybrid_rejects_unrelated_query(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("law.txt", "The capital of France is Paris.",
                 title="Notes")
    assert mem.search("nothing here", k=2) == []
    mem.close()


def test_hashed_hybrid_stays_silent_without_an_fts_match(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("launch.txt", "The launch code is ORANGE-42.",
                 title="Launch")
    mem.add_text("support.txt", "The support email is help@example.test.",
                 title="Support")
    assert mem.search("what is the population of Paris?", k=2) == []
    mem.close()


def test_hybrid_vector_cache_is_pruned_on_replace(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("warranty.txt", "The warranty code is W-7781.")
    mem.search("warranty code", k=2)
    assert mem.con.execute(
        "SELECT COUNT(*) FROM doc_vectors").fetchone()[0] == 1
    mem.add_text("warranty.txt", "The warranty code is W-9900.")
    assert mem.search("7781", k=2) == []
    assert mem.con.execute(
        "SELECT COUNT(*) FROM doc_vectors").fetchone()[0] == 1
    assert mem.search("warranty code", k=2)[0]["source"] == "warranty.txt"
    mem.close()


def test_prewarmed_index_matches_linear_scan(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("printer.txt",
                 "Replace the cyan cartridge when print quality fades.",
                 title="Printer")
    mem.add_text("law.txt", "The capital of France is Paris.",
                 title="Notes")
    mem.add_text("weather.txt", "Tomorrow will be sunny with light wind.",
                 title="Weather")
    stats = mem.prewarm_vectors()
    assert stats["built"] is True
    assert stats["chunks"] == 3
    assert stats["dim"] == 512
    assert stats["backend"] == "numpy"
    indexed, indexed_scores = mem._embedding_candidates(
        "Why are my documents printing pale?", None, 3)
    scanned, scan_scores = mem._embedding_candidates_scan(
        "Why are my documents printing pale?", None, 3)
    assert [hit["source"] for hit in indexed] == [
        hit["source"] for hit in scanned]
    for hit in indexed:
        assert indexed_scores[hit["_rowid"]] == pytest.approx(
            scan_scores[hit["_rowid"]], abs=1e-6)
    mem.close()


def test_ann_index_matches_exact_when_probing_every_bucket(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    for index in range(20):
        mem.add_text(f"policy-{index}.txt",
                     f"Policy {index}: approval code CODE-{index}.",
                     title=f"Policy {index}", replace=False)
    mem.vector_index_mode = "ann"
    mem.ann_min_chunks = 4
    stats = mem.prewarm_vectors()
    assert stats["mode"] == "ann"
    assert stats["ann"] is True
    assert stats["exact_fallback"] is True
    assert stats["nlist"] >= 2
    index = mem._vector_index
    query = embed_text("Policy 3 approval code")
    exact_positions, _ = mem._exact_candidates(index, query, 5)
    ann_positions, _ = ann_topk(
        index["matrix"], index["ann"], query, 5,
        probe=index["ann"]["nlist"])
    assert [int(value) for value in ann_positions] == [
        int(value) for value in exact_positions]
    mem.close()


def test_ann_index_falls_back_to_exact_on_low_confidence(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    for index in range(20):
        mem.add_text(f"policy-{index}.txt",
                     f"Policy {index}: approval code CODE-{index}.",
                     title=f"Policy {index}", replace=False)
    mem.vector_index_mode = "ann"
    mem.ann_min_chunks = 4
    mem.ann_probe = 1
    mem.prewarm_vectors()
    mem._embedding_candidates("zzzz qqqq", None, 3)
    stats = mem.vector_index_stats()
    assert stats["mode"] == "ann"
    assert stats["fallbacks"] >= 1
    mem.close()


def test_vector_index_invalidates_on_add_and_forget(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("launch.txt", "The launch code is ORANGE-42.")
    mem.prewarm_vectors()
    assert mem.vector_index_stats()["built"] is True
    mem.add_text("support.txt", "The support email is help@example.test.")
    assert mem.vector_index_stats()["built"] is False
    mem.search("support email", k=2)
    assert mem.vector_index_stats()["built"] is True
    mem.forget("support.txt")
    assert mem.vector_index_stats()["built"] is False
    mem.close()


def test_teacher_metadata_is_returned(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    mem.add_text(
        "teacher://chatgpt-web/abc", "Topic: Ankara\n\nAnkara is in Turkey.",
        title="Teacher note: Ankara",
        meta={"kind": "teacher", "source_url": "https://example.com/chat",
              "teacher": "chatgpt-web", "model": "gpt-5", "verified": False})
    hit = mem.search("Ankara")[0]
    assert hit["kind"] == "teacher"
    assert hit["source_url"] == "https://example.com/chat"
    assert hit["teacher"] == "chatgpt-web"
    assert hit["model"] == "gpt-5"
    assert hit["verified"] is False
    source = mem.sources()[0]
    assert source["kind"] == "teacher"
    assert source["verified"] is False
    mem.close()


def test_verify_teacher_note_records_reviewer_and_timestamp(tmp_path):
    try:
        mem = DocMemory(tmp_path / "docs.db")
    except Exception as exc:
        pytest.skip(f"sqlite fts5 unavailable: {exc}")
    source = "teacher://chatgpt-web/verify"
    mem.add_text(
        source, "Topic: Ankara\n\nAnkara is in Turkey.",
        title="Teacher note: Ankara",
        meta={"kind": "teacher", "source_url": "https://example.com/chat",
              "teacher": "chatgpt-web", "model": "gpt-5", "verified": False})
    with pytest.raises(SourceNotFound):
        mem.verify_source("teacher://missing", "reviewer")
    with pytest.raises(ValueError):
        mem.verify_source(source, "")
    mem.add_text("notes.txt", "A normal document.", title="Notes")
    with pytest.raises(ValueError):
        mem.verify_source("notes.txt", "reviewer")
    note = mem.verify_source(source, "reviewer-1")
    assert note["verified"] is True
    assert note["reviewer"] == "reviewer-1"
    assert note["verified_at"]
    hit = mem.search("Ankara")[0]
    assert hit["verified"] is True
    assert hit["reviewer"] == "reviewer-1"
    listed = next(item for item in mem.sources()
                  if item["source"] == source)
    assert listed["verified"] is True
    assert listed["verified_at"] == note["verified_at"]
    mem.close()


def test_cli_accepts_db_after_subcommand(tmp_path, capsys):
    note = tmp_path / "note.txt"
    note.write_text("The project codename is Blue Lantern.", encoding="utf-8")
    db = tmp_path / "docs.db"
    main(["add", str(note), "--db", str(db), "--title", "Demo"])
    added = json.loads(capsys.readouterr().out)
    assert added["added"] == 1
    main(["--db", str(db), "search", "codename"])
    hits = json.loads(capsys.readouterr().out)
    assert "Blue Lantern" in hits[0]["chunk"]


def test_search_filters_by_source_prefix_and_kind(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("launch.txt", "The launch code is L-7781.", title="Launch")
    mem.add_text("warranty.txt", "The warranty code is W-9000.",
                 title="Warranty")
    mem.add_text("teacher://chatgpt/1", "The teacher code is T-42.",
                 title="Teacher", meta={"kind": "teacher"})
    hits = mem.search("code", source="launch.txt")
    assert [hit["source"] for hit in hits] == ["launch.txt"]
    assert mem.search("code", source="missing.txt") == []
    assert [hit["source"] for hit in mem.search(
        "code", source_prefix="launch")] == ["launch.txt"]
    assert [hit["source"] for hit in mem.search(
        "code", kind="teacher")] == ["teacher://chatgpt/1"]
    with pytest.raises(ValueError):
        mem.search("code", source="bad\nsource")
    mem.close()


def test_add_file_result_skips_unchanged_content(tmp_path):
    note = tmp_path / "note.txt"
    note.write_text("The project codename is Blue Lantern.", encoding="utf-8")
    mem = DocMemory(tmp_path / "docs.db")
    first = mem.add_file_result(note)
    assert first["chunks"] == 1
    assert first["skipped"] is False
    assert len(first["content_sha256"]) == 64
    assert len(first["file_sha256"]) == 64
    second = mem.add_file_result(note, skip_if_unchanged=True)
    assert second["chunks"] == 0
    assert second["skipped"] is True
    assert mem.stats()["sources"] == 1
    note.write_text("The project codename is Red Lantern.", encoding="utf-8")
    third = mem.add_file_result(note, skip_if_unchanged=True)
    assert third["skipped"] is False
    assert third["chunks"] == 1
    assert mem.search("Red Lantern")[0]["source"] == str(note)
    mem.close()


def test_add_directory_ingests_supported_files_and_skips_unchanged(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "launch.txt").write_text(
        "The launch code is L-7781.", encoding="utf-8")
    (source / "warranty.md").write_text(
        "# Warranty\n\nThe warranty period is 24 months.", encoding="utf-8")
    (source / "ignore.bin").write_bytes(b"\x00\x01")
    mem = DocMemory(tmp_path / "docs.db")
    first = mem.add_directory(source, skip_if_unchanged=True)
    assert first["files"] == 2
    assert first["chunks"] == 2
    assert first["failed"] == 0
    assert first["skipped"] == 0
    second = mem.add_directory(source, skip_if_unchanged=True)
    assert second["files"] == 2
    assert second["chunks"] == 0
    assert second["skipped"] == 2
    assert mem.stats()["sources"] == 2
    mem.close()


def test_cli_directory_ingest_and_source_filter(tmp_path, capsys):
    source = tmp_path / "corpus"
    source.mkdir()
    note = source / "launch.txt"
    note.write_text("The launch code is L-7781.", encoding="utf-8")
    db = tmp_path / "docs.db"
    main(["add", str(source), "--db", str(db)])
    added = json.loads(capsys.readouterr().out)
    assert added["files"] == 1
    assert added["added"] == 1
    main(["search", "launch code", "--db", str(db),
          "--source", str(note)])
    hits = json.loads(capsys.readouterr().out)
    assert len(hits) == 1
    assert hits[0]["source"] == str(note)
    main(["stats", "--db", str(db)])
    stats = json.loads(capsys.readouterr().out)
    assert stats["sources"] == 1


def test_diverse_hits_and_retrieval_confidence():
    hits = [
        {"source": "a", "chunk": "The launch code is L-7781.",
         "title": "Launch", "rerank_score": 0.9},
        {"source": "a", "chunk": "The launch code is L-7781.",
         "title": "Launch", "rerank_score": 0.8},
        {"source": "a", "chunk": "Another launch note.",
         "title": "Launch", "rerank_score": 0.7},
        {"source": "b", "chunk": "The warranty code is W-9000.",
         "title": "Warranty", "rerank_score": 0.5},
    ]
    selected = diverse_hits(hits, limit=3, per_source=2)
    assert [hit["source"] for hit in selected] == ["a", "a", "b"]
    confidence = retrieval_confidence("what is the launch code?", selected)
    assert 0.0 <= confidence["score"] <= 1.0
    assert confidence["label"] in {"low", "medium", "high"}
    assert retrieval_confidence("launch", [])["label"] == "none"


def test_directory_dedupe_skips_same_content_under_another_name(tmp_path):
    source = tmp_path / "corpus"
    source.mkdir()
    (source / "a.txt").write_text(
        "The launch code is L-7781.", encoding="utf-8")
    (source / "copy.txt").write_text(
        "The launch code is L-7781.", encoding="utf-8")
    mem = DocMemory(tmp_path / "docs.db")
    result = mem.add_directory(source, dedupe_across_sources=True)
    assert result["files"] == 2
    assert result["chunks"] == 1
    assert result["duplicates"] == 1
    duplicate = next(item for item in result["results"]
                     if item.get("duplicate_of"))
    assert duplicate["duplicate_of"][0]["source"] == str(source / "a.txt")
    assert mem.stats()["sources"] == 1
    assert mem.find_by_content_sha(duplicate["content_sha256"])[0][
        "source"] == str(source / "a.txt")
    with pytest.raises(ValueError):
        mem.find_by_content_sha("not-a-sha")
    mem.close()


def test_cli_answer_returns_evidence_and_no_source(tmp_path, capsys):
    note = tmp_path / "launch.txt"
    note.write_text("The launch code is L-7781.", encoding="utf-8")
    db = tmp_path / "docs.db"
    main(["add", str(note), "--db", str(db)])
    capsys.readouterr()
    main(["answer", "what is the launch code?", "--db", str(db)])
    answer = json.loads(capsys.readouterr().out)
    assert "L-7781" in answer["text"]
    assert answer["evidence"][0]["source"] == str(note)
    assert answer["confidence"]["label"] in {"low", "medium", "high"}
    main(["answer", "population of Paris", "--db", str(db)])
    missing = json.loads(capsys.readouterr().out)
    assert missing["sources"] == []
    assert missing["confidence"]["label"] == "none"


def test_sync_directory_is_portable_and_prunes_removed_files(tmp_path):
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text(
        "The launch code is L-7781.", encoding="utf-8")
    (root / "sub" / "b.md").write_text(
        "# Warranty\n\nThe warranty period is 24 months.", encoding="utf-8")
    mem = DocMemory(tmp_path / "docs.db")
    first = mem.sync_directory(root, prefix="notes")
    assert first["files"] == 2
    assert first["chunks"] == 2
    assert first["pruned"] == []
    indexed = {item["source"] for item in mem.sources()}
    assert indexed == {"notes/a.txt", "notes/sub/b.md"}
    second = mem.sync_directory(root, prefix="notes")
    assert second["chunks"] == 0
    assert second["skipped"] == 2
    (root / "sub" / "b.md").unlink()
    third = mem.sync_directory(root, prefix="notes")
    assert third["chunks"] == 0
    assert [item["source"] for item in third["pruned"]] == [
        "notes/sub/b.md"]
    assert {item["source"] for item in mem.sources()} == {"notes/a.txt"}
    mem.close()


def test_directory_status_reports_changes_without_writing(tmp_path):
    root = tmp_path / "notes"
    (root / "sub").mkdir(parents=True)
    (root / "a.txt").write_text("The launch code is L-7781.",
                                encoding="utf-8")
    (root / "sub" / "b.md").write_text(
        "The warranty period is 24 months.", encoding="utf-8")
    mem = DocMemory(tmp_path / "docs.db")
    mem.sync_directory(root, prefix="notes")
    clean = mem.directory_status(root, prefix="notes")
    assert clean["added"] == 0
    assert clean["changed"] == 0
    assert clean["unchanged"] == 2
    assert clean["missing"] == []
    (root / "a.txt").write_text("The launch code is L-9900.",
                                encoding="utf-8")
    (root / "sub" / "b.md").unlink()
    (root / "c.txt").write_text("The visitor badge is VB-42.",
                                encoding="utf-8")
    status = mem.directory_status(root, prefix="notes")
    assert status["added"] == 1
    assert status["changed"] == 1
    assert status["missing"] == ["notes/sub/b.md"]
    assert {item["status"] for item in status["details"]} == {
        "added", "changed"}
    assert mem.stats()["sources"] == 2
    mem.close()


def test_integrity_reports_clean_index_and_orphans(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("a.txt", "The launch code is L-7781.")
    clean = mem.integrity()
    assert clean["ok"] is True
    assert clean["issues"] == []
    assert clean["chunks"] == 1
    assert clean["sources"] == 1
    mem.con.execute("DELETE FROM doc_meta WHERE source = 'a.txt'")
    mem.con.execute(
        "INSERT INTO doc_meta(source, kind, source_url, teacher, model, "
        "verified, created, reviewer, verified_at) "
        "VALUES ('ghost.txt', 'document', '', '', '', 1, '', '', '')")
    mem.con.execute(
        "INSERT INTO doc_vectors(doc_rowid, model, dim, vec) "
        "VALUES (9999, 'x', 1, ?)", (b"\x00",))
    mem.con.execute(
        "INSERT INTO doc_chunk_meta(doc_rowid, page, section) "
        "VALUES (9999, NULL, '')")
    mem.con.commit()
    broken = mem.integrity()
    assert broken["ok"] is False
    assert broken["missing_meta"] == 1
    assert broken["orphan_meta"] == 1
    assert broken["orphan_vectors"] == 1
    assert broken["orphan_chunk_meta"] == 1
    assert len(broken["issues"]) >= 4
    mem.close()


def test_extractive_summary_selects_sentences_in_order():
    chunks = [
        "Launch policy. The launch code is L-7781. "
        "The launch window is 06:30. Warranty lasts 24 months. "
        "Visitors need a badge."
    ]
    result = extractive_summary(chunks, max_sentences=3)
    assert result["method"] == "extractive"
    assert result["candidates"] == 5
    assert len(result["sentences"]) == 3
    assert "L-7781" in result["text"]
    assert result["text"].index("L-7781") < result["text"].index("06:30")
    assert extractive_summary([], max_sentences=3)["text"] == ""


def test_summarize_source_and_cli_summary(tmp_path, capsys):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_blocks(
        "manual.pdf",
        [{"text": "The launch code is L-7781. The launch window is 06:30.",
          "page": 4, "section": "Launch"}],
        title="Manual")
    result = mem.summarize_source("manual.pdf", max_sentences=2)
    assert result["source"] == "manual.pdf"
    assert result["title"] == "Manual"
    assert result["chunks"] == 1
    assert result["citations"] == [{"page": 4, "section": "Launch"}]
    assert "L-7781" in result["text"]
    with pytest.raises(SourceNotFound):
        mem.summarize_source("missing.pdf")
    mem.close()

    note = tmp_path / "note.txt"
    note.write_text("The launch code is L-7781. The launch window is 06:30.",
                    encoding="utf-8")
    db = tmp_path / "cli.db"
    main(["add", str(note), "--db", str(db)])
    capsys.readouterr()
    main(["summary", str(note), "--db", str(db), "--sentences", "1"])
    output = json.loads(capsys.readouterr().out)
    assert "launch" in output["text"].lower()
    assert output["method"] == "extractive"


def test_export_import_roundtrip_preserves_provenance(tmp_path):
    mem = DocMemory(tmp_path / "source.db")
    mem.add_blocks(
        "manual.pdf",
        [{"text": "The launch code is L-7781.", "page": 4,
          "section": "Launch"}],
        title="Manual")
    mem.add_text(
        "teacher://chatgpt/1", "The warranty period is 36 months.",
        title="Teacher",
        meta={"kind": "teacher", "verified": False,
              "source_url": "https://example.com/chat"})
    backup = tmp_path / "backup.jsonl"
    exported = mem.export_jsonl(backup)
    assert exported["sources"] == 2
    assert exported["chunks"] == 2
    mem.close()

    restored = DocMemory(tmp_path / "restored.db")
    imported = restored.import_jsonl(backup)
    assert imported["sources"] == 2
    assert imported["chunks"] == 2
    assert imported["failed"] == 0
    listed = {item["source"]: item for item in restored.sources()}
    assert listed["manual.pdf"]["page_start"] == 4
    assert listed["teacher://chatgpt/1"]["verified"] is False
    assert listed["teacher://chatgpt/1"]["source_url"] == (
        "https://example.com/chat")
    evidence = restored.source_evidence("manual.pdf", page=4)
    assert "L-7781" in evidence["data"][0]["chunk"]
    assert restored.search("warranty period")[0]["kind"] == "teacher"
    restored.close()


def test_import_jsonl_reports_bad_lines(tmp_path):
    mem = DocMemory(tmp_path / "docs.db")
    backup = tmp_path / "partial.jsonl"
    backup.write_text(
        "not json\n"
        '{"source": "ok.txt", "blocks": [{"text": "hello world"}]}\n',
        encoding="utf-8")
    result = mem.import_jsonl(backup)
    assert result["sources"] == 1
    assert result["failed"] == 1
    assert result["errors"][0]["line"] == 1
    assert mem.search("hello")[0]["source"] == "ok.txt"
    mem.close()


def test_cli_export_import_roundtrip(tmp_path, capsys):
    note = tmp_path / "note.txt"
    note.write_text("The launch code is L-7781.", encoding="utf-8")
    source = tmp_path / "source.db"
    target = tmp_path / "target.db"
    backup = tmp_path / "index.jsonl"
    main(["add", str(note), "--db", str(source)])
    capsys.readouterr()
    main(["export", "--out", str(backup), "--db", str(source)])
    exported = json.loads(capsys.readouterr().out)
    assert exported["sources"] == 1
    main(["import", str(backup), "--db", str(target)])
    imported = json.loads(capsys.readouterr().out)
    assert imported["sources"] == 1
    assert imported["failed"] == 0


def test_extractive_summary_units_preserves_locators():
    units = [
        {"text": "Launch code is L-7781. Visitors need a badge.",
         "source": "notes/a.txt", "page": 1, "section": "Launch"},
        {"text": "The warranty period is 24 months.",
         "source": "notes/b.md", "page": None, "section": "Warranty"},
    ]
    result = extractive_summary_units(units, max_sentences=3)
    assert result["method"] == "extractive"
    assert result["candidates"] == 3
    assert {item["source"] for item in result["sentences"]} == {
        "notes/a.txt", "notes/b.md"}
    assert {"source": "notes/b.md", "page": None, "section": "Warranty"} in (
        result["citations"])
    assert "L-7781" in result["text"]
    assert "24 months" in result["text"]


def test_summarize_collection_and_cli(tmp_path, capsys):
    mem = DocMemory(tmp_path / "docs.db")
    mem.add_text("notes/a.txt", "The launch code is L-7781.",
                 title="Launch")
    mem.add_text("notes/b.md", "The warranty period is 24 months.",
                 title="Warranty")
    result = mem.summarize_collection("notes/", max_sentences=4)
    assert result["sources"] == 2
    assert result["chunks"] == 2
    assert "L-7781" in result["text"]
    assert "24 months" in result["text"]
    assert {item["source"] for item in result["sentences"]} == {
        "notes/a.txt", "notes/b.md"}
    with pytest.raises(ValueError):
        mem.summarize_collection()
    with pytest.raises(SourceNotFound):
        mem.summarize_collection(source_prefix="missing/")
    mem.close()

    db = tmp_path / "cli.db"
    mem = DocMemory(db)
    mem.add_text("notes/a.txt", "The launch code is L-7781.")
    mem.close()
    main(["summary-collection", "--prefix", "notes/", "--sentences", "2",
          "--db", str(db)])
    output = json.loads(capsys.readouterr().out)
    assert "L-7781" in output["text"]
    assert output["source_prefix"] == "notes/"
