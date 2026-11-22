import json

from atomzero.doc_memory import DocMemory, main
from atomzero.knowledge_graph import (
    build_graph, entity_sources, extract_entities, graph_path)


def test_extract_entities_finds_places_codes_and_years():
    text = ("Ankara is the capital of Turkey. Istanbul is in Turkey. "
            "The launch code is L-7781. In 2024, New York hosted it.")
    entities = extract_entities(text)
    for expected in ("Ankara", "Turkey", "Istanbul", "L-7781",
                     "New York", "2024"):
        assert expected in entities
    assert "The" not in entities


def test_graph_and_entity_sources(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text(
        "geo.txt",
        "Ankara is the capital of Turkey. Istanbul is in Turkey. "
        "Ankara is in Turkey.",
        title="Geography")
    docs.add_text("launch.txt", "The launch code is L-7781.",
                  title="Launch")
    graph = build_graph(docs)
    names = {node["name"] for node in graph["nodes"]}
    assert {"Ankara", "Istanbul", "Turkey", "L-7781"} <= names
    edges = {(edge["source"], edge["target"]): edge
             for edge in graph["edges"]}
    assert edges[("Ankara", "Turkey")]["weight"] >= 1
    assert edges[("Ankara", "Turkey")]["sources"] == 1

    data = entity_sources(docs, "Ankara")
    assert data["entity"] == "Ankara"
    assert data["matched_chunks"] == 1
    assert data["sources"][0]["source"] == "geo.txt"
    assert "Ankara" in data["sources"][0]["snippet"]
    assert any(connection["target"] == "Turkey"
               for connection in data["connections"])
    docs.close()


def test_graph_path_multihop_and_missing(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("chain1.txt", "Ankara is in Turkey.", title="Chain 1")
    docs.add_text("chain2.txt",
                  "Turkey uses the launch code L-7781.", title="Chain 2")
    result = graph_path(docs, "Ankara", "L-7781", max_depth=3)
    assert result["found"] is True
    assert result["path"] == ["Ankara", "Turkey", "L-7781"]
    assert result["hops"] == 2
    assert len(result["edges"]) == 2
    shallow = graph_path(docs, "Ankara", "L-7781", max_depth=1)
    assert shallow["found"] is False
    missing = graph_path(docs, "Ankara", "Missing")
    assert missing["found"] is False
    assert "available_entities" in missing
    same = graph_path(docs, "Ankara", "Ankara")
    assert same["found"] is True and same["hops"] == 0
    docs.close()


def test_cli_graph_and_entity(tmp_path, capsys):
    note = tmp_path / "geo.txt"
    note.write_text("Ankara is the capital of Turkey.", encoding="utf-8")
    db = tmp_path / "docs.db"
    main(["add", str(note), "--db", str(db)])
    capsys.readouterr()
    main(["graph", "--db", str(db)])
    graph = json.loads(capsys.readouterr().out)
    assert any(node["name"] == "Ankara" for node in graph["nodes"])
    main(["entity", "Ankara", "--db", str(db)])
    entity = json.loads(capsys.readouterr().out)
    assert entity["entity"] == "Ankara"
    assert entity["sources"][0]["source"] == str(note)
    main(["path", "Ankara", "Turkey", "--db", str(db)])
    path = json.loads(capsys.readouterr().out)
    assert path["found"] is True
    assert path["hops"] == 1
