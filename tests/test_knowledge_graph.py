import json

from atomzero.doc_memory import DocMemory, main
from atomzero.knowledge_graph import (
    build_graph, entity_sources, extract_entities)


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
