import json

from atomzero.doc_memory import DocMemory, main
from atomzero.relations import extract_relations, relation_path


def _docs(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("geo.txt",
                  "Ankara is the capital of Turkey. "
                  "Ankara is in Turkey.", title="Geography")
    docs.add_text("company.txt",
                  "Acme was founded by Jane Doe. "
                  "Jane Doe is the CEO of Acme.", title="Company")
    docs.add_text("life.txt", "Jane Doe was born in Istanbul.", title="Life")
    docs.add_text("tr.txt",
                  "Ankara, Türkiye'nin başkentidir. "
                  "Acme, Jane Doe tarafından kuruldu. "
                  "Jane Doe, Acme'nin CEO'su. "
                  "Jane Doe Ankara'da doğdu.", title="Türkçe")
    docs.add_text("control.txt",
                  "Ankara and Turkey are mentioned together.",
                  title="Control")
    return docs


def test_extract_relations_finds_typed_triples(tmp_path):
    docs = _docs(tmp_path)
    result = extract_relations(docs)
    names = {item["relation"] for item in result["relations"]}
    assert {"capital_of", "located_in", "founded_by", "ceo_of",
            "born_in"} <= names
    assert result["weights_changed"] is False
    for item in result["relations"]:
        assert item["subject"] != item["object"]
        assert item["source"]
        assert item["text"]
    filtered = extract_relations(docs, entity="Ankara")
    assert filtered["count"] >= 2
    for item in filtered["relations"]:
        text = (item["subject"] + " " + item["object"]).casefold()
        assert "ankara" in text
    capitals = extract_relations(docs, relation="capital_of")
    assert capitals["count"] >= 1
    assert all(item["relation"] == "capital_of"
               for item in capitals["relations"])
    docs.close()


def test_relation_path_multihop_and_limits(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("path.txt",
                  "Jane Doe founded Acme. Acme is in Boston.",
                  title="Path")
    docs.add_text("geo.txt", "Ankara is the capital of Turkey.",
                  title="Geography")
    direct = relation_path(docs, "Ankara", "Turkey", max_depth=1)
    assert direct["found"] is True
    assert direct["hops"] == 1
    assert direct["path"] == ["Ankara", "Turkey"]
    multihop = relation_path(docs, "Jane Doe", "Boston", max_depth=2)
    assert multihop["found"] is True
    assert multihop["hops"] == 2
    assert "Acme" in multihop["path"]
    assert multihop["edges"]
    assert multihop["sources"]
    shallow = relation_path(docs, "Jane Doe", "Boston", max_depth=1)
    assert shallow["found"] is False
    missing = relation_path(docs, "Atlantis", "Boston")
    assert missing["found"] is False
    assert "source entity not found" in missing["reason"]
    same = relation_path(docs, "Ankara", "Ankara")
    assert same["found"] is True and same["hops"] == 0
    docs.close()


def test_relations_cli(tmp_path, capsys):
    source = tmp_path / "geo.txt"
    source.write_text("Ankara is the capital of Turkey.", encoding="utf-8")
    db = tmp_path / "cli.db"
    main(["add", str(source), "--db", str(db), "--title", "Geography"])
    capsys.readouterr()
    main(["relations", "--db", str(db)])
    output = json.loads(capsys.readouterr().out)
    assert output["relations"][0]["relation"] == "capital_of"
    assert output["relations"][0]["subject"] == "Ankara"
    assert output["relations"][0]["object"] == "Turkey"
