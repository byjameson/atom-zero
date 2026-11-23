import json

from atomzero.doc_memory import DocMemory, main
from atomzero.timeline import extract_timeline


def _docs(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("history.txt",
                  "The launch was on 5 January 2020. "
                  "A follow-up happened in 2021.", title="History")
    docs.add_text("toplanti.txt",
                  "Toplantı 6 Ocak 2020 tarihinde yapıldı.",
                  title="Toplantı")
    docs.add_text("notes.txt", "Bananas are yellow.", title="Notes")
    return docs


def test_extract_timeline_sorts_dates_and_keeps_provenance(tmp_path):
    docs = _docs(tmp_path)
    result = extract_timeline(docs)
    assert result["count"] == 3
    assert [event["date"] for event in result["events"]] == [
        "2020-01-05", "2020-01-06", "2021"]
    assert result["events"][0]["source"] == "history.txt"
    assert result["events"][0]["title"] == "History"
    assert result["events"][0]["precision"] == "day"
    assert result["events"][2]["precision"] == "year"
    assert all(event["text"] for event in result["events"])
    assert not any(event["source"] == "notes.txt"
                   for event in result["events"])
    docs.close()


def test_timeline_filters_limits_and_cli(tmp_path, capsys):
    docs = _docs(tmp_path)
    filtered = extract_timeline(docs, source="toplanti.txt")
    assert filtered["count"] == 1
    assert filtered["events"][0]["date"] == "2020-01-06"
    limited = extract_timeline(docs, limit=1)
    assert len(limited["events"]) == 1
    docs.close()

    path = tmp_path / "cli.db"
    source = tmp_path / "event.txt"
    source.write_text("The event is on 5 January 2020.", encoding="utf-8")
    main(["add", str(source), "--db", str(path), "--title", "Event"])
    capsys.readouterr()
    main(["timeline", "--db", str(path)])
    output = json.loads(capsys.readouterr().out)
    assert output["events"][0]["date"] == "2020-01-05"
    assert output["events"][0]["title"] == "Event"
