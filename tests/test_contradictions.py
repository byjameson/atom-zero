import json

from atomzero.contradictions import compare_sources, extract_claims
from atomzero.doc_memory import DocMemory, main


def test_compare_sources_flags_numeric_negation_and_codes(tmp_path):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("policy_a.txt",
                  "The warranty period is 24 months. "
                  "The warranty covers water damage.", title="Policy A")
    docs.add_text("policy_b.txt",
                  "The warranty period is 36 months. "
                  "The warranty does not cover water damage.",
                  title="Policy B")
    docs.add_text("launch_a.txt", "The launch code is L-7781.",
                  title="Launch A")
    docs.add_text("launch_b.txt", "The launch code is W-7781.",
                  title="Launch B")
    docs.add_text("unrelated.txt", "Bananas are yellow.", title="Unrelated")

    claims = extract_claims(docs, "policy_a.txt")
    assert len(claims) == 2
    result = compare_sources(docs, "policy_a.txt", "policy_b.txt")
    reasons = {reason for conflict in result["conflicts"]
               for reason in conflict["reasons"]}
    assert "numeric_mismatch" in reasons
    assert "negation_mismatch" in reasons
    assert result["summary"]["conflict_count"] >= 2
    assert result["weights_changed"] is False

    code = compare_sources(docs, "launch_a.txt", "launch_b.txt")
    assert any("identifier_mismatch" in conflict["reasons"]
               for conflict in code["conflicts"])
    clean = compare_sources(docs, "unrelated.txt", "policy_a.txt")
    assert clean["conflicts"] == []
    docs.close()


def test_compare_all_sources_and_cli(tmp_path, capsys):
    docs = DocMemory(tmp_path / "docs.db")
    docs.add_text("policy_a.txt", "The warranty period is 24 months.")
    docs.add_text("policy_b.txt", "The warranty period is 36 months.")
    docs.add_text("other.txt", "Bananas are yellow.")
    all_sources = compare_sources(docs, "policy_a.txt")
    assert all_sources["summary"]["sources_compared"] == 2
    assert all_sources["conflicts"]
    docs.close()

    source_a = tmp_path / "policy_a.txt"
    source_b = tmp_path / "policy_b.txt"
    source_a.write_text("The warranty period is 24 months.", encoding="utf-8")
    source_b.write_text("The warranty period is 36 months.", encoding="utf-8")
    db = tmp_path / "cli.db"
    main(["add", str(source_a), "--db", str(db)])
    main(["add", str(source_b), "--db", str(db)])
    capsys.readouterr()
    main(["compare", str(source_a), "--against", str(source_b),
          "--db", str(db)])
    output = json.loads(capsys.readouterr().out)
    assert output["source"].endswith("policy_a.txt")
    assert output["conflicts"]
