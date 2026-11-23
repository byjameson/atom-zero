import json

from atomzero.claims import (
    audit_answer,
    compare_claims,
    extract_claims,
    link_evidence,
    main,
)

TEXT = (
    "The Skyline battery lasts 8 hours. "
    "The motor uses 250 watts at peak load.\n"
    "Charging the Skyline takes 45 minutes."
)
BATTERY_HIT = {
    "source": "battery.md",
    "title": "Battery",
    "section": "Runtime",
    "page": 7,
    "chunk": (
        "Under normal load the Skyline battery lasts 8 hours. "
        "It charges in 45 minutes."
    ),
}
MOTOR_HIT = {
    "source": "motor.md",
    "title": "Motor",
    "section": "Power",
    "page": 2,
    "chunk": "The motor uses 250 watts at peak load.",
}


def test_extract_claims_splits_sentences_and_keeps_offsets():
    claims = extract_claims(TEXT)
    assert [claim["text"] for claim in claims] == [
        "The Skyline battery lasts 8 hours.",
        "The motor uses 250 watts at peak load.",
        "Charging the Skyline takes 45 minutes.",
    ]
    for claim in claims:
        assert TEXT[claim["start"]:claim["end"]] == claim["text"]
    assert [claim["claim_id"][:4] for claim in claims] == [
        "c001", "c002", "c003"]
    assert len({claim["claim_id"] for claim in claims}) == 3
    assert claims == extract_claims(TEXT)


def test_extract_claims_skips_empty_and_short_fragments():
    assert extract_claims("") == []
    assert extract_claims("   \n\n  ") == []
    assert extract_claims("Hi. Ok.") == []
    assert extract_claims("A real claim about the battery.") != []


def test_link_evidence_picks_best_hit_and_locators():
    claims = extract_claims("The Skyline battery lasts 8 hours.")
    linked = link_evidence(claims, [MOTOR_HIT, BATTERY_HIT])
    assert len(linked) == 1
    item = linked[0]
    assert item["status"] == "supported"
    assert item["support"] >= 0.45
    assert item["evidence"]["source"] == "battery.md"
    assert item["evidence"]["title"] == "Battery"
    assert item["evidence"]["section"] == "Runtime"
    assert item["evidence"]["page"] == 7
    assert "Skyline battery lasts 8 hours" in item["evidence"]["snippet"]


def test_link_evidence_supported_weak_and_unsupported():
    claims = extract_claims(
        "The Skyline battery lasts 8 hours. "
        "The Skyline battery warranty covers five years. "
        "The parcel weighs ten kilograms.")
    linked = link_evidence(claims, [BATTERY_HIT])
    assert [item["status"] for item in linked] == [
        "supported", "weak", "unsupported"]
    assert linked[0]["evidence"] is not None
    assert linked[1]["evidence"] is not None
    assert 0.0 < linked[1]["support"] < 0.45
    assert linked[2]["evidence"] is None
    assert linked[2]["support"] == 0.0


def test_compare_claims_flags_numeric_mismatch():
    claims = extract_claims(
        "The Skyline battery lasts 8 hours. "
        "The Skyline battery lasts 10 hours.")
    result = compare_claims(claims)
    assert result["claim_count"] == 2
    assert result["conflict_count"] == 1
    assert result["counts"]["numeric_mismatch"] == 1
    pair = result["conflicts"][0]
    assert pair["claim_ids"] == [claim["claim_id"] for claim in claims]
    assert pair["reasons"] == ["numeric_mismatch"]
    assert "8 hours" in pair["left"]
    assert "10 hours" in pair["right"]


def test_compare_claims_does_not_invent_conflicts():
    claims = extract_claims(
        "The Skyline battery lasts 8 hours. "
        "The parcel weighs 10 kilograms.")
    result = compare_claims(claims)
    assert result["conflict_count"] == 0
    assert result["conflicts"] == []
    assert result["weights_changed"] is False


def test_audit_answer_composes_claims_evidence_and_summary():
    query = (
        "The Skyline battery lasts 8 hours. "
        "The Skyline battery lasts 10 hours.")
    result = audit_answer(query, [BATTERY_HIT])
    assert result["query"] == query
    assert len(result["claims"]) == 2
    assert all(item["evidence"] for item in result["claims"])
    assert result["summary"]["claim_count"] == 2
    assert result["summary"]["status_counts"]["supported"] == 2
    assert result["summary"]["conflict_count"] == 1
    assert result["summary"]["conflict_types"] == {"numeric_mismatch": 1}
    assert result["weights_changed"] is False
    assert result["method"] == "deterministic_claim_evidence_v1"


def test_audit_answer_empty_input_and_cli_exit_codes(capsys):
    result = audit_answer("x", [])
    assert result["query"] == "x"
    assert result["claims"] == []
    assert result["summary"]["claim_count"] == 0
    assert result["summary"]["status_counts"] == {
        "supported": 0, "weak": 0, "unsupported": 0}
    assert result["weights_changed"] is False
    assert result["method"] == "deterministic_claim_evidence_v1"
    assert extract_claims("") == []
    assert link_evidence([], [BATTERY_HIT]) == []
    assert compare_claims([])["conflict_count"] == 0

    assert main(["--text", ""]) == 2
    assert main(["--text", "x", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["claims"] == []
    assert payload["method"] == "deterministic_claim_evidence_v1"


def test_cli_reads_file_and_rejects_empty_file(tmp_path, capsys):
    path = tmp_path / "note.txt"
    path.write_text("The Skyline battery lasts 8 hours.", encoding="utf-8")
    assert main(["--file", str(path), "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["query"].startswith("The Skyline")
    empty = tmp_path / "empty.txt"
    empty.write_text("", encoding="utf-8")
    assert main(["--file", str(empty)]) == 2


def _conflict(left, right):
    return compare_claims([
        {"claim_id": "a", "text": left, "source": "a.txt"},
        {"claim_id": "b", "text": right, "source": "b.txt"},
    ])


def test_compare_claims_reports_date_identifier_and_negation_conflicts():
    date = _conflict(
        "The policy starts on 2020-01-01.",
        "The policy starts on 2021-01-01.")
    assert date["conflicts"][0]["reasons"] == ["date_mismatch"]
    identifier = _conflict(
        "The warranty code is W-7781.",
        "The warranty code is W-9900.")
    assert identifier["conflicts"][0]["reasons"] == ["identifier_mismatch"]
    negation = _conflict(
        "The store is open.",
        "The store is not open.")
    assert negation["conflicts"][0]["reasons"] == ["negation_mismatch"]
    unit = _conflict(
        "The battery lasts 8 hours.",
        "The battery lasts 10 hours.")
    assert "numeric_mismatch" in unit["conflicts"][0]["reasons"]


def test_structured_tokens_do_not_hide_real_numeric_conflicts():
    date_and_price = _conflict(
        "The 2020-01-01 policy price is 10.",
        "The 2021-01-01 policy price is 12.")
    assert "date_mismatch" in date_and_price["conflicts"][0]["reasons"]
    assert "numeric_mismatch" in date_and_price["conflicts"][0]["reasons"]
    code_and_count = _conflict(
        "Code W-7781 has 4 parts.",
        "Code W-9900 has 6 parts.")
    assert "identifier_mismatch" in code_and_count["conflicts"][0]["reasons"]
    assert "numeric_mismatch" in code_and_count["conflicts"][0]["reasons"]


def test_compare_claims_does_not_conflict_unrelated_or_matching_claims():
    assert _conflict(
        "The capital of France is Paris.",
        "The warranty code is W-7781.")["conflict_count"] == 0
    assert _conflict(
        "The battery lasts 8 hours.",
        "It lasts 8 hours.")["conflict_count"] == 0


def test_audit_output_is_deterministic():
    assert extract_claims(TEXT) == extract_claims(TEXT)
    first = link_evidence(extract_claims(TEXT), [MOTOR_HIT, BATTERY_HIT])
    second = link_evidence(extract_claims(TEXT), [MOTOR_HIT, BATTERY_HIT])
    assert first == second
    assert compare_claims(extract_claims(TEXT)) == compare_claims(
        extract_claims(TEXT))
    assert audit_answer(TEXT, [BATTERY_HIT]) == audit_answer(
        TEXT, [BATTERY_HIT])
    assert json.dumps(audit_answer(TEXT, [BATTERY_HIT]), sort_keys=True) == (
        json.dumps(audit_answer(TEXT, [BATTERY_HIT]), sort_keys=True))
