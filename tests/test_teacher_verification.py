import io
import json
import sys

import pytest

from atomzero.teacher_verification import (
    METHOD,
    STATUS_REASONS,
    main,
    verification_event,
    verify_answer,
)

SUPPORTED_ANSWER = (
    "The Skyline battery lasts 8 hours. "
    "The motor uses 250 watts at peak load."
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
HITS = [BATTERY_HIT, MOTOR_HIT]
WEAK_ANSWER = "The Skyline battery warranty covers five years."
UNSUPPORTED_ANSWER = "The parcel weighs ten kilograms."

RESULT_KEYS = {
    "status",
    "reason_code",
    "verified",
    "claim_count",
    "supported_count",
    "weak_count",
    "unsupported_count",
    "conflict_count",
    "source_count",
    "min_support",
    "requires_human",
    "auto_apply",
    "weights_changed",
    "method",
}
COUNT_KEYS = {
    "claim_count",
    "supported_count",
    "weak_count",
    "unsupported_count",
    "conflict_count",
    "source_count",
}
EVENT_KEYS = {
    "version",
    "status",
    "reason_code",
    "outcome",
    "verified",
    "promotable",
    "requires_human",
    "auto_apply",
    "weights_changed",
    "growth_event",
    "method",
} | COUNT_KEYS
FORBIDDEN_OUTPUT_KEYS = {
    "answer",
    "text",
    "claim_text",
    "snippet",
    "evidence",
    "source",
    "sources",
    "source_path",
    "hits",
    "path",
}


def _args(**overrides):
    data = {"answer": SUPPORTED_ANSWER, "hits": HITS}
    data.update(overrides)
    return data


def test_all_claims_supported_is_verified():
    result = verify_answer(SUPPORTED_ANSWER, HITS)
    assert set(result) == RESULT_KEYS
    assert result["status"] == "verified"
    assert result["reason_code"] == "all_claims_supported"
    assert result["verified"] is True
    assert result["claim_count"] == 2
    assert result["supported_count"] == 2
    assert result["weak_count"] == 0
    assert result["unsupported_count"] == 0
    assert result["conflict_count"] == 0
    assert result["source_count"] == 2
    assert result["min_support"] == 0.6
    assert result["requires_human"] is True
    assert result["auto_apply"] is False
    assert result["weights_changed"] is False
    assert result["method"] == METHOD


@pytest.mark.parametrize("answer, key", [
    (WEAK_ANSWER, "weak_count"),
    (UNSUPPORTED_ANSWER, "unsupported_count"),
])
def test_weak_or_unsupported_claim_is_unverified(answer, key):
    result = verify_answer(answer, [BATTERY_HIT])
    assert result["status"] == "unverified"
    assert result["reason_code"] == "insufficient_support"
    assert result["verified"] is False
    assert result["claim_count"] == 1
    assert result[key] == 1
    assert result["requires_human"] is False
    assert result["auto_apply"] is False
    assert result["weights_changed"] is False


def test_weak_claim_can_pass_a_lower_support_threshold():
    result = verify_answer(WEAK_ANSWER, [BATTERY_HIT], min_support=0.0)
    assert result["status"] == "verified"
    assert result["supported_count"] == 1
    assert result["weak_count"] == 0
    assert result["min_support"] == 0.0


@pytest.mark.parametrize("answer, hit", [
    (
        "The Skyline battery lasts 8 hours. "
        "The Skyline battery lasts 10 hours.",
        BATTERY_HIT,
    ),
    (
        "The Skyline policy starts on 2020-01-01. "
        "The Skyline policy starts on 2021-01-01.",
        {
            "source": "policy.md",
            "chunk": "The Skyline policy starts on 2020-01-01.",
        },
    ),
])
def test_conflicting_numeric_or_date_claim_is_conflicted(answer, hit):
    result = verify_answer(answer, [hit])
    assert result["status"] == "conflicted"
    assert result["reason_code"] == "contradiction"
    assert result["verified"] is False
    assert result["claim_count"] == 2
    assert result["conflict_count"] == 1
    assert result["requires_human"] is False
    assert result["auto_apply"] is False
    assert result["weights_changed"] is False


def test_no_claims_is_empty():
    result = verify_answer("Hi.", HITS)
    assert result["status"] == "empty"
    assert result["reason_code"] == "no_claims"
    assert result["verified"] is False
    assert result["claim_count"] == 0
    assert result["source_count"] == 0
    assert result["requires_human"] is False
    assert result["auto_apply"] is False
    assert result["weights_changed"] is False


def test_require_sources_controls_the_source_gate():
    hit = {"chunk": "The Skyline battery lasts 8 hours."}
    required = verify_answer(
        "The Skyline battery lasts 8 hours.", [hit],
        require_sources=True)
    assert required["status"] == "unverified"
    assert required["reason_code"] == "no_sources"
    assert required["supported_count"] == 1
    assert required["source_count"] == 0
    assert required["verified"] is False

    optional = verify_answer(
        "The Skyline battery lasts 8 hours.", [hit],
        require_sources=False)
    assert optional["status"] == "verified"
    assert optional["reason_code"] == "all_claims_supported"
    assert optional["source_count"] == 0
    assert optional["requires_human"] is True


@pytest.mark.parametrize("case, message", [
    (_args(answer=None), "answer must be a string"),
    (_args(answer=123), "answer must be a string"),
    (_args(answer="   \n"), "answer must be a non-empty string"),
    (_args(hits=None), "hits must be a list of mappings"),
    (_args(hits="not a list"), "hits must be a list of mappings"),
    (_args(hits=tuple(HITS)), "hits must be a list of mappings"),
    (_args(hits=[None]), r"hits\[0\] must be a mapping"),
    (_args(hits=[BATTERY_HIT, "x"]), r"hits\[1\] must be a mapping"),
    (_args(hits=[{1: "chunk"}]), r"hits\[0\] keys must be strings"),
    (_args(min_support="0.6"),
     "min_support must be a number between 0 and 1"),
    (_args(min_support=-0.1),
     "min_support must be a number between 0 and 1"),
    (_args(min_support=1.01),
     "min_support must be a number between 0 and 1"),
    (_args(min_support=True),
     "min_support must be a number between 0 and 1"),
    (_args(min_support=float("nan")),
     "min_support must be a number between 0 and 1"),
    (_args(require_sources=1), "require_sources must be a boolean"),
])
def test_malformed_inputs_raise_value_error(case, message):
    with pytest.raises(ValueError, match=message):
        verify_answer(**case)


def test_result_is_content_free():
    result = verify_answer(SUPPORTED_ANSWER, HITS)
    assert set(result) == RESULT_KEYS
    assert not (set(result) & FORBIDDEN_OUTPUT_KEYS)
    rendered = json.dumps(result)
    assert SUPPORTED_ANSWER not in rendered
    assert "The Skyline battery" not in rendered
    assert "battery.md" not in rendered
    assert "motor.md" not in rendered
    assert "250 watts" not in rendered
    for status, reason in STATUS_REASONS.items():
        assert status in {
            "empty", "conflicted", "unverified", "verified"}
        assert reason


def test_verification_event_is_content_free_and_promotable_only_when_verified():
    verified = verify_answer(SUPPORTED_ANSWER, HITS)
    event = verification_event(verified)
    assert set(event) == EVENT_KEYS
    assert event["version"] == 1
    assert event["status"] == "verified"
    assert event["reason_code"] == "all_claims_supported"
    assert event["outcome"] == "success"
    assert event["verified"] is True
    assert event["promotable"] is True
    assert event["requires_human"] is True
    assert event["growth_event"] == "verified_teacher"
    assert event["auto_apply"] is False
    assert event["weights_changed"] is False
    assert event["method"] == METHOD
    for key in COUNT_KEYS:
        assert event[key] == verified[key]
    rendered = json.dumps(event)
    assert SUPPORTED_ANSWER not in rendered
    assert "battery.md" not in rendered
    assert "The Skyline battery" not in rendered

    for result in (
            verify_answer(WEAK_ANSWER, [BATTERY_HIT]),
            verify_answer("Hi.", HITS),
            verify_answer(
                "The Skyline battery lasts 8 hours. "
                "The Skyline battery lasts 10 hours.",
                [BATTERY_HIT])):
        other = verification_event(result)
        assert other["promotable"] is False
        assert other["requires_human"] is False
        assert other["growth_event"] == ""
        assert other["auto_apply"] is False
        assert other["weights_changed"] is False

    failed = verification_event(verified, outcome="failure")
    assert failed["promotable"] is False
    assert failed["growth_event"] == ""
    assert failed["verified"] is True


@pytest.mark.parametrize("change, message", [
    ({"status": "unknown"}, "result.status must be one of"),
    ({"reason_code": "no_claims"}, "reason_code is not valid for status"),
    ({"verified": 1}, "result.verified must be a boolean"),
    ({"verified": False}, "result.verified does not match status"),
    ({"claim_count": -1}, "claim_count must be a non-negative integer"),
    ({"claim_count": 3}, "counts do not add up to claim_count"),
    ({"source_count": 3}, "source_count cannot exceed claim_count"),
])
def test_verification_event_rejects_malformed_results(change, message):
    result = dict(verify_answer(SUPPORTED_ANSWER, HITS))
    result.update(change)
    with pytest.raises(ValueError, match=message):
        verification_event(result)


def test_verification_event_validates_outcome():
    result = verify_answer(SUPPORTED_ANSWER, HITS)
    with pytest.raises(ValueError, match="outcome must be one of"):
        verification_event(result, outcome="ok")
    with pytest.raises(ValueError, match="result must be a mapping"):
        verification_event("not a result")


def _payload(answer=SUPPORTED_ANSWER, hits=None):
    return {"answer": answer, "hits": HITS if hits is None else hits}


def _write_json(tmp_path, name, value):
    path = tmp_path / name
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_cli_json_file(tmp_path, capsys):
    path = _write_json(tmp_path, "answer.json", _payload())
    code = main(["--answer-json", str(path), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert set(payload) == RESULT_KEYS
    assert payload["status"] == "verified"
    assert payload["verified"] is True
    assert payload["auto_apply"] is False
    assert payload["weights_changed"] is False
    assert SUPPORTED_ANSWER not in captured.out
    assert "battery.md" not in captured.out


def test_cli_reads_stdin(monkeypatch, capsys):
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps(_payload())))
    code = main(["--answer-json", "-", "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "verified"
    assert payload["verified"] is True


def test_cli_valid_unverified_result_exits_zero(tmp_path, capsys):
    path = _write_json(
        tmp_path, "answer.json", _payload(answer="Hi.", hits=[]))
    code = main(["--answer-json", str(path), "--json"])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    payload = json.loads(captured.out)
    assert payload["status"] == "empty"
    assert payload["verified"] is False


def test_cli_human_output_is_content_free(tmp_path, capsys):
    path = _write_json(tmp_path, "answer.json", _payload())
    code = main(["--answer-json", str(path)])
    captured = capsys.readouterr()
    assert code == 0
    assert captured.err == ""
    assert "status: verified" in captured.out
    assert "reason: all_claims_supported" in captured.out
    assert "requires_human: true" in captured.out
    assert "auto_apply: false" in captured.out
    assert "weights_changed: false" in captured.out
    assert SUPPORTED_ANSWER not in captured.out
    assert "battery.md" not in captured.out


def test_cli_input_errors_exit_two(tmp_path, capsys):
    assert main([]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--answer-json is required" in captured.err

    missing = tmp_path / "missing.json"
    assert main(["--answer-json", str(missing), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error:")

    invalid = tmp_path / "invalid.json"
    invalid.write_text("{not json", encoding="utf-8")
    assert main(["--answer-json", str(invalid), "--json"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "invalid JSON" in captured.err

    as_list = _write_json(tmp_path, "list.json", [])
    assert main(["--answer-json", str(as_list), "--json"]) == 2
    captured = capsys.readouterr()
    assert "must be a JSON object" in captured.err

    missing_hits = _write_json(
        tmp_path, "missing_hits.json", {"answer": SUPPORTED_ANSWER})
    assert main(["--answer-json", str(missing_hits), "--json"]) == 2
    captured = capsys.readouterr()
    assert "hits are required" in captured.err

    empty_answer = _write_json(
        tmp_path, "empty_answer.json", {"answer": "", "hits": HITS})
    assert main(["--answer-json", str(empty_answer), "--json"]) == 2
    captured = capsys.readouterr()
    assert "answer must be a non-empty string" in captured.err


def test_cli_rejects_empty_stdin(monkeypatch, capsys):
    monkeypatch.setattr(sys, "stdin", io.StringIO(""))
    code = main(["--answer-json", "-", "--json"])
    captured = capsys.readouterr()
    assert code == 2
    assert captured.out == ""
    assert "empty JSON input: stdin" in captured.err
