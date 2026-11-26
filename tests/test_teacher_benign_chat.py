"""Tests for the teacher-generated benign chat builder.

The teacher is always a fake callable. No test downloads or loads a model.
"""
import hashlib
import json
from collections import Counter

import pytest

from atomzero.safety import SafetyFilter
from research import build_teacher_benign_chat as chat

ROW_FIELDS = {
    "user", "assistant", "family", "source", "teacher_model", "style",
}
CONTENT_KEYS = {
    "user", "prompt", "prompts", "assistant", "reply", "text", "rows",
    "messages", "content",
}
GOOD_ANSWER = (
    "This is a calm and helpful answer with enough words for the filter.")


def _fake_generator(prompts):
    return [GOOD_ANSWER for _ in prompts]


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def _build_dataset(root, *, rows=16, seed=7, batch_size=4):
    out = root / "train.jsonl"
    manifest_path = root / "manifest.json"
    rows_out, counts = chat.build_rows(
        _fake_generator, rows=rows, seed=seed, batch_size=batch_size)
    manifest = chat.write_dataset(
        rows_out, out, manifest=manifest_path, counts=counts, seed=seed,
        batch_size=batch_size)
    return out, manifest_path, rows_out, manifest


def test_prompt_pool_is_large_unique_and_valid():
    assert chat.available_prompts() == 768
    assert chat.available_prompts() >= chat.DEFAULT_ROWS
    assert len(chat.CATEGORY_ORDER) == 8

    all_prompts = []
    for category in chat.CATEGORY_ORDER:
        prompts = chat.prompt_pool(category)
        assert len(prompts) == chat.PROMPTS_PER_CATEGORY == 96
        assert len(set(prompts)) == len(prompts)
        assert all(chat.validate_user(prompt) for prompt in prompts)
        assert all(len(prompt) <= chat.MAX_USER_CHARS for prompt in prompts)
        assert all("\n" not in prompt for prompt in prompts)
        all_prompts.extend(prompts)
    assert len(set(all_prompts)) == len(all_prompts)


def test_clean_answer_strips_fences_prefixes_and_linebreaks():
    assert chat.clean_answer(
        '  "Assistant: Hello there.\nThis is fine."  '
    ) == "Hello there. This is fine."
    assert chat.clean_answer(
        "```text\nHello there. This is a test.\n```"
    ) == "Hello there. This is a test."
    assert chat.clean_answer("- Answer: A short reply.") == "A short reply."
    assert chat.clean_answer(None) == ""
    assert chat.clean_answer("   ") == ""


def test_coerce_answers_handles_strings_none_and_wrong_length():
    assert chat.coerce_answers("one", 1) == ["one"]
    assert chat.coerce_answers(["one"], 3) == ["one", "", ""]
    assert chat.coerce_answers(None, 2) == ["", ""]
    assert chat.coerce_answers([1, None], 2) == ["1", ""]


def test_validate_answer_enforces_length_and_word_bounds():
    assert len("This is a good test.") == 20
    assert len("This is a good test.".split()) == 5
    assert chat.validate_answer("This is a good test.")
    assert not chat.validate_answer("This is a good tes.")
    assert not chat.validate_answer("I am a cat now.")
    assert not chat.validate_answer("one two three four")
    assert not chat.validate_answer("")
    assert not chat.validate_answer(None)
    assert not chat.validate_answer("x" * (chat.MAX_ANSWER_CHARS + 1))
    assert chat.validate_answer("word " * 59 + "word.")
    assert chat.validate_answer(
        "Keep your phone charged and your documents together.")


def test_validate_answer_rejects_refusals_unsafe_terms_and_personal_data():
    safety = SafetyFilter()
    refusals = (
        "I'm sorry, but I can't help with that request.",
        "I cannot assist with this particular question.",
        "I would rather not answer that question today.",
        "As an AI language model, I must decline this request.",
    )
    for answer in refusals:
        assert not chat.validate_answer(answer, safety=safety), answer

    unsafe = (
        "Here is how to make a bomb with common household materials.",
        "Use this api key to access the private service right now.",
        "You can steal a car by following these simple steps today.",
        "Send the details to person@example.com before the meeting.",
        "Call 555 123 4567 to confirm the booking with the hotel.",
        "The code is 1234567 and it unlocks the storage room.",
    )
    for answer in unsafe:
        assert not chat.validate_answer(answer, safety=safety), answer


def test_rejection_reason_detects_prompt_and_pair_reuse():
    safety = SafetyFilter()
    prompt = "Hello! How are you?"
    answer = "I am doing well and ready to help with your questions."
    assert chat.rejection_reason(prompt, answer, safety=safety) is None

    seen_users = {chat.normalize_text(prompt)}
    assert chat.rejection_reason(
        "hello!   how are you?", answer, safety=safety,
        seen_users=seen_users) == "duplicate_prompt"

    seen_pairs = {(chat.normalize_text(prompt), chat.normalize_text(answer))}
    assert chat.rejection_reason(
        prompt, answer, safety=safety,
        seen_pairs=seen_pairs) == "duplicate_pair"


def test_build_rows_with_fake_generator_is_deterministic_and_valid():
    rows, counts = chat.build_rows(
        _fake_generator, rows=16, seed=7, batch_size=4)
    again, counts_again = chat.build_rows(
        _fake_generator, rows=16, seed=7, batch_size=4)

    assert rows == again
    assert counts == counts_again
    assert len(rows) == 16
    assert counts["accepted"] == 16
    assert counts["rejected"] == 0
    assert counts["duplicates"] == 0
    assert counts["shortfall"] == 0
    assert counts["by_category"] == {
        category: 2 for category in chat.CATEGORY_ORDER}
    assert Counter(row["family"] for row in rows) == {
        category: 2 for category in chat.CATEGORY_ORDER}

    for row in rows:
        assert set(row) == ROW_FIELDS
        assert row["source"] == "teacher_generated"
        assert row["style"] == "benign_chat"
        assert row["teacher_model"] == chat.DEFAULT_MODEL
        assert chat.validate_row(row)
        _filtered, verdict = SafetyFilter().filter_reply(row["assistant"])
        assert verdict is None

    other, _ = chat.build_rows(
        _fake_generator, rows=16, seed=8, batch_size=4)
    assert [row["user"] for row in other] != [row["user"] for row in rows]
    assert len(other) == len(rows)


def test_build_rows_rejects_refusal_answers_and_reports_counts():
    def refusing(prompts):
        return ["I'm sorry, but I can't help with that request."] * len(prompts)

    rows, counts = chat.build_rows(
        refusing, rows=8, seed=5, batch_size=4)

    assert rows == []
    assert counts["requested"] == 8
    assert counts["accepted"] == 0
    assert counts["rejected"] == counts["attempted"]
    assert counts["rejected"] > 0
    assert counts["shortfall"] == 8
    assert counts["duplicates"] == 0
    assert all(value == 0 for value in counts["by_category"].values())


def test_build_rows_rejects_invalid_targets_and_missing_generator():
    with pytest.raises(ValueError):
        chat.build_rows(None, rows=4)
    with pytest.raises(ValueError):
        chat.build_rows(_fake_generator, rows=chat.available_prompts() + 1)
    with pytest.raises(ValueError):
        chat.build_rows(_fake_generator, rows=True)
    with pytest.raises(ValueError):
        chat.build_rows(_fake_generator, rows=4, seed=True)
    with pytest.raises(ValueError):
        chat.build_rows(_fake_generator, rows=4, batch_size=0)
    with pytest.raises(ValueError):
        chat.build_rows(_fake_generator, rows=4, model="  ")


def test_manifest_is_content_free_and_matches_the_output(tmp_path):
    out, manifest_path, rows, manifest = _build_dataset(tmp_path)
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest["schema"] == "teacher_benign_chat_v1"
    assert manifest["source"] == "teacher_generated"
    assert manifest["style"] == "benign_chat"
    assert manifest["seed"] == 7
    assert manifest["requested"] == 16
    assert manifest["accepted"] == 16
    assert manifest["rejected"] == 0
    assert manifest["shortfall"] == 0
    assert manifest["by_category"] == {
        category: 2 for category in chat.CATEGORY_ORDER}
    assert sum(manifest["by_category"].values()) == len(rows)
    assert manifest["sha256"] == hashlib.sha256(
        out.read_bytes()).hexdigest()
    assert manifest["bytes"] == len(out.read_bytes())
    assert manifest["content_free"] is True

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows:
        assert row["user"] not in text
        assert row["assistant"] not in text
    assert str(tmp_path) not in text
    assert len(_read_jsonl(out)) == len(rows)


def test_cli_writes_dataset_with_injected_teacher(tmp_path):
    out = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"

    def factory(model, *, device, max_new_tokens, local_files_only):
        assert model == "stub-teacher"
        assert device == "cpu"
        assert max_new_tokens == 48
        assert local_files_only is True
        return _fake_generator

    code = chat.main([
        "--model", "stub-teacher",
        "--device", "cpu",
        "--rows", "16",
        "--batch-size", "4",
        "--max-new-tokens", "48",
        "--seed", "7",
        "--out", str(out),
        "--manifest", str(manifest_path),
    ], teacher_factory=factory)

    assert code == 0
    assert len(_read_jsonl(out)) == 16
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["teacher_model"] == "stub-teacher"
    assert manifest["by_category"] == {
        category: 2 for category in chat.CATEGORY_ORDER}


def test_cli_zero_rows_does_not_load_the_teacher(tmp_path):
    def factory(*args, **kwargs):
        raise AssertionError("teacher must not be loaded when rows is zero")

    out = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    assert chat.main([
        "--rows", "0",
        "--out", str(out),
        "--manifest", str(manifest_path),
    ], teacher_factory=factory) == 0
    assert out.read_text(encoding="utf-8") == ""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["requested"] == 0
    assert manifest["accepted"] == 0
    assert manifest["shortfall"] == 0


def test_existing_output_requires_force(tmp_path):
    out = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"
    args = [
        "--model", "stub-teacher",
        "--device", "cpu",
        "--rows", "8",
        "--out", str(out),
        "--manifest", str(manifest_path),
    ]

    def factory(*args, **kwargs):
        return _fake_generator

    assert chat.main(args, teacher_factory=factory) == 0
    first_out = out.read_bytes()
    first_manifest = manifest_path.read_bytes()

    with pytest.raises(SystemExit) as excinfo:
        chat.main(args, teacher_factory=factory)
    assert excinfo.value.code == 2
    assert out.read_bytes() == first_out
    assert manifest_path.read_bytes() == first_manifest

    assert chat.main(args + ["--force"], teacher_factory=factory) == 0
    assert out.read_bytes() == first_out
    assert manifest_path.read_bytes() == first_manifest


def test_missing_local_model_fails_cleanly(tmp_path, capsys):
    missing = tmp_path / "definitely-missing-teacher"
    out = tmp_path / "train.jsonl"
    manifest_path = tmp_path / "manifest.json"

    with pytest.raises(FileNotFoundError):
        chat.load_teacher(
            str(missing), device="cpu", local_files_only=True)

    with pytest.raises(SystemExit) as excinfo:
        chat.main([
            "--model", str(missing),
            "--device", "cpu",
            "--rows", "2",
            "--out", str(out),
            "--manifest", str(manifest_path),
        ])

    assert excinfo.value.code == 2
    captured = capsys.readouterr()
    assert "local teacher path does not exist" in captured.err
    assert not out.exists()
    assert not manifest_path.exists()
