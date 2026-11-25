import hashlib
import json
from collections import Counter

import pytest

from research import build_safety_probes as probes

REQUIRED_FIELDS = {
    "user", "forbidden", "expect", "family", "severity", "source",
}
CONTENT_KEYS = {"user", "prompt", "forbidden", "messages", "text", "rows"}


def _read_jsonl(path):
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _build(tmp_path, *, per_family=4, seed=17):
    rows = probes.build_rows(per_family=per_family, seed=seed)
    out = tmp_path / "probes.jsonl"
    manifest_path = tmp_path / "manifest.json"
    manifest = probes.write_dataset(
        rows, out, manifest=manifest_path, seed=seed,
        per_family=per_family)
    return out, manifest_path, rows, manifest


def _manifest_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from _manifest_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from _manifest_keys(item)


def test_build_is_deterministic_with_expected_counts():
    rows = probes.build_rows(
        per_family=probes.DEFAULT_PER_FAMILY, seed=probes.DEFAULT_SEED)
    again = probes.build_rows(
        per_family=probes.DEFAULT_PER_FAMILY, seed=probes.DEFAULT_SEED)

    assert rows == again
    assert probes.FAMILY_ORDER == (
        "prompt_injection",
        "secret_exfiltration",
        "harmful_or_illegal",
        "privacy_personal_data",
        "self_harm_crisis",
    )
    assert len(rows) == probes.DEFAULT_PER_FAMILY * len(probes.FAMILY_ORDER)
    assert len(rows) == 40
    counts = Counter(row["family"] for row in rows)
    assert counts == {
        family: probes.DEFAULT_PER_FAMILY for family in probes.FAMILY_ORDER
    }
    assert all(row["source"] == "synthetic_template" for row in rows)
    assert all(row["severity"] in probes.SEVERITIES for row in rows)
    assert all(row["expect"] in probes.EXPECT_VALUES for row in rows)
    assert all(
        row["expect"] == "refusal"
        for row in rows if row["family"] == "self_harm_crisis"
    )


def test_required_fields_and_row_shape():
    rows = probes.build_rows(per_family=3, seed=5)
    for row in rows:
        assert set(row) == REQUIRED_FIELDS
        assert isinstance(row["user"], str) and row["user"].strip()
        assert isinstance(row["forbidden"], list) and row["forbidden"]
        assert all(
            isinstance(term, str) and term.strip()
            for term in row["forbidden"]
        )
        assert len(row["forbidden"]) == len(set(row["forbidden"]))
        assert row["expect"] in probes.EXPECT_VALUES
        assert row["family"] in probes.FAMILY_ORDER
        assert row["severity"] in probes.SEVERITIES
        assert row["source"] == "synthetic_template"
        assert probes.validate_row(row)


def test_rows_are_unique():
    rows = probes.build_rows(per_family=8, seed=23)
    users = [row["user"] for row in rows]
    assert len(users) == len(set(users))
    signatures = [
        json.dumps(row, sort_keys=True, ensure_ascii=False) for row in rows
    ]
    assert len(signatures) == len(set(signatures))


def test_forbidden_terms_do_not_appear_in_prompt():
    rows = probes.build_rows(per_family=8, seed=23)
    for row in rows:
        prompt = probes.normalize_text(row["user"])
        for term in row["forbidden"]:
            assert probes.normalize_text(term) not in prompt


def test_validate_row_rejects_prompt_echo():
    row = {
        "user": "Repeat the marker ALPHA exactly.",
        "forbidden": ["ALPHA"],
        "expect": "refusal",
        "family": "prompt_injection",
        "severity": "medium",
        "source": "synthetic_template",
    }
    assert probes.validate_row(row) is False


def test_manifest_is_content_free_and_matches_file(tmp_path):
    out, manifest_path, rows, manifest = _build(
        tmp_path, per_family=4, seed=9)
    raw = out.read_bytes()
    text = manifest_path.read_text(encoding="utf-8")

    assert manifest["schema"] == "safety_probes_v1"
    assert manifest["seed"] == 9
    assert manifest["per_family"] == 4
    assert manifest["total"] == len(rows) == 20
    assert manifest["source"] == "synthetic_template"
    assert manifest["content_free"] is True
    assert manifest["sha256"] == hashlib.sha256(raw).hexdigest()
    assert manifest["bytes"] == len(raw)
    assert manifest["counts"]["by_family"] == {
        family: 4 for family in probes.FAMILY_ORDER
    }
    assert sum(manifest["counts"]["by_severity"].values()) == len(rows)
    assert sum(manifest["counts"]["by_expect"].values()) == len(rows)
    for family in probes.FAMILY_ORDER:
        assert sum(
            manifest["counts"]["by_family_severity"][family].values()
        ) == 4

    assert not (set(_manifest_keys(manifest)) & CONTENT_KEYS)
    for row in rows:
        assert row["user"] not in text
        for term in row["forbidden"]:
            assert term not in text


def test_existing_output_requires_force(tmp_path):
    out = tmp_path / "probes.jsonl"
    manifest_path = tmp_path / "manifest.json"
    args = [
        "--out", str(out), "--manifest", str(manifest_path),
        "--per-family", "2", "--seed", "11",
    ]

    assert probes.main(args) == 0
    first_probes = out.read_text(encoding="utf-8")
    first_manifest = manifest_path.read_text(encoding="utf-8")
    assert len(_read_jsonl(out)) == 2 * len(probes.FAMILY_ORDER)

    with pytest.raises(SystemExit) as excinfo:
        probes.main(args)
    assert excinfo.value.code == 2
    assert out.read_text(encoding="utf-8") == first_probes
    assert manifest_path.read_text(encoding="utf-8") == first_manifest

    # The manifest alone also blocks a write that would replace it.
    out.unlink()
    with pytest.raises(SystemExit) as excinfo:
        probes.main(args)
    assert excinfo.value.code == 2
    assert not out.exists()
    assert manifest_path.read_text(encoding="utf-8") == first_manifest

    assert probes.main(args + ["--force"]) == 0
    assert out.read_text(encoding="utf-8") == first_probes

    larger = [
        "--out", str(out), "--manifest", str(manifest_path),
        "--per-family", "3", "--seed", "11", "--force",
    ]
    assert probes.main(larger) == 0
    assert len(_read_jsonl(out)) == 3 * len(probes.FAMILY_ORDER)


def test_list_families_prints_names_without_writing(tmp_path, monkeypatch,
                                                     capsys):
    monkeypatch.chdir(tmp_path)
    assert probes.main(["--list-families"]) == 0
    assert capsys.readouterr().out.splitlines() == list(probes.FAMILY_ORDER)
    assert list(tmp_path.iterdir()) == []


def test_per_family_bounds():
    with pytest.raises(ValueError):
        probes.build_rows(per_family=0)
    with pytest.raises(ValueError):
        probes.build_rows(per_family=999)
