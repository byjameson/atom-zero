import json
import subprocess
import sys
import tempfile
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from atomzero.temporal import assess_validity, filter_as_of, parse_date

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, stdin: str | None = None):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.temporal", *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


class DateTests(unittest.TestCase):
    def test_parse_iso_date(self):
        self.assertEqual(parse_date("2026-11-23"), date(2026, 11, 23))

    def test_parse_datetime_and_date(self):
        self.assertEqual(
            parse_date(datetime(2026, 11, 23, 12, tzinfo=timezone.utc)),
            date(2026, 11, 23))
        self.assertEqual(parse_date(date(2026, 1, 2)), date(2026, 1, 2))

    def test_parse_invalid(self):
        with self.assertRaises(ValueError):
            parse_date("not-a-date")
        with self.assertRaises(ValueError):
            parse_date(None)


class ValidityTests(unittest.TestCase):
    def test_valid_window(self):
        hit = {"source": "a", "valid_from": "2026-01-01",
               "valid_until": "2026-12-31"}
        result = assess_validity(hit, "2026-11-23")
        self.assertEqual(result["status"], "valid")
        self.assertEqual(result["days_remaining"], 38)

    def test_boundaries_are_inclusive(self):
        hit = {"valid_from": "2026-01-01", "valid_until": "2026-12-31"}
        self.assertEqual(
            assess_validity(hit, "2026-01-01")["status"], "valid")
        self.assertEqual(
            assess_validity(hit, "2026-12-31")["status"], "valid")

    def test_expired(self):
        hit = {"valid_until": "2025-12-31"}
        self.assertEqual(
            assess_validity(hit, "2026-11-23")["status"], "expired")

    def test_not_yet(self):
        hit = {"valid_from": "2027-01-01"}
        result = assess_validity(hit, "2026-11-23")
        self.assertEqual(result["status"], "not_yet")
        self.assertEqual(result["days_until_start"], 39)

    def test_undated(self):
        result = assess_validity({"source": "x"}, "2026-11-23")
        self.assertEqual(result["status"], "undated")
        self.assertIsNone(result["valid_from"])
        self.assertIsNone(result["valid_until"])

    def test_invalid_window_value(self):
        with self.assertRaises(ValueError):
            assess_validity({"valid_from": "bad"}, "2026-11-23")

    def test_non_object_hit(self):
        with self.assertRaises(ValueError):
            assess_validity("bad")  # type: ignore[arg-type]


class FilterTests(unittest.TestCase):
    def _hits(self):
        return [
            {"source": "valid", "valid_from": "2026-01-01",
             "valid_until": "2026-12-31"},
            {"source": "expired", "valid_until": "2025-12-31"},
            {"source": "future", "valid_from": "2027-01-01"},
            {"source": "undated"},
        ]

    def test_filter_keeps_valid_and_undated(self):
        report = filter_as_of(self._hits(), "2026-11-23")
        self.assertEqual(report["summary"]["included"], 2)
        self.assertEqual(report["summary"]["excluded"], 2)
        self.assertEqual(report["summary"]["status_counts"], {
            "valid": 1, "expired": 1, "not_yet": 1, "undated": 1})
        self.assertFalse(report["weights_changed"])

    def test_filter_can_drop_undated(self):
        report = filter_as_of(
            self._hits(), "2026-11-23", keep_undated=False)
        self.assertEqual(report["summary"]["included"], 1)
        self.assertEqual(report["summary"]["excluded"], 3)

    def test_empty_hits(self):
        report = filter_as_of([], "2026-11-23")
        self.assertEqual(report["summary"]["total"], 0)
        self.assertEqual(report["included"], [])

    def test_non_list(self):
        with self.assertRaises(ValueError):
            filter_as_of("bad", "2026-11-23")  # type: ignore[arg-type]

    def test_deterministic(self):
        first = filter_as_of(self._hits(), "2026-11-23")
        second = filter_as_of(self._hits(), "2026-11-23")
        self.assertEqual(first, second)

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.json"
            path.write_text(json.dumps(self._hits()), encoding="utf-8")
            result = run_cli("--file", str(path), "--as-of", "2026-11-23",
                             "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["summary"]["included"], 2)

    def test_cli_exclude_undated(self):
        result = run_cli(
            "--as-of", "2026-11-23", "--exclude-undated", "--json",
            stdin=json.dumps(self._hits()))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["summary"]["included"], 1)

    def test_cli_empty_exit_two(self):
        result = run_cli("--json", stdin="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no hits", result.stderr)

    def test_cli_invalid_as_of(self):
        result = run_cli(
            "--as-of", "bad", stdin=json.dumps(self._hits()))
        self.assertEqual(result.returncode, 2)
        self.assertIn("input error", result.stderr)


if __name__ == "__main__":
    unittest.main()
