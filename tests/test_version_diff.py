import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atomzero.version_diff import (
    audit_document_diff,
    diff_blocks,
    split_blocks,
)

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.version_diff", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


class BlockTests(unittest.TestCase):
    def test_split_blocks_normalizes_whitespace(self):
        blocks = split_blocks("  First   line \n\n\n Second\tline  \r\n")
        self.assertEqual(blocks, ["First line", "Second line"])

    def test_split_empty(self):
        self.assertEqual(split_blocks("   \n\n "), [])

    def test_split_rejects_non_string(self):
        with self.assertRaises(ValueError):
            split_blocks(None)  # type: ignore[arg-type]

    def test_diff_identical(self):
        result = diff_blocks(["a", "b"], ["a", "b"])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["changed"], [])
        self.assertEqual(result["unchanged"], 2)
        self.assertEqual(result["similarity"], 1.0)

    def test_diff_added_and_removed(self):
        inserted = diff_blocks(["a", "b"], ["a", "x", "b"])
        self.assertEqual(inserted["added"], ["x"])
        self.assertEqual(inserted["removed"], [])
        self.assertEqual(inserted["unchanged"], 2)
        deleted = diff_blocks(["a", "x", "b"], ["a", "b"])
        self.assertEqual(deleted["removed"], ["x"])
        self.assertEqual(deleted["added"], [])
        self.assertEqual(deleted["unchanged"], 2)

    def test_diff_replaced_pair(self):
        result = diff_blocks(["old text"], ["new text"])
        self.assertEqual(result["changed"], [{"old": "old text", "new": "new text"}])
        self.assertEqual(result["added"], [])
        self.assertEqual(result["removed"], [])


class AuditTests(unittest.TestCase):
    def test_identical_documents(self):
        report = audit_document_diff("A\n\nB", "A\n\nB")
        self.assertFalse(report["summary"]["changed"])
        self.assertEqual(report["summary"]["similarity"], 1.0)
        self.assertFalse(report["weights_changed"])

    def test_added_paragraph(self):
        report = audit_document_diff("A", "A\n\nB")
        self.assertTrue(report["summary"]["changed"])
        self.assertEqual(report["summary"]["added"], 1)
        self.assertEqual(report["changes"]["added"], ["B"])

    def test_removed_paragraph(self):
        report = audit_document_diff("A\n\nB\n\nC", "A\n\nC")
        self.assertEqual(report["summary"]["removed"], 1)
        self.assertEqual(report["changes"]["removed"], ["B"])

    def test_changed_paragraph(self):
        report = audit_document_diff("The price is 10.", "The price is 12.")
        self.assertEqual(report["summary"]["changed"], 1)
        self.assertEqual(report["changes"]["changed"][0]["old"], "The price is 10.")

    def test_labels_and_method(self):
        report = audit_document_diff("A", "B", old_label="v1", new_label="v2")
        self.assertEqual(report["old_label"], "v1")
        self.assertEqual(report["new_label"], "v2")
        self.assertEqual(report["method"], "deterministic_block_diff_v1")

    def test_empty_old(self):
        report = audit_document_diff("", "A\n\nB")
        self.assertEqual(report["summary"]["old_blocks"], 0)
        self.assertEqual(report["summary"]["added"], 2)
        self.assertEqual(report["summary"]["similarity"], 0.0)

    def test_both_empty(self):
        report = audit_document_diff("", "")
        self.assertFalse(report["summary"]["changed"])
        self.assertEqual(report["summary"]["similarity"], 1.0)

    def test_deterministic_output(self):
        first = audit_document_diff("A\n\nB", "A\n\nC")
        second = audit_document_diff("A\n\nB", "A\n\nC")
        self.assertEqual(first, second)

    def test_cli_json(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.txt"
            new = Path(tmp) / "new.txt"
            old.write_text("A\n\nB", encoding="utf-8")
            new.write_text("A\n\nC", encoding="utf-8")
            result = run_cli("--old", str(old), "--new", str(new), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["summary"]["changed"], 1)

    def test_cli_human_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = Path(tmp) / "old.txt"
            new = Path(tmp) / "new.txt"
            old.write_text("A", encoding="utf-8")
            new.write_text("B", encoding="utf-8")
            result = run_cli("--old", str(old), "--new", str(new))
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Document diff", result.stdout)

    def test_cli_missing_file(self):
        result = run_cli("--old", "does-not-exist.txt", "--new", "also-missing.txt")
        self.assertEqual(result.returncode, 2)
        self.assertIn("input error", result.stderr)


if __name__ == "__main__":
    unittest.main()
