import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from atomzero.consensus import (
    audit_consensus,
    cluster_claims,
    consensus_from_rows,
)

ROOT = Path(__file__).resolve().parents[1]


def run_cli(*args: str, stdin: str | None = None):
    return subprocess.run(
        [sys.executable, "-m", "atomzero.consensus", *args],
        cwd=ROOT,
        input=stdin,
        capture_output=True,
        text=True,
        encoding="utf-8",
        env={"PYTHONPATH": str(ROOT / "src"), "PYTHONIOENCODING": "utf-8"},
        check=False,
    )


class ConsensusTests(unittest.TestCase):
    def test_same_claim_two_sources_is_consensus(self):
        report = audit_consensus([
            {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
            {"source": "b.txt", "chunk": "The battery lasts 8 hours."},
        ])
        self.assertEqual(report["summary"]["claim_count"], 2)
        self.assertEqual(report["summary"]["cluster_count"], 1)
        self.assertEqual(report["summary"]["consensus_clusters"], 1)
        cluster = report["clusters"][0]
        self.assertEqual(cluster["status"], "consensus")
        self.assertEqual(cluster["sources"], ["a.txt", "b.txt"])
        self.assertEqual(cluster["source_count"], 2)
        self.assertFalse(report["weights_changed"])
        self.assertEqual(
            report["method"], "deterministic_cross_source_consensus_v1")

    def test_single_source_is_not_consensus(self):
        report = audit_consensus([
            {"source": "warranty.txt", "chunk": "The warranty code is W-7781."},
        ])
        self.assertEqual(report["summary"]["consensus_clusters"], 0)
        self.assertEqual(report["summary"]["single_source_clusters"], 1)
        self.assertEqual(report["clusters"][0]["status"], "single_source")

    def test_numeric_conflict_is_visible_and_not_merged(self):
        report = audit_consensus([
            {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
            {"source": "b.txt", "chunk": "The battery lasts 10 hours."},
        ])
        self.assertEqual(report["summary"]["cluster_count"], 2)
        self.assertEqual(report["summary"]["consensus_clusters"], 0)
        self.assertEqual(report["conflict_count"], 1)
        self.assertEqual(
            report["conflicts"][0]["reasons"], ["numeric_mismatch"])

    def test_unrelated_claims_stay_separate(self):
        report = audit_consensus([
            {"source": "a.txt", "chunk": "The capital of France is Paris."},
            {"source": "b.txt", "chunk": "The warranty code is W-7781."},
        ])
        self.assertEqual(report["summary"]["cluster_count"], 2)
        self.assertEqual(report["summary"]["consensus_clusters"], 0)

    def test_same_source_repeated_claim_is_single_source(self):
        report = audit_consensus([
            {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
            {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
        ])
        self.assertEqual(report["summary"]["cluster_count"], 1)
        self.assertEqual(report["summary"]["consensus_clusters"], 0)
        self.assertEqual(report["clusters"][0]["source_count"], 1)

    def test_empty_hits(self):
        report = audit_consensus([])
        self.assertEqual(report["summary"]["claim_count"], 0)
        self.assertEqual(report["clusters"], [])
        self.assertEqual(report["conflict_count"], 0)

    def test_cluster_claims_rejects_bad_overlap(self):
        with self.assertRaises(ValueError):
            cluster_claims([], min_overlap=2.0)

    def test_min_sources_one_marks_every_cluster(self):
        report = audit_consensus([
            {"source": "a.txt", "chunk": "The warranty code is W-7781."},
        ], min_sources=1)
        self.assertEqual(report["clusters"][0]["status"], "consensus")

    def test_consensus_from_rows_is_deterministic(self):
        rows = [
            {"claim_id": "b", "text": "The battery lasts 8 hours.",
             "start": 0, "source": "b.txt"},
            {"claim_id": "a", "text": "The battery lasts 8 hours.",
             "start": 0, "source": "a.txt"},
        ]
        first = consensus_from_rows(rows)
        second = consensus_from_rows(list(reversed(rows)))
        self.assertEqual(first, second)
        self.assertEqual(first["clusters"][0]["sources"], ["a.txt", "b.txt"])

    def test_cli_json_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "hits.json"
            path.write_text(json.dumps([
                {"source": "a.txt", "chunk": "The battery lasts 8 hours."},
                {"source": "b.txt", "chunk": "The battery lasts 8 hours."},
            ]), encoding="utf-8")
            result = run_cli("--file", str(path), "--json")
        self.assertEqual(result.returncode, 0, result.stderr)
        data = json.loads(result.stdout)
        self.assertEqual(data["summary"]["consensus_clusters"], 1)

    def test_cli_stdin(self):
        result = run_cli(
            "--json",
            stdin=json.dumps([
                {"source": "a.txt", "chunk": "The warranty code is W-7781."},
            ]),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(
            json.loads(result.stdout)["summary"]["claim_count"], 1)

    def test_cli_empty_exit_two(self):
        result = run_cli("--json", stdin="")
        self.assertEqual(result.returncode, 2)
        self.assertIn("no hits", result.stderr)


if __name__ == "__main__":
    unittest.main()
