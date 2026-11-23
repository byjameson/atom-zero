"""End-to-end HTTP/UI regression for source comparison.

Checks the served compare controls, numeric/negation conflicts, identifier
conflicts and a no-conflict comparison.

Run:

    python benchmarks/contradiction_regression.py
"""
from __future__ import annotations

import argparse
import json
import tempfile
import threading
import time
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from atomzero import chat_server
from atomzero.doc_memory import DocMemory


def _get(base: str, route: str) -> dict:
    with urllib.request.urlopen(base + route, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-compare-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
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
        docs.add_text("price_a.txt", "The contract value is $1,500,000.",
                      title="Price A")
        docs.add_text("price_b.txt",
                      "The contract value is 1.5 million dollars.",
                      title="Price B")
        docs.add_text("price_c.txt", "The contract value is $1,300,000.",
                      title="Price C")
        docs.add_text("report_a.txt", "The launch is on 5 January 2020.",
                      title="Report A")
        docs.add_text("report_b.txt", "The launch is on January 5, 2020.",
                      title="Report B")
        docs.add_text("report_c.txt", "The launch is on 6 January 2020.",
                      title="Report C")
        docs.add_text("warranty_24.txt",
                      "The warranty period is 24 months.",
                      title="Warranty 24")
        docs.add_text("warranty_2y.txt",
                      "The warranty period is 2 years.",
                      title="Warranty 2y")
        docs.add_text("warranty_36.txt",
                      "The warranty period is 36 months.",
                      title="Warranty 36")
        docs.add_text("outage_days.txt", "The outage lasted 3 days.",
                      title="Outage days")
        docs.add_text("outage_hours.txt", "The outage lasted 72 hours.",
                      title="Outage hours")
        docs.add_text("trail_km.txt",
                      "The hiking trail length is 5 km.",
                      title="Trail km")
        docs.add_text("trail_m.txt",
                      "The hiking trail length is 5000 meters.",
                      title="Trail m")
        docs.add_text("trail_6km.txt",
                      "The hiking trail length is 6 km.",
                      title="Trail 6km")
        docs.add_text("unrelated.txt", "Bananas are yellow.",
                      title="Unrelated")
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "compare-test", "params": 1, "config": {}, "step": 0,
            "api_key": "", "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en",
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                html = response.read().decode("utf-8", "replace")
            controls_ok = (
                "/v1/documents/compare" in html
                and 'id="cmpbtn"' in html
                and "compareSources" in html)
            check("ui_compare_controls", controls_ok,
                  "present" if controls_ok else "missing")
            policy = _get(
                base,
                "/v1/documents/compare?source=policy_a.txt"
                "&against=policy_b.txt")
            reasons = {reason for conflict in policy["conflicts"]
                       for reason in conflict["reasons"]}
            check(
                "policy_conflict_reasons",
                "numeric_mismatch" in reasons
                and "negation_mismatch" in reasons,
                str(sorted(reasons)))
            snippets = " ".join(
                side["text"]
                for conflict in policy["conflicts"]
                for side in (conflict["left"], conflict["right"]))
            check(
                "policy_conflict_snippets",
                "24 months" in snippets and "36 months" in snippets
                and "does not cover" in snippets,
                snippets[:200])
            code = _get(
                base,
                "/v1/documents/compare?source=launch_a.txt"
                "&against=launch_b.txt")
            check(
                "identifier_conflict",
                any("identifier_mismatch" in conflict["reasons"]
                    for conflict in code["conflicts"]),
                str(code["summary"]))
            formatted = _get(
                base,
                "/v1/documents/compare?source=price_a.txt"
                "&against=price_b.txt")
            formatted_reasons = {reason for conflict in formatted["conflicts"]
                                 for reason in conflict["reasons"]}
            check(
                "numeric_formatting_no_false_positive",
                "numeric_mismatch" not in formatted_reasons,
                str(sorted(formatted_reasons)))
            scaled = _get(
                base,
                "/v1/documents/compare?source=price_a.txt"
                "&against=price_c.txt")
            check(
                "numeric_scale_conflict",
                any("numeric_mismatch" in conflict["reasons"]
                    for conflict in scaled["conflicts"]),
                str(scaled["summary"]))
            date_formatted = _get(
                base,
                "/v1/documents/compare?source=report_a.txt"
                "&against=report_b.txt")
            date_reasons = {reason for conflict in date_formatted["conflicts"]
                            for reason in conflict["reasons"]}
            check(
                "date_formatting_no_false_positive",
                "date_mismatch" not in date_reasons,
                str(sorted(date_reasons)))
            date_conflict = _get(
                base,
                "/v1/documents/compare?source=report_a.txt"
                "&against=report_c.txt")
            check(
                "date_conflict",
                any("date_mismatch" in conflict["reasons"]
                    for conflict in date_conflict["conflicts"]),
                str(date_conflict["summary"]))
            unit_pairs = (
                ("warranty_24.txt", "warranty_2y.txt"),
                ("outage_days.txt", "outage_hours.txt"),
                ("trail_km.txt", "trail_m.txt"),
            )
            unit_results = [
                _get(base,
                     f"/v1/documents/compare?source={left}&against={right}")
                for left, right in unit_pairs
            ]
            unit_reasons = {reason
                            for result in unit_results
                            for conflict in result["conflicts"]
                            for reason in conflict["reasons"]}
            check(
                "unit_formatting_no_false_positive",
                "numeric_mismatch" not in unit_reasons,
                str(sorted(unit_reasons)))
            unit_conflicts = [
                _get(base,
                     f"/v1/documents/compare?source={left}&against={right}")
                for left, right in (
                    ("warranty_24.txt", "warranty_36.txt"),
                    ("trail_km.txt", "trail_6km.txt"),
                )
            ]
            check(
                "unit_scale_conflict",
                all(any("numeric_mismatch" in conflict["reasons"]
                        for conflict in result["conflicts"])
                    for result in unit_conflicts),
                str([result["summary"] for result in unit_conflicts]))
            clean = _get(
                base,
                "/v1/documents/compare?source=unrelated.txt"
                "&against=policy_a.txt")
            check(
                "no_false_positive",
                clean["conflicts"] == []
                and clean["summary"]["conflict_count"] == 0,
                str(clean["summary"]))
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-contradiction-regression",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "checks": checks,
        "passed": len(checks) - failed,
        "failed": failed,
        "ok": failed == 0,
    }
    if output:
        target = Path(output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", default="benchmarks/results/contradiction_regression.json")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
    payload = run(args.out)
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        for item in payload["checks"]:
            status = "OK" if item["ok"] else "FAIL"
            print(f"[{status}] {item['name']}: {item['detail']}")
        print(f"summary: {payload['passed']} passed, "
              f"{payload['failed']} failed")
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
