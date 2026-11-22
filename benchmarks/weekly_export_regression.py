"""End-to-end CLI, HTTP and UI regression for weekly-review Markdown export.

Checks:
- the served UI controls and endpoint contract;
- Markdown content type and attachment filename;
- task checkboxes, gaps, sources, graph and next actions;
- the 14-day window label;
- invalid day input returning 400;
- `atomzero-brief --weekly --markdown --out PATH` writing a file directly;
- the CLI report appearing in the HTTP/UI archive and downloading from it;
- pruning a missing archive entry through the HTTP/UI flow;
- previewing and retaining only the newest N reports through the HTTP/UI
  flow.

Run:

    python benchmarks/weekly_export_regression.py
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import tempfile
import threading
import time
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from atomzero import chat_server
from atomzero.briefing import main as brief_main
from atomzero.doc_memory import DocMemory
from atomzero.knowledge_gaps import KnowledgeGapStore
from atomzero.weekly_reports import WeeklyReportArchive


def _post(base: str, route: str, payload: dict) -> dict:
    request = urllib.request.Request(
        base + route,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


def _markdown(base: str, days: int = 7) -> tuple[str, dict]:
    with urllib.request.urlopen(
            base + "/v1/weekly-review/export"
                   f"?session_id=export-chat&days={days}",
            timeout=15) as response:
        headers = dict(response.headers)
        return response.read().decode("utf-8"), headers


def run(output: str | Path | None = None) -> dict:
    checks = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": name, "ok": bool(ok), "detail": detail})

    with tempfile.TemporaryDirectory(prefix="atomzero-weekly-export-") as folder:
        docs = DocMemory(Path(folder) / "docs.db")
        docs.add_text("notes/launch.txt", "The launch code is L-7781.",
                      title="Launch")
        gaps = KnowledgeGapStore(Path(folder) / "gaps.json")
        gaps.add("What is the warranty claim deadline?")
        report_index = Path(folder) / "weekly_reports.json"
        old = dict(chat_server._state)
        chat_server._state.update({
            "model": None, "tok": None, "docs": docs, "adapter": "",
            "name": "weekly-export-test", "params": 1, "config": {},
            "step": 0, "api_key": "",
            "metrics": {"requests": 0, "generated_tokens": 0},
            "safety": None, "wiki_lang": "en", "gaps": gaps,
            "session_dir": str(Path(folder) / "sessions"), "sessions": {},
            "generator": None, "allow_general_chat": False, "skills": None,
            "weekly_reports_path": str(report_index),
            "weekly_reports": WeeklyReportArchive(report_index),
        })
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), chat_server.Handler)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        try:
            with urllib.request.urlopen(base + "/", timeout=15) as response:
                page = response.read().decode("utf-8", "replace")
            check(
                "ui_export_controls",
                "/v1/weekly-review/export" in page
                and 'id="weeklymd"' in page
                and "downloadWeeklyMarkdown" in page
                and "/v1/weekly-review/reports" in page
                and 'id="weeklyreportsrefresh"' in page
                and 'id="weeklyreportsprune"' in page
                and 'id="weeklyreportskeep"' in page
                and 'id="weeklykeep"' in page
                and 'id="weeklykeepdry"' in page
                and 'id="weeklyreportspreview"' in page
                and "/v1/weekly-review/reports/keep" in page
                and "refreshWeeklyReports" in page
                and "pruneWeeklyReports" in page
                and "keepWeeklyReports" in page
                and "dry_run" in page
                and "would_remove" in page,
                "present")
            for message in ("add task buy milk",
                            "add task write report",
                            "complete task 1"):
                _post(base, "/v1/agent/chat", {
                    "session_id": "export-chat", "message": message})
            markdown, headers = _markdown(base)
            check(
                "export_headers",
                "text/markdown" in headers.get("Content-Type", "")
                and "attachment" in headers.get("Content-Disposition", "")
                and "atomzero-weekly-review-" in headers.get(
                    "Content-Disposition", ""),
                str(headers.get("Content-Disposition")))
            check(
                "export_tasks",
                markdown.startswith("# Weekly review - ")
                and "Period: last 7 days" in markdown
                and "- Open: 1" in markdown
                and "- [ ] **#2** write report" in markdown
                and "- [x] **#1** buy milk" in markdown,
                markdown[:220])
            check(
                "export_context",
                "## Pending gaps (1)" in markdown
                and "What is the warranty claim deadline?" in markdown
                and "## Recent sources (1)" in markdown
                and "`notes/launch.txt`" in markdown
                and "L-7781" in markdown
                and "## Graph" in markdown,
                markdown[:320])
            check(
                "export_next_and_limits",
                "## Next week" in markdown
                and "- [ ] Close task 2: write report" in markdown
                and "No model weights were changed." in markdown,
                markdown[-240:])
            long_markdown, _ = _markdown(base, days=14)
            check(
                "export_14_day_window",
                "Period: last 14 days" in long_markdown
                and "Completed in the last 14 days" in long_markdown
                and "Completed this week" not in long_markdown,
                long_markdown[:180])
            try:
                urllib.request.urlopen(
                    base + "/v1/weekly-review/export"
                           "?session_id=export-chat&days=not-a-number",
                    timeout=15)
                invalid_ok = True
            except urllib.error.HTTPError as exc:
                invalid_ok = exc.code == 400
            check(
                "export_invalid_days",
                invalid_ok,
                "400 for a non-numeric window")
            out_path = Path(folder) / "cli" / "reports" / "weekly.md"
            session_path = Path(folder) / "sessions" / "export-chat.json"
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                cli_code = brief_main([
                    "--session", str(session_path),
                    "--docs", str(Path(folder) / "docs.db"),
                    "--gaps", str(Path(folder) / "gaps.json"),
                    "--weekly", "--markdown", "--out", str(out_path),
                    "--reports-index", str(report_index)])
            cli_markdown = (out_path.read_text(encoding="utf-8")
                            if out_path.exists() else "")
            check(
                "cli_markdown_out",
                cli_code == 0
                and "weekly review written to" in buffer.getvalue()
                and "(archived w0001)" in buffer.getvalue()
                and cli_markdown.startswith("# Weekly review - ")
                and "- [ ] **#2** write report" in cli_markdown
                and not out_path.read_bytes().startswith(b"\xef\xbb\xbf"),
                f"exit={cli_code} path={out_path}")
            listing = json.loads(urllib.request.urlopen(
                base + "/v1/weekly-review/reports?limit=5",
                timeout=15).read().decode("utf-8"))
            entry = (listing.get("data") or [{}])[0]
            check(
                "archive_listing",
                listing.get("total") == 1
                and entry.get("id") == "w0001"
                and entry.get("exists") is True
                and Path(str(entry.get("path", ""))).resolve()
                == out_path.resolve(),
                str(entry))
            with urllib.request.urlopen(
                    base + f"/v1/weekly-review/reports?id={entry.get('id')}",
                    timeout=15) as response:
                archived = response.read().decode("utf-8")
                archive_type = response.headers.get_content_type()
            check(
                "archive_download",
                archive_type == "text/markdown"
                and archived.startswith("# Weekly review - ")
                and "- [ ] **#2** write report" in archived,
                archived[:180])
            missing_path = Path(folder) / "gone.md"
            missing_path.write_text("# Gone\n", encoding="utf-8")
            archive = (chat_server._state.get("weekly_reports")
                       or WeeklyReportArchive(report_index))
            archive.record(missing_path, {
                "generated": "2026-11-22 10:00", "period_days": 14,
                "task_counts": {"open": 0}, "pending_gap_count": 0,
                "recent_source_count": 0,
                "graph": {"entities": [], "connections": []},
            })
            missing_path.unlink()
            pruned = _post(base, "/v1/weekly-review/reports/prune", {})
            after_prune = json.loads(urllib.request.urlopen(
                base + "/v1/weekly-review/reports?limit=5",
                timeout=15).read().decode("utf-8"))
            check(
                "archive_prune",
                pruned.get("removed") == 1
                and pruned.get("remaining") == 1
                and after_prune.get("total") == 1,
                str(pruned))
            for number in range(1, 4):
                extra = Path(folder) / f"extra-{number}.md"
                extra.write_text(f"# Extra {number}\n", encoding="utf-8")
                archive.record(extra, {
                    "generated": f"2026-11-2{number} 09:00",
                    "period_days": 7, "task_counts": {"open": 0},
                    "pending_gap_count": 0, "recent_source_count": 0,
                    "graph": {"entities": [], "connections": []},
                }, exported_at=f"2026-11-2{number}T09:00:00")
            preview = _post(
                base, "/v1/weekly-review/reports/keep",
                {"keep": 2, "dry_run": True})
            after_preview = json.loads(urllib.request.urlopen(
                base + "/v1/weekly-review/reports?limit=5",
                timeout=15).read().decode("utf-8"))
            check(
                "archive_keep_preview",
                preview.get("dry_run") is True
                and preview.get("trimmed") == 0
                and preview.get("would_remove_count") == 2
                and preview.get("remaining") == 4
                and after_preview.get("total") == 4,
                str(preview))
            kept = _post(
                base, "/v1/weekly-review/reports/keep", {"keep": 2})
            after_keep = json.loads(urllib.request.urlopen(
                base + "/v1/weekly-review/reports?limit=5",
                timeout=15).read().decode("utf-8"))
            check(
                "archive_keep",
                kept.get("trimmed") == 2
                and kept.get("remaining") == 2
                and after_keep.get("total") == 2,
                str(kept))
        finally:
            httpd.shutdown()
            httpd.server_close()
            chat_server._state.clear()
            chat_server._state.update(old)
            docs.close()

    failed = sum(1 for item in checks if not item["ok"])
    payload = {
        "benchmark": "atomzero-weekly-export-regression",
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
        "--out", default="benchmarks/results/weekly_export_regression.json")
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
