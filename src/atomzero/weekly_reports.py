"""Local archive index for exported weekly-review reports.

The index stores metadata only; report files stay where the user wrote them.
Each output path is recorded once and refreshed on later exports to the same
path. JSON is read with BOM tolerance and written through a sibling temp file
plus atomic replace.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import threading
import time
from pathlib import Path

DEFAULT_REPORTS_PATH = "data/weekly_reports.json"


class WeeklyReportError(ValueError):
    pass


class WeeklyReportArchive:
    def __init__(self, path: str | Path = DEFAULT_REPORTS_PATH,
                 max_entries: int = 50):
        self.path = Path(path)
        try:
            self.max_entries = max(1, min(int(max_entries), 500))
        except (TypeError, ValueError):
            self.max_entries = 50
        self._lock = threading.RLock()
        self.reports = self._load()

    def _load(self) -> list[dict]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as exc:
            raise WeeklyReportError(
                f"cannot read weekly report index: {exc}") from exc
        if isinstance(data, dict):
            data = data.get("reports", [])
        if not isinstance(data, list):
            raise WeeklyReportError(
                "weekly report index must be a JSON list")
        return [item for item in data if isinstance(item, dict)]

    def _reload(self) -> None:
        self.reports = self._load()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            {"version": 1, "reports": self.reports},
            indent=2, ensure_ascii=False) + "\n"
        fd, temp_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp",
            dir=str(self.path.parent))
        temp = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                fd = -1
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.path)
        finally:
            if fd != -1:
                try:
                    os.close(fd)
                except OSError:
                    pass
            try:
                temp.unlink()
            except OSError:
                pass

    @staticmethod
    def _next_id(reports: list[dict]) -> str:
        highest = 0
        for item in reports:
            match = re.fullmatch(r"w(\d+)", str(item.get("id", "")))
            if match:
                highest = max(highest, int(match.group(1)))
        return f"w{highest + 1:04d}"

    def record(self, report_path: str | Path, review: dict,
               exported_at: str = "") -> dict:
        path = Path(report_path).expanduser()
        try:
            path = path.resolve()
        except OSError:
            path = path.absolute()
        if not path.is_file():
            raise WeeklyReportError(f"report file not found: {path}")
        data = path.read_bytes()
        counts = review.get("task_counts", {})
        graph = review.get("graph", {})
        generated = str(review.get("generated", "") or "")
        entry = {
            "id": "",
            "title": (f"Weekly review - {generated}" if generated
                      else path.name),
            "path": str(path),
            "name": path.name,
            "generated": generated,
            "exported_at": exported_at or time.strftime("%Y-%m-%dT%H:%M:%S"),
            "period_days": int(review.get("period_days", 7) or 7),
            "task_counts": counts if isinstance(counts, dict) else {},
            "pending_gap_count": int(
                review.get("pending_gap_count", 0) or 0),
            "recent_source_count": int(
                review.get("recent_source_count", 0) or 0),
            "entities": len(graph.get("entities", []))
            if isinstance(graph, dict) else 0,
            "connections": len(graph.get("connections", []))
            if isinstance(graph, dict) else 0,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
        key = str(path)
        with self._lock:
            self._reload()
            existing = next(
                (item for item in self.reports
                 if item.get("path") == key), None)
            self.reports = [item for item in self.reports
                            if item.get("path") != key]
            if existing and existing.get("id"):
                entry["id"] = str(existing["id"])
            else:
                entry["id"] = self._next_id(self.reports)
            self.reports.insert(0, entry)
            self.reports = self.reports[:self.max_entries]
            self._save()
        return dict(entry)

    def list(self, limit: int = 10) -> list[dict]:
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 10
        with self._lock:
            self._reload()
            items = []
            for entry in self.reports[:limit]:
                item = dict(entry)
                raw = str(item.get("path", "") or "")
                item["exists"] = bool(raw) and Path(raw).is_file()
                items.append(item)
            return items

    def prune(self) -> int:
        """Remove entries whose report file no longer exists."""
        with self._lock:
            self._reload()
            kept = []
            removed = 0
            for entry in self.reports:
                raw = str(entry.get("path", "") or "")
                if raw and Path(raw).is_file():
                    kept.append(entry)
                else:
                    removed += 1
            if removed:
                self.reports = kept
                self._save()
            return removed

    def count(self) -> int:
        with self._lock:
            self._reload()
            return len(self.reports)

    def get(self, report_id: str) -> dict:
        report_id = str(report_id or "").strip()
        if not report_id:
            raise WeeklyReportError("report id is required")
        with self._lock:
            self._reload()
            for entry in self.reports:
                if str(entry.get("id", "")) == report_id:
                    return dict(entry)
        raise WeeklyReportError(f"weekly report not found: {report_id}")

    def markdown(self, report_id: str) -> tuple[dict, str]:
        entry = self.get(report_id)
        path = Path(str(entry.get("path", "")))
        if not str(entry.get("path", "")) or not path.is_file():
            raise WeeklyReportError(
                f"report file not found: {entry.get('path', '')}")
        try:
            return entry, path.read_text(encoding="utf-8-sig")
        except OSError as exc:
            raise WeeklyReportError(f"cannot read report: {exc}") from exc
