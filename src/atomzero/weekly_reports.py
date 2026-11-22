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

    @staticmethod
    def _matches(entry: dict, needle: str) -> bool:
        if not needle:
            return True
        fields = ("id", "title", "name", "path", "generated", "exported_at")
        haystack = " ".join(
            str(entry.get(field, "") or "") for field in fields).lower()
        return needle in haystack

    def list(self, limit: int = 10, query: str = "",
             order: str = "newest") -> list[dict]:
        try:
            limit = max(1, min(int(limit), 100))
        except (TypeError, ValueError):
            limit = 10
        needle = str(query or "").strip().lower()
        order = str(order or "newest").strip().lower()
        if order not in ("newest", "oldest", "largest"):
            raise WeeklyReportError(
                "order must be newest, oldest or largest")

        def sort_key(entry: dict):
            if order == "largest":
                try:
                    size = int(entry.get("size_bytes", 0) or 0)
                except (TypeError, ValueError):
                    size = 0
                return (size, str(entry.get("exported_at", "")))
            return str(entry.get("exported_at", ""))

        with self._lock:
            self._reload()
            ordered = sorted(
                self.reports, key=sort_key,
                reverse=order in ("newest", "largest"))
            items = []
            for entry in ordered:
                if not self._matches(entry, needle):
                    continue
                item = dict(entry)
                raw = str(item.get("path", "") or "")
                item["exists"] = bool(raw) and Path(raw).is_file()
                items.append(item)
                if len(items) >= limit:
                    break
            return items

    def _keep_split(self, limit: int) -> tuple[list[dict], list[dict]]:
        try:
            limit = int(limit)
        except (TypeError, ValueError) as exc:
            raise WeeklyReportError("keep must be a positive integer") from exc
        if limit < 1:
            raise WeeklyReportError("keep must be a positive integer")
        with self._lock:
            self._reload()
            ordered = sorted(
                self.reports,
                key=lambda item: str(item.get("exported_at", "")),
                reverse=True)
            return ordered[:limit], ordered[limit:]

    def keep_preview(self, limit: int) -> list[dict]:
        """Return the entries ``keep(limit)`` would remove."""
        with self._lock:
            _, removed = self._keep_split(limit)
            return [dict(item) for item in removed]

    def keep(self, limit: int) -> int:
        """Keep the newest ``limit`` entries and remove older ones."""
        with self._lock:
            kept, removed = self._keep_split(limit)
            if removed:
                self.reports = kept
                self._save()
            return len(removed)

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

    def count(self, query: str = "") -> int:
        needle = str(query or "").strip().lower()
        with self._lock:
            self._reload()
            if not needle:
                return len(self.reports)
            return sum(1 for entry in self.reports
                       if self._matches(entry, needle))

    def verify(self) -> dict:
        """Compare stored size/SHA-256 values with the report files."""
        with self._lock:
            self._reload()
            entries = []
            counts = {"ok": 0, "missing": 0, "size_mismatch": 0,
                      "hash_mismatch": 0}
            for entry in self.reports:
                raw = str(entry.get("path", "") or "")
                stored_size = entry.get("size_bytes")
                try:
                    stored_size = int(stored_size)
                except (TypeError, ValueError):
                    stored_size = -1
                stored_sha = str(entry.get("sha256", "") or "")
                actual_size = None
                actual_sha = ""
                problems = []
                if not raw or not Path(raw).is_file():
                    problems.append("missing")
                else:
                    try:
                        data = Path(raw).read_bytes()
                    except OSError:
                        problems.append("missing")
                    else:
                        actual_size = len(data)
                        actual_sha = hashlib.sha256(data).hexdigest()
                        if actual_size != stored_size:
                            problems.append("size_mismatch")
                        if stored_sha and actual_sha != stored_sha:
                            problems.append("hash_mismatch")
                if not problems:
                    status = "ok"
                elif "missing" in problems:
                    status = "missing"
                else:
                    status = "failed"
                for problem in problems:
                    counts[problem] = counts.get(problem, 0) + 1
                if status == "ok":
                    counts["ok"] += 1
                item = dict(entry)
                item.update({
                    "status": status,
                    "problems": problems,
                    "actual_size_bytes": actual_size,
                    "actual_sha256": actual_sha,
                })
                entries.append(item)
            total = len(entries)
            summary = {
                "total": total,
                "ok": counts["ok"],
                "failed": total - counts["ok"],
                "missing": counts["missing"],
                "size_mismatch": counts["size_mismatch"],
                "hash_mismatch": counts["hash_mismatch"],
            }
            return {
                "verified_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "index": str(self.path),
                "summary": summary,
                "entries": entries,
                "ok": summary["failed"] == 0,
            }

    def repair(self, report_id: str = "") -> dict:
        """Recompute size/SHA-256 metadata for existing report files."""
        report_id = str(report_id or "").strip()
        stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
        with self._lock:
            self._reload()
            targets = self.reports
            if report_id:
                targets = [entry for entry in self.reports
                           if str(entry.get("id", "")) == report_id]
                if not targets:
                    raise WeeklyReportError(
                        f"weekly report not found: {report_id}")
            entries = []
            counts = {"repaired": 0, "ok": 0, "missing": 0}
            for entry in targets:
                raw = str(entry.get("path", "") or "")
                try:
                    old_size = int(entry.get("size_bytes"))
                except (TypeError, ValueError):
                    old_size = None
                old_sha = str(entry.get("sha256", "") or "")
                new_size = None
                new_sha = ""
                if not raw or not Path(raw).is_file():
                    status = "missing"
                else:
                    try:
                        data = Path(raw).read_bytes()
                    except OSError:
                        status = "missing"
                    else:
                        new_size = len(data)
                        new_sha = hashlib.sha256(data).hexdigest()
                        needs_update = (
                            new_size != old_size
                            or not old_sha
                            or new_sha != old_sha)
                        if needs_update:
                            entry["size_bytes"] = new_size
                            entry["sha256"] = new_sha
                            entry["metadata_repaired_at"] = stamp
                            status = "repaired"
                        else:
                            status = "ok"
                counts[status] += 1
                item = dict(entry)
                item.update({
                    "status": status,
                    "old_size_bytes": old_size,
                    "old_sha256": old_sha,
                    "actual_size_bytes": new_size,
                    "actual_sha256": new_sha,
                })
                entries.append(item)
            if counts["repaired"]:
                self._save()
            total = len(entries)
            summary = {
                "total": total,
                "repaired": counts["repaired"],
                "ok": counts["ok"],
                "missing": counts["missing"],
                "failed": counts["missing"],
            }
            return {
                "repaired_at": stamp,
                "index": str(self.path),
                "summary": summary,
                "entries": entries,
                "ok": summary["missing"] == 0,
            }

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

    def delete(self, report_id: str) -> dict:
        """Remove one index entry; the report file is left on disk."""
        report_id = str(report_id or "").strip()
        if not report_id:
            raise WeeklyReportError("report id is required")
        with self._lock:
            self._reload()
            deleted = None
            kept = []
            for entry in self.reports:
                if (deleted is None
                        and str(entry.get("id", "")) == report_id):
                    deleted = dict(entry)
                else:
                    kept.append(entry)
            if deleted is None:
                raise WeeklyReportError(
                    f"weekly report not found: {report_id}")
            self.reports = kept
            self._save()
            return deleted

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
