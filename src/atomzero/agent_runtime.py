"""Session-based agent runtime around the tiny model and local tools."""
from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from atomzero.briefing import (
    build_briefing, build_weekly_review, due_soon_view, overdue_view,
    normalize_tag_rename, plan_goal, render_due_soon_view,
    render_overdue_view, render_task_search_view, render_task_tag_view,
    render_weekly_report_comparison, task_search_view, task_tag_view)
from atomzero.contradictions import compare_sources
from atomzero.doc_memory import format_citation, grounded_answer
from atomzero.knowledge_gaps import KnowledgeGapError
from atomzero.knowledge_graph import entity_sources, graph_path
from atomzero.skills import execute_skill
from atomzero.weekly_reports import (
    DEFAULT_REPORTS_PATH, WeeklyReportArchive, WeeklyReportError)
from atomzero.web_research import wikipedia_research

from atomzero.chat_human import ChatHuman

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_MATH_PREFIX = re.compile(
    r"^(?:what\s+is|what's|calculate|compute|calc)\s+", re.IGNORECASE)
_MATH_CHARS = re.compile(r"[0-9+\-*/().%\s]+")


def _eval_math(node):
    if isinstance(node, ast.Expression):
        return _eval_math(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left = _eval_math(node.left)
        right = _eval_math(node.right)
        if isinstance(node.op, ast.Pow) and (abs(left) > 1e6 or abs(right) > 100):
            raise ValueError("exponent is too large")
        value = _BIN_OPS[type(node.op)](left, right)
        if abs(value) > 10**12:
            raise ValueError("result is too large")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_math(node.operand))
    raise ValueError("unsupported expression")


def calculate(expression: str) -> str:
    """Evaluate a small arithmetic expression without eval()."""
    expr = expression.strip().rstrip("=?").strip()
    if not expr or len(expr) > 80:
        raise ValueError("invalid expression")
    if not _MATH_CHARS.fullmatch(expr):
        raise ValueError("only numbers and + - * / % ** are allowed")
    value = _eval_math(ast.parse(expr, mode="eval"))
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def math_query(text: str) -> str | None:
    candidate = _MATH_PREFIX.sub("", text.strip()).strip()
    candidate = candidate.rstrip("=?").strip()
    if not candidate or not re.search(r"\d", candidate):
        return None
    if not _MATH_CHARS.fullmatch(candidate):
        return None
    if not re.search(r"[+\-*/%]", candidate):
        return None
    return candidate


def parse_task_search_filters(raw: str) -> tuple[str, str, str, str, str, str]:
    """Split trailing task-search filters from the text query."""
    tokens = str(raw or "").strip().split()
    status = "all"
    due = "all"
    priority = "all"
    sort = "default"
    tag = "all"
    status_words = {"open", "done"}
    due_words = {"overdue", "today", "week", "none"}
    priority_words = {"low", "normal", "high"}
    sort_words = {"default", "priority"}
    while tokens:
        last = tokens[-1].lower()
        if (len(tokens) >= 3 and tokens[-3].lower() == "sorted"
                and tokens[-2].lower() == "by" and last in sort_words):
            sort = last
            tokens = tokens[:-3]
            continue
        if (len(tokens) >= 2 and tokens[-2].lower() == "sort"
                and last in sort_words):
            sort = last
            tokens = tokens[:-2]
            continue
        if (len(tokens) >= 2 and tokens[-2].lower() == "priority"
                and last in priority_words):
            priority = last
            tokens = tokens[:-2]
            continue
        if (len(tokens) >= 2 and tokens[-2].lower() in {"tag", "tags"}):
            tag = last.lstrip("#")
            tokens = tokens[:-2]
            continue
        if (len(tokens) >= 2 and tokens[-2].lower() == "status"
                and last in status_words):
            status = last
            tokens = tokens[:-2]
            continue
        if (len(tokens) >= 2 and tokens[-2].lower() == "due"
                and last in due_words):
            due = last
            tokens = tokens[:-2]
            continue
        if last in priority_words and priority == "all":
            priority = last
            tokens.pop()
            continue
        if last in status_words and status == "all":
            status = last
            tokens.pop()
            continue
        if last in due_words and due == "all":
            due = last
            tokens.pop()
            continue
        if last.startswith("#") and len(last) > 1:
            tag = last[1:]
            tokens.pop()
            continue
        break
    return " ".join(tokens).strip(), status, due, priority, sort, tag


class AgentSession:
    def __init__(self, session_id: str, memory_path: str | Path,
                 model=None, tokenizer=None, device=None, docs=None,
                 wiki_lang: str = "en", gap_store=None, safety=None,
                 generator=None, allow_general_chat=False, skills=None,
                 weekly_reports_path: str = DEFAULT_REPORTS_PATH):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("invalid session id")
        self.id = session_id
        self.docs = docs
        self.wiki_lang = wiki_lang
        self.gap_store = gap_store
        self.safety = safety
        self.skills = skills
        self.weekly_reports_path = str(
            weekly_reports_path or DEFAULT_REPORTS_PATH)
        self.pending_plan = None
        self.pending_report_action = None
        self.pending_task_action = None
        self.human = ChatHuman(memory_path, model, tokenizer, device,
                               "transformer", docs, on_gap=self._record_gap,
                               generator=generator,
                               allow_general_chat=allow_general_chat)

    def _record_gap(self, text: str) -> None:
        if self.gap_store is None:
            return
        try:
            self.gap_store.add(text, session_id=self.id)
        except (KnowledgeGapError, OSError):
            pass

    def forget(self, text: str = "") -> dict:
        """Delete matching session memory, or all of it when text is empty."""
        return self.human.forget(text)

    def export_memory(self) -> dict:
        return self.human.export_memory()

    def import_memory(self, data: dict) -> dict:
        return self.human.import_memory(data)

    def _research(self, query: str, limit: int = 2) -> dict:
        if self.docs is None:
            raise ValueError("no document memory configured")
        pages = wikipedia_research(self.docs, query, self.wiki_lang, limit)
        if not pages:
            return {"reply": "I could not find a useful page.",
                    "tool_calls": [{"name": "research", "query": query}],
                    "sources": []}
        lines = "\n".join(
            f"- {p['title']} ({p['source']})" for p in pages)
        return {
            "reply": f"I indexed these pages:\n{lines}\n\n"
                     "Ask me about them and I will quote the notes.",
            "tool_calls": [{"name": "research", "query": query}],
            "sources": [{"source": p["source"], "title": p["title"]}
                        for p in pages],
        }

    def _search_docs(self, query: str) -> dict:
        hits = self.docs.search(query, k=6) if self.docs else []
        if not hits:
            return {"reply": "I found no matching local notes.",
                    "tool_calls": [{"name": "search_docs", "query": query}],
                    "sources": [], "evidence": [],
                    "confidence": {"score": 0.0, "label": "none",
                                   "coverage": 0.0, "margin": 0.0}}
        answer = grounded_answer(query, hits)
        return {
            "reply": answer["text"],
            "tool_calls": [{"name": "search_docs", "query": query}],
            "sources": answer["sources"],
            "evidence": answer["evidence"],
            "confidence": answer["confidence"],
        }

    def _graph_entity(self, name: str) -> dict:
        if self.docs is None:
            return {"reply": "no document memory configured",
                    "tool_calls": [], "sources": []}
        try:
            data = entity_sources(self.docs, name, limit=5)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "connections": []}
        tool = [{"name": "graph_entity", "entity": name}]
        if not data["sources"]:
            return {"reply": f"I found no local evidence for {name}.",
                    "tool_calls": tool, "sources": [],
                    "connections": []}
        lines = []
        sources = []
        for item in data["sources"]:
            label = ("teacher note" if item.get("kind") == "teacher"
                     else "source")
            lines.append(f"- {item.get('snippet', '')} "
                         f"({label}: {format_citation(item)})")
            source = {
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "kind": item.get("kind", "document"),
                "verified": bool(item.get("verified", True)),
            }
            if source not in sources:
                sources.append(source)
        reply = f"Local graph evidence for {name}:\n" + "\n".join(lines)
        connections = [
            f"{edge['source']} ↔ {edge['target']}"
            for edge in data.get("connections", [])[:8]]
        if connections:
            reply += "\n\nConnections: " + "; ".join(connections)
        reply += ("\n\nThis is local co-occurrence evidence, not a semantic "
                  "relation claim.")
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": sources,
            "evidence": data["sources"],
            "connections": data.get("connections", []),
        }

    def _graph_path(self, source: str, target: str) -> dict:
        if self.docs is None:
            return {"reply": "no document memory configured",
                    "tool_calls": [], "sources": [], "evidence": []}
        try:
            data = graph_path(self.docs, source, target)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        tool = [{"name": "graph_path", "source": source,
                 "target": target}]
        if not data.get("found"):
            return {
                "reply": (f"I found no local co-occurrence path from "
                          f"{source} to {target}: "
                          f"{data.get('reason', 'not found')}."),
                "tool_calls": tool,
                "sources": [],
                "evidence": [],
                "path": data,
            }
        lines = [f"Local graph path ({data['hops']} hop(s)):",
                 " → ".join(data["path"])]
        for edge in data["edges"]:
            lines.append(
                f"- {edge['source']} ↔ {edge['target']} "
                f"({edge['sources']} source(s))")
        lines.append("\nThis is co-occurrence navigation, not a semantic "
                     "relation claim.")
        sources = []
        evidence = []
        for name in data["path"]:
            try:
                info = entity_sources(self.docs, name, limit=1)
            except ValueError:
                continue
            for item in info.get("sources", []):
                source_item = {
                    "source": item.get("source", ""),
                    "title": item.get("title", ""),
                    "kind": item.get("kind", "document"),
                    "verified": bool(item.get("verified", True)),
                }
                if source_item not in sources:
                    sources.append(source_item)
                evidence.append({
                    "source": item.get("source", ""),
                    "title": item.get("title", ""),
                    "page": item.get("page"),
                    "section": item.get("section", ""),
                    "snippet": item.get("snippet", ""),
                })
        return {
            "reply": "\n".join(lines),
            "tool_calls": tool,
            "sources": sources,
            "evidence": evidence,
            "path": data,
        }

    def _compare_sources(self, source: str, against: str) -> dict:
        if self.docs is None:
            return {"reply": "no document memory configured",
                    "tool_calls": [], "sources": [], "evidence": []}
        source = source.strip().strip("\"'")
        against = against.strip().strip("\"'")
        try:
            comparison = compare_sources(self.docs, source, against)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        tool = [{"name": "compare_sources", "source": source,
                 "against": against}]
        conflicts = comparison.get("conflicts", [])
        sources = []
        evidence = []
        if conflicts:
            lines = [
                f"Found {len(conflicts)} conflict hint(s) between "
                f"{source} and {against}:"
            ]
            for conflict in conflicts[:6]:
                left = conflict["left"]
                right = conflict["right"]
                reasons = ", ".join(conflict.get("reasons", []))
                lines.append(
                    f"- {reasons}: \"{left['text'][:180]}\" "
                    f"({left['source']}) ↔ \"{right['text'][:180]}\" "
                    f"({right['source']})")
                for side in (left, right):
                    source_item = {
                        "source": side.get("source", ""),
                        "title": side.get("title", ""),
                        "page": side.get("page"),
                        "section": side.get("section", ""),
                    }
                    if source_item not in sources:
                        sources.append(source_item)
                    edge = {
                        "source": side.get("source", ""),
                        "title": side.get("title", ""),
                        "page": side.get("page"),
                        "section": side.get("section", ""),
                        "snippet": side.get("text", ""),
                    }
                    if edge not in evidence:
                        evidence.append(edge)
            lines.append("\nThis is a comparison hint, not a truth verdict.")
            reply = "\n".join(lines)
        else:
            reply = (
                f"No conflicting overlapping claims found between {source} "
                f"and {against} (numbers, identifiers, negation).")
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": sources,
            "evidence": evidence,
            "comparison": comparison,
        }

    def _briefing(self, sort: str = "default",
                  tag: str = "all") -> dict:
        try:
            brief = build_briefing(
                self.human.data, self.gap_store, self.docs,
                sort=sort, tag=tag)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        sources = brief.get("documents", {}).get("items", [])
        tool = {"name": "briefing", "sort": brief["sort"]}
        if brief.get("tag", "all") != "all":
            tool["tag"] = brief["tag"]
        return {
            "reply": brief["text"],
            "tool_calls": [tool],
            "sources": sources,
            "brief": brief,
            "weights_changed": False,
        }

    def _search_weekly_reports(self, query: str,
                               order: str = "newest") -> dict:
        query = str(query or "").strip()
        order = str(order or "newest").strip().lower()
        tool = [{"name": "search_weekly_reports", "query": query,
                 "sort": order}]
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            reports = archive.list(limit=10, query=query, order=order)
            total = archive.count()
            matched = archive.count(query=query)
        except (WeeklyReportError, OSError) as exc:
            return {
                "reply": f"report archive error: {exc}",
                "tool_calls": tool,
                "sources": [],
                "evidence": [],
                "reports": [],
                "query": query,
            }
        tool[0]["matches"] = len(reports)
        if not reports:
            if total == 0:
                reply = ("No archived weekly reports yet. Create one with "
                         "atomzero-brief --weekly --markdown --out PATH.")
            else:
                reply = (f'No archived weekly reports match "{query}" '
                         f"({total} total).")
        else:
            lines = [
                f'Archived weekly reports matching "{query}" '
                f"({matched} of {total}) | sort: {order}:"
            ]
            for item in reports:
                marker = "" if item.get("exists", True) else " [missing]"
                lines.append(
                    f"- {item.get('id', '?')} | "
                    f"exported {item.get('exported_at', '')} | "
                    f"last {item.get('period_days', 7)} days | "
                    f"{item.get('path', '')}{marker}")
            lines.append(
                "Report files stay on your disk; use "
                "atomzero-brief --reports --delete ID to remove an index entry.")
            reply = "\n".join(lines)
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "reports": reports,
            "query": query,
            "matched": matched,
            "total": total,
            "sort": order,
        }

    def _list_weekly_reports(self, limit: int = 10,
                             order: str = "newest") -> dict:
        try:
            limit = int(limit)
        except (TypeError, ValueError):
            return {"reply": "Report limit must be a positive integer.",
                    "tool_calls": [], "sources": [], "evidence": []}
        if limit < 1:
            return {"reply": "Report limit must be at least 1.",
                    "tool_calls": [], "sources": [], "evidence": []}
        limit = min(limit, 100)
        order = str(order or "newest").strip().lower()
        tool = [{"name": "list_weekly_reports", "limit": limit,
                 "sort": order}]
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            reports = archive.list(limit=limit, order=order)
            total = archive.count()
        except (WeeklyReportError, OSError) as exc:
            return {
                "reply": f"report archive error: {exc}",
                "tool_calls": tool,
                "sources": [],
                "evidence": [],
                "reports": [],
                "limit": limit,
                "sort": order,
            }
        tool[0]["matches"] = len(reports)
        if not reports:
            reply = ("No archived weekly reports yet. Create one with "
                     "atomzero-brief --weekly --markdown --out PATH.")
        else:
            lines = [f"Archived weekly reports ({len(reports)} of {total}) "
                     f"| sort: {order}:"]
            for item in reports:
                marker = "" if item.get("exists", True) else " [missing]"
                lines.append(
                    f"- {item.get('id', '?')} | "
                    f"exported {item.get('exported_at', '')} | "
                    f"last {item.get('period_days', 7)} days | "
                    f"{item.get('path', '')}{marker}")
            lines.append(
                "Report files stay on your disk; use "
                "atomzero-brief --reports --delete ID to remove an index entry.")
            reply = "\n".join(lines)
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "reports": reports,
            "limit": limit,
            "total": total,
            "sort": order,
        }

    def _compare_weekly_reports(self, left_id: str, right_id: str) -> dict:
        tool = [{"name": "compare_weekly_reports",
                 "left": left_id, "right": right_id}]
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            comparison = archive.compare(left_id, right_id)
        except (WeeklyReportError, OSError) as exc:
            return {"reply": str(exc), "tool_calls": tool,
                    "sources": [], "evidence": [], "comparison": None}
        tool[0]["hash_equal"] = comparison["hash_equal"]
        return {
            "reply": render_weekly_report_comparison(comparison),
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "comparison": comparison,
        }

    def _show_weekly_report(self, report_id: str) -> dict:
        report_id = str(report_id or "").strip()
        tool = [{"name": "show_weekly_report", "id": report_id}]
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            entry, markdown = archive.markdown(report_id)
        except (WeeklyReportError, OSError) as exc:
            return {"reply": str(exc), "tool_calls": tool,
                    "sources": [], "evidence": [], "reports": []}
        tool[0]["path"] = entry.get("path", "")
        body = markdown.strip()
        truncated = len(body) > 6000
        if truncated:
            body = body[:6000].rstrip() + "\n\n[truncated]"
        if not body:
            body = "(empty report)"
        reply = "\n".join([
            f"Report {entry.get('id', report_id)} | "
            f"exported {entry.get('exported_at', '')} | "
            f"{entry.get('path', '')}",
            "",
            body,
        ])
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "reports": [entry],
            "report_id": entry.get("id", report_id),
            "report_markdown": body,
            "truncated": truncated,
        }

    def _verify_weekly_reports(self, report_id: str = "") -> dict:
        report_id = str(report_id or "").strip()
        tool = [{"name": "verify_weekly_reports",
                 "scope": "report" if report_id else "archive"}]
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            result = archive.verify()
        except (WeeklyReportError, OSError) as exc:
            return {"reply": f"report archive error: {exc}",
                    "tool_calls": tool, "sources": [], "evidence": []}
        if report_id:
            entry = next(
                (item for item in result["entries"]
                 if str(item.get("id", "")) == report_id), None)
            if entry is None:
                return {"reply": f"weekly report not found: {report_id}",
                        "tool_calls": tool, "sources": [], "evidence": []}
            problems = list(entry.get("problems", []))
            summary = {
                "total": 1, "ok": 0, "failed": 0, "missing": 0,
                "size_mismatch": 0, "hash_mismatch": 0,
            }
            if entry["status"] == "ok":
                summary["ok"] = 1
            else:
                summary["failed"] = 1
            for problem in problems:
                summary[problem] = summary.get(problem, 0) + 1
            verification = {
                "verified_at": result["verified_at"],
                "index": result["index"],
                "summary": summary,
                "entries": [entry],
                "ok": entry["status"] == "ok",
            }
            tool[0].update({"id": report_id, "status": entry["status"],
                            "problems": problems})
            lines = [
                f"Report {report_id} integrity: {entry['status']}",
                f"- {entry.get('path', '')}",
                f"Stored size: {entry.get('size_bytes')} | "
                f"actual: {entry.get('actual_size_bytes')}",
                f"Stored SHA-256: {entry.get('sha256', '')}",
                f"Actual SHA-256: {entry.get('actual_sha256', '') or '-'}",
            ]
            if problems:
                lines.append("Problems: " + ", ".join(problems))
            return {
                "reply": "\n".join(lines),
                "tool_calls": tool,
                "sources": [],
                "evidence": [],
                "verification": verification,
                "reports": [entry],
                "report_id": report_id,
                "ok": entry["status"] == "ok",
            }
        summary = result["summary"]
        tool[0]["summary"] = summary
        if summary["failed"] == 0:
            reply = (f"Report integrity: all {summary['total']} report(s) "
                     "match stored size and SHA-256.")
        else:
            lines = [
                f"Report integrity: {summary['ok']} ok, "
                f"{summary['failed']} failed, {summary['missing']} missing "
                f"({summary['total']} total)"
            ]
            for entry in result["entries"]:
                if entry["status"] == "ok":
                    continue
                problems = ", ".join(entry.get("problems", []))
                lines.append(
                    f"- {entry.get('id', '?')} | {entry['status']} | "
                    f"{entry.get('path', '')} [{problems}]")
            reply = "\n".join(lines)
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "verification": result,
            "reports": result["entries"],
            "ok": result["ok"],
        }

    @staticmethod
    def _report_action_lines(entries: list[dict]) -> str:
        lines = []
        for item in entries:
            marker = "" if item.get("exists", True) else " [missing]"
            lines.append(
                f"- {item.get('id', '?')} | "
                f"{item.get('exported_at', '')} | "
                f"{item.get('path', '')}{marker}")
        return "\n".join(lines)

    def _preview_repair_reports(self, report_id: str = "") -> dict:
        self.pending_report_action = None
        report_id = str(report_id or "").strip()
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            verification = archive.verify()
        except (WeeklyReportError, OSError) as exc:
            return {"reply": f"report archive error: {exc}",
                    "tool_calls": [], "sources": [], "evidence": []}
        entries = verification["entries"]
        if report_id:
            entries = [item for item in entries
                       if str(item.get("id", "")) == report_id]
            if not entries:
                return {"reply": f"weekly report not found: {report_id}",
                        "tool_calls": [], "sources": [], "evidence": []}
        repairable = [item for item in entries
                      if item["status"] == "failed"]
        missing = [item for item in entries
                   if item["status"] == "missing"]
        tool = [{"name": "preview_weekly_reports", "action": "repair",
                 "repairable": len(repairable), "missing": len(missing)}]
        if report_id:
            tool[0]["id"] = report_id
            if not repairable:
                if missing:
                    reply = (f"Report {report_id} file is missing and "
                             "cannot be repaired.")
                else:
                    reply = (f"Report {report_id} metadata is already "
                             "valid.")
                return {
                    "reply": reply, "tool_calls": tool, "sources": [],
                    "evidence": [], "reports": entries,
                }
        if not repairable:
            if missing:
                reply = (f"No repairable metadata; {len(missing)} missing "
                         "report file(s) cannot be repaired.")
            else:
                reply = "No report metadata needs repair."
            return {
                "reply": reply, "tool_calls": tool, "sources": [],
                "evidence": [], "reports": repairable + missing,
            }
        self.pending_report_action = {"action": "repair"}
        if report_id:
            self.pending_report_action["id"] = report_id
            lines = [
                f"Report {report_id} metadata repair preview:",
            ]
            for item in repairable:
                problems = ", ".join(item.get("problems", []))
                lines.append(
                    f"- {item.get('path', '')} [{problems}]")
        else:
            lines = [
                f"Report metadata repair preview: "
                f"{len(repairable)} repairable, {len(missing)} missing "
                f"({verification['summary']['total']} total)",
            ]
        for item in repairable:
            if not report_id:
                problems = ", ".join(item.get("problems", []))
                lines.append(
                    f"- {item.get('id', '?')} | repairable | "
                    f"{item.get('path', '')} [{problems}]")
        for item in missing:
            lines.append(
                f"- {item.get('id', '?')} | missing | "
                f"{item.get('path', '')}")
        lines.append('Reply "confirm reports" to repair metadata, or '
                     '"cancel reports".')
        return {
            "reply": "\n".join(lines),
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "reports": repairable + missing,
        }

    def _preview_prune_reports(self) -> dict:
        self.pending_report_action = None
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            entries = [item for item in archive.list(limit=100)
                       if not item.get("exists")]
        except (WeeklyReportError, OSError) as exc:
            return {"reply": f"report archive error: {exc}",
                    "tool_calls": [], "sources": [], "evidence": []}
        tool = [{"name": "preview_weekly_reports", "action": "prune",
                 "matches": len(entries)}]
        if not entries:
            return {
                "reply": "No missing report entries to prune.",
                "tool_calls": tool, "sources": [], "evidence": [],
            }
        self.pending_report_action = {"action": "prune"}
        reply = "\n".join([
            f"Missing report entries ({len(entries)}):",
            self._report_action_lines(entries),
            'Reply "confirm reports" to remove these index entries, or '
            '"cancel reports".',
        ])
        return {
            "reply": reply,
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "reports": entries,
        }

    def _preview_keep_reports(self, keep: int, dry_run: bool = False) -> dict:
        self.pending_report_action = None
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            entries = archive.keep_preview(keep)
            total = archive.count()
        except (WeeklyReportError, OSError) as exc:
            return {"reply": f"report archive error: {exc}",
                    "tool_calls": [], "sources": [], "evidence": []}
        tool = [{"name": "preview_weekly_reports", "action": "keep",
                 "keep": keep, "matches": len(entries)}]
        if not entries:
            return {
                "reply": (f"Nothing would be removed; {total} report(s) "
                          f"already fit in keep {keep}."),
                "tool_calls": tool, "sources": [], "evidence": [],
            }
        lines = [
            f"Keeping the newest {keep} report(s) would remove "
            f"{len(entries)} older entry(ies):",
            self._report_action_lines(entries),
        ]
        if dry_run:
            lines.append("Dry run: no archive changes were made.")
            reply = "\n".join(lines)
            return {
                "reply": reply,
                "tool_calls": tool + [{"name": "keep_weekly_reports",
                                       "keep": keep, "dry_run": True}],
                "sources": [],
                "evidence": [],
                "reports": entries,
                "would_remove_count": len(entries),
                "dry_run": True,
            }
        self.pending_report_action = {"action": "keep", "keep": keep}
        lines.append('Reply "confirm reports" to apply, or "cancel reports".')
        return {
            "reply": "\n".join(lines),
            "tool_calls": tool,
            "sources": [],
            "evidence": [],
            "reports": entries,
        }

    def _preview_delete_report(self, report_id: str) -> dict:
        self.pending_report_action = None
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            entry = archive.get(report_id)
        except (WeeklyReportError, OSError) as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        self.pending_report_action = {
            "action": "delete", "id": report_id}
        reply = "\n".join([
            f"Delete archive entry {report_id}? "
            "The report file is left on disk:",
            self._report_action_lines([entry]),
            'Reply "confirm reports" to delete the index entry, or '
            '"cancel reports".',
        ])
        return {
            "reply": reply,
            "tool_calls": [{"name": "preview_weekly_reports",
                            "action": "delete", "id": report_id}],
            "sources": [],
            "evidence": [],
            "reports": [entry],
        }

    def _apply_report_action(self) -> dict:
        pending = self.pending_report_action
        self.pending_report_action = None
        if pending is None:
            return {
                "reply": ("There is no pending report action. Use "
                          "'prune report archive', 'keep reports N' or "
                          "'delete report ID' first."),
                "tool_calls": [], "sources": [], "evidence": [],
            }
        action = pending.get("action")
        try:
            archive = WeeklyReportArchive(self.weekly_reports_path)
            if action == "prune":
                removed = archive.prune()
                remaining = archive.count()
                reply = (f"Pruned {removed} missing report(s). "
                         f"Archive now has {remaining} report(s).")
                tool = {"name": "prune_weekly_reports",
                        "removed": removed}
            elif action == "keep":
                keep = int(pending["keep"])
                trimmed = archive.keep(keep)
                remaining = archive.count()
                reply = (f"Kept the newest {keep} report(s); removed "
                         f"{trimmed} older entry(ies). "
                         f"Archive now has {remaining} report(s).")
                tool = {"name": "keep_weekly_reports", "keep": keep,
                        "trimmed": trimmed}
            elif action == "delete":
                report_id = str(pending["id"])
                deleted = archive.delete(report_id)
                remaining = archive.count()
                reply = (f"Deleted archive entry {report_id}. "
                         f"The report file was left on disk. "
                         f"Archive now has {remaining} report(s).")
                tool = {"name": "delete_weekly_report",
                        "id": report_id,
                        "path": deleted.get("path", "")}
            elif action == "repair":
                report_id = str(pending.get("id", "") or "")
                repair = archive.repair(report_id)
                remaining = archive.count()
                summary = repair["summary"]
                if report_id:
                    reply = (f"Report {report_id} metadata repair: "
                             f"{summary['repaired']} repaired; "
                             f"{summary['ok']} unchanged; "
                             f"{summary['missing']} missing. "
                             f"Archive now has {remaining} report(s).")
                else:
                    reply = (
                        f"Repaired {summary['repaired']} report metadata; "
                        f"{summary['ok']} unchanged; "
                        f"{summary['missing']} missing. "
                        f"Archive now has {remaining} report(s).")
                tool = {"name": "repair_weekly_reports",
                        "repaired": summary["repaired"],
                        "missing": summary["missing"]}
                if report_id:
                    tool["id"] = report_id
            else:
                return {"reply": "Unknown report action.",
                        "tool_calls": [], "sources": [], "evidence": []}
        except (WeeklyReportError, OSError, TypeError, ValueError) as exc:
            return {"reply": f"report archive error: {exc}",
                    "tool_calls": [], "sources": [], "evidence": []}
        return {
            "reply": reply,
            "tool_calls": [tool],
            "sources": [],
            "evidence": [],
            "report_action": {"action": action, "remaining": remaining},
        }

    def _cancel_report_action(self) -> dict:
        had_action = self.pending_report_action is not None
        self.pending_report_action = None
        return {
            "reply": ("Report action cancelled." if had_action
                      else "There is no pending report action."),
            "tool_calls": [], "sources": [], "evidence": [],
        }

    def _preview_delete_task(self, tid: int) -> dict:
        task = next((item for item in self.human.data.get("tasks", [])
                     if int(item.get("id", 0)) == int(tid)), None)
        if task is None:
            return {"reply": "I could not find that task.",
                    "tool_calls": [], "sources": [], "evidence": []}
        self.pending_task_action = {"action": "delete", "id": int(tid)}
        return {
            "reply": (
                f"Delete task {tid}: {task.get('text', '')}? "
                'Reply "confirm task delete" to remove it, or '
                '"cancel task delete".'),
            "tool_calls": [{"name": "preview_task_delete", "id": int(tid)}],
            "sources": [],
            "evidence": [],
            "task": task,
        }

    def _apply_task_delete(self) -> dict:
        pending = self.pending_task_action
        if pending is None or pending.get("action") != "delete":
            return {"reply": "There is no pending task delete.",
                    "tool_calls": [], "sources": [], "evidence": []}
        self.pending_task_action = None
        deleted = self.human.delete_task(int(pending["id"]))
        if deleted is None:
            return {"reply": "I could not find that task.",
                    "tool_calls": [], "sources": [], "evidence": []}
        return {
            "reply": (f"Deleted task {deleted['id']}: "
                      f"{deleted.get('text', '')}."),
            "tool_calls": [{"name": "delete_task", "id": deleted["id"]}],
            "sources": [],
            "evidence": [],
            "deleted_task": deleted,
        }

    def _cancel_task_delete(self) -> dict:
        pending = self.pending_task_action
        if pending is None or pending.get("action") != "delete":
            return {"reply": "There is no pending task delete.",
                    "tool_calls": [], "sources": [], "evidence": []}
        self.pending_task_action = None
        return {
            "reply": "Task delete cancelled.",
            "tool_calls": [], "sources": [], "evidence": [],
        }

    def _preview_tag_rename(self, old, new) -> dict:
        try:
            old, new = normalize_tag_rename(old, new)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        view = task_tag_view(self.human.data)
        entry = next((item for item in view["tags"]
                      if item["tag"] == old), None)
        if entry is None:
            return {"reply": f"Tag '{old}' is not used.",
                    "tool_calls": [], "sources": [], "evidence": []}
        self.pending_task_action = {
            "action": "rename_tag", "old": old, "new": new}
        return {
            "reply": (
                f"Rename tag '{old}' to '{new}' on {entry['total']} task(s) "
                f"({entry['open']} open, {entry['done']} done)? "
                'Reply "confirm tag rename" to apply, or '
                '"cancel tag rename".'),
            "tool_calls": [{"name": "preview_tag_rename",
                            "old": old, "new": new,
                            "tasks": entry["total"]}],
            "sources": [],
            "evidence": [],
            "tag_rename": {"old": old, "new": new,
                           "tasks": entry["total"],
                           "open": entry["open"],
                           "done": entry["done"]},
        }

    def _apply_tag_rename(self) -> dict:
        pending = self.pending_task_action
        if pending is None or pending.get("action") != "rename_tag":
            return {"reply": "There is no pending tag rename.",
                    "tool_calls": [], "sources": [], "evidence": []}
        self.pending_task_action = None
        try:
            result = self.human.rename_task_tag(
                pending["old"], pending["new"])
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        return {
            "reply": (f"Renamed tag '{result['old']}' to '{result['new']}' "
                      f"on {result['updated']} task(s)."),
            "tool_calls": [{"name": "rename_task_tag",
                            "old": result["old"],
                            "new": result["new"],
                            "updated": result["updated"]}],
            "sources": [],
            "evidence": [],
            "tag_rename": {**result, "applied": True},
        }

    def _cancel_tag_rename(self) -> dict:
        pending = self.pending_task_action
        if pending is None or pending.get("action") != "rename_tag":
            return {"reply": "There is no pending tag rename.",
                    "tool_calls": [], "sources": [], "evidence": []}
        self.pending_task_action = None
        return {
            "reply": "Tag rename cancelled.",
            "tool_calls": [], "sources": [], "evidence": [],
        }

    def _list_tasks(self, text: str) -> dict:
        view = self.human.build_task_list(text)
        return {
            "reply": view["text"],
            "tool_calls": [{"name": "list_tasks",
                            "status": view["status"],
                            "due": view["due"],
                            "priority": view["priority"],
                            "tag": view["tag"],
                            "sort": view["sort"],
                            "count": view["count"]}],
            "sources": [],
            "evidence": [],
            "tasks": view["tasks"],
            "task_list": view,
            "weights_changed": False,
        }

    def _list_task_tags(self) -> dict:
        view = task_tag_view(self.human.data)
        return {
            "reply": render_task_tag_view(view),
            "tool_calls": [{"name": "list_task_tags",
                            "tag_count": view["tag_count"],
                            "tagged_task_count": view["tagged_task_count"],
                            "untagged_task_count":
                                view["untagged_task_count"]}],
            "sources": [],
            "evidence": [],
            "task_tags": view,
            "weights_changed": False,
        }

    def _search_tasks(self, raw_query: str) -> dict:
        query, status, due, priority, sort, tag = (
            parse_task_search_filters(raw_query))
        if not query:
            return {
                "reply": ("Tell me task text to search, for example "
                          "'search tasks report open sort priority'."),
                "tool_calls": [], "sources": [], "evidence": [],
            }
        view = task_search_view(
            self.human.data, query, status=status, due=due,
            priority=priority, sort=sort, tag=tag)
        return {
            "reply": render_task_search_view(view),
            "tool_calls": [{"name": "search_tasks",
                            "query": view["query"],
                            "status": view["status"],
                            "due": view["due"],
                            "priority": view["priority"],
                            "tag": view["tag"],
                            "sort": view["sort"],
                            "matches": view["count"]}],
            "sources": [],
            "evidence": [],
            "task_search": view,
            "weights_changed": False,
        }

    def _overdue(self, sort: str = "default") -> dict:
        view = overdue_view(self.human.data, sort=sort)
        return {
            "reply": render_overdue_view(view),
            "tool_calls": [{"name": "overdue",
                            "sort": view["sort"],
                            "count": view["overdue_task_count"]}],
            "sources": [],
            "evidence": [],
            "overdue": view,
            "weights_changed": False,
        }

    def _due_soon(self, days: int = 7, sort: str = "default") -> dict:
        view = due_soon_view(self.human.data, days=days, sort=sort)
        return {
            "reply": render_due_soon_view(view),
            "tool_calls": [{"name": "due_soon",
                            "days": view["days"],
                            "sort": view["sort"]}],
            "sources": [],
            "evidence": [],
            "due_soon": view,
            "weights_changed": False,
        }

    def _weekly_review(self, days: int = 7, sort: str = "default",
                       tag: str = "all") -> dict:
        try:
            review = build_weekly_review(
                self.human.data, self.gap_store, self.docs,
                days=days, sort=sort, tag=tag)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        sources = []
        for item in review.get("recent_sources", []):
            source = {
                "source": item.get("source", ""),
                "title": (item.get("title") or item.get("source", "")),
                "kind": item.get("kind", "document"),
                "source_url": item.get("source_url", ""),
                "verified": bool(item.get("verified", False)),
            }
            if source not in sources:
                sources.append(source)
        tool = {
            "name": "weekly_review",
            "days": review.get("period_days", days),
            "sort": review["sort"],
        }
        if review.get("tag", "all") != "all":
            tool["tag"] = review["tag"]
        return {
            "reply": review["text"],
            "tool_calls": [tool],
            "sources": sources,
            "review": review,
            "weights_changed": False,
        }

    def _plan_goal(self, goal: str, confirm: bool = False) -> dict:
        if confirm:
            if self.pending_plan is None:
                return {
                    "reply": ("There is no pending goal plan. Use "
                              "'plan goal ...' first."),
                    "tool_calls": [], "sources": [], "evidence": [],
                }
            goal = self.pending_plan["goal"]
        try:
            plan = plan_goal(
                goal, self.docs, self.gap_store, self.human.data)
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}
        sources = []
        evidence = []
        for item in plan.get("context", []):
            source = {"source": item.get("source", ""),
                      "title": item.get("title", "")}
            if source not in sources:
                sources.append(source)
            evidence.append({
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "page": item.get("page"),
                "section": item.get("section", ""),
                "snippet": item.get("snippet", ""),
            })
        if not confirm:
            self.pending_plan = plan
            reply = (plan["text"] + "\n\nReply 'confirm plan' to create "
                     "these tasks, or 'cancel plan'.")
            return {
                "reply": reply,
                "tool_calls": [{"name": "goal_plan", "goal": goal}],
                "sources": sources,
                "evidence": evidence,
                "confidence": plan.get("confidence"),
                "suggested_tasks": plan.get("suggested_tasks", []),
            }
        created = []
        for text in plan.get("suggested_tasks", [])[:3]:
            task_id = self.human.add_task(text)
            created.append({"id": task_id, "text": text})
        self.pending_plan = None
        lines = [f"Created {len(created)} task(s) from the plan:"]
        lines.extend(f"- {item['id']}: {item['text']}" for item in created)
        return {
            "reply": "\n".join(lines),
            "tool_calls": [{"name": "goal_plan_confirm", "goal": goal}],
            "sources": sources,
            "evidence": evidence,
            "confidence": plan.get("confidence"),
            "created_tasks": created,
        }

    def _run_skill(self, skill: dict, text: str) -> dict:
        try:
            return execute_skill(
                skill, text, self.docs,
                getattr(self.human, "generator", None))
        except ValueError as exc:
            return {"reply": str(exc), "tool_calls": [],
                    "sources": [], "evidence": []}

    def chat(self, text: str) -> dict:
        if self.safety is not None:
            verdict = self.safety.check(text)
            if verdict is not None:
                return {
                    "reply": self.safety.refusal(verdict["category"]),
                    "tool_calls": [], "sources": [], "safety": verdict,
                }
        result = self._chat(text)
        if self.safety is not None:
            safe, verdict = self.safety.filter_reply(
                result.get("reply", ""))
            if verdict is not None:
                result = {**result, "reply": safe, "safety": verdict}
        return result

    def _chat(self, text: str) -> dict:
        low = text.lower().strip()
        if low in ("what time is it", "what time is it?", "time"):
            now = datetime.now().strftime("%Y-%m-%d %H:%M")
            return {"reply": f"It is {now}.", "tool_calls": [
                {"name": "time"}], "sources": []}
        if low in ("what can you do", "what can you do?", "help"):
            return {
                "reply": ("I can keep short personal notes, manage tasks, "
                          "build daily briefs and weekly reviews, search "
                          "your local documents, look up Wikipedia with "
                          "sources, calculate simple arithmetic, and tell "
                          "the time. I am a small local model, so I am not "
                          "a general reasoning or coding assistant."),
                "tool_calls": [{"name": "help"}], "sources": [],
            }
        expression = math_query(text)
        if expression is not None:
            try:
                result = calculate(expression)
            except (ValueError, SyntaxError, ZeroDivisionError,
                    OverflowError) as exc:
                return {
                    "reply": f"I could not calculate that: {exc}.",
                    "tool_calls": [{"name": "calculator",
                                    "expression": expression}],
                    "sources": [],
                }
            return {
                "reply": f"{expression} = {result}",
                "tool_calls": [{"name": "calculator",
                                "expression": expression,
                                "result": result}],
                "sources": [],
            }
        research = re.match(
            r"^(?:research|look up|search the web for|web search)\s*:?\s*(.+)$",
            text, re.IGNORECASE)
        if research:
            return self._research(research.group(1).strip())
        if (low.startswith("search docs ") or low.startswith("search my notes ")
                or low.startswith("search documents ")):
            query = re.sub(r"^search (docs|my notes|documents)\s*:?\s*", "",
                           text, flags=re.IGNORECASE)
            return self._search_docs(query)
        if low in ("confirm task delete", "confirm delete task",
                   "yes delete task"):
            return self._apply_task_delete()
        if low in ("cancel task delete", "cancel delete task"):
            return self._cancel_task_delete()
        if low in ("confirm tag rename", "confirm rename tag",
                   "yes rename tag"):
            return self._apply_tag_rename()
        if low in ("cancel tag rename", "cancel rename tag"):
            return self._cancel_tag_rename()
        rename_tag = re.match(
            r"^rename\s+tag\s+(\S+)\s+(?:to\s+|as\s+|->\s*)?(\S+?)\??$",
            text, re.IGNORECASE)
        if not rename_tag:
            rename_tag = re.match(
                r"^etiket(?:i)?\s+yeniden\s+adland[ıi]r\s+(\S+)\s+"
                r"(?:olarak\s+)?(\S+?)\??$", text, re.IGNORECASE)
        if rename_tag:
            return self._preview_tag_rename(
                rename_tag.group(1), rename_tag.group(2))
        task_delete = re.match(
            r"^(?:delete|remove)\s+task\s*(\d+)\??$",
            text, re.IGNORECASE)
        if task_delete:
            return self._preview_delete_task(int(task_delete.group(1)))
        if low in ("delete task", "remove task"):
            return {
                "reply": ("Tell me the task id, for example "
                          "'delete task 1'."),
                "tool_calls": [], "sources": [], "evidence": [],
            }
        compare_reports = re.match(
            r"^(?:compare|diff)\s+(?:the\s+)?reports?\s+"
            r"(w\d+)\s+(?:and|to|with|vs\.?|versus)?\s*"
            r"(w\d+)\??$", text, re.IGNORECASE)
        if compare_reports:
            return self._compare_weekly_reports(
                compare_reports.group(1), compare_reports.group(2))
        verify_match = re.match(
            r"^(?:verify|check)\s+(?:the\s+)?"
            r"(?:report\s+archive|reports?|report\s+integrity)"
            r"(?:\s+(w\d+))?\??$", text, re.IGNORECASE)
        if not verify_match:
            verify_match = re.match(
                r"^(?:verify|check)\s+(w\d+)\??$",
                text, re.IGNORECASE)
        if not verify_match:
            verify_match = re.match(
                r"^(?:rapor|raporlar[ıi])\s+ar[şs]ivini\s+"
                r"do[ğg]rula\??$", text, re.IGNORECASE)
        if not verify_match:
            verify_match = re.match(
                r"^(?:rapor|raporu)\s+(w\d+)\s+do[ğg]rula\??$",
                text, re.IGNORECASE)
        if verify_match:
            report_id = (verify_match.group(1)
                         if verify_match.lastindex else "")
            return self._verify_weekly_reports(report_id or "")
        show_match = re.match(
            r"^(?:show|open|display|view)\s+(?:the\s+)?"
            r"(?:report|report\s+archive|archive\s+entry)\s+"
            r"(w\d+)\??$", text, re.IGNORECASE)
        if not show_match:
            show_match = re.match(
                r"^(?:show|open|display|view)\s+(w\d+)\s+"
                r"(?:report|archive\s+entry)\??$", text, re.IGNORECASE)
        if not show_match:
            show_match = re.match(
                r"^(?:rapor|raporu)\s+(?:g[öo]ster|a[çc])\s+"
                r"(w\d+)\??$", text, re.IGNORECASE)
        if show_match:
            return self._show_weekly_report(show_match.group(1))
        report_sort = "newest"
        report_text = text
        if re.match(r"^(?:find|search|show|list)\s+", text,
                    re.IGNORECASE):
            sort_match = re.search(
                r"\s+(newest|oldest|largest)\s*\??$", text,
                re.IGNORECASE)
            if sort_match:
                report_sort = sort_match.group(1).lower()
                report_text = text[:sort_match.start()].rstrip()
        list_match = re.match(
            r"^(?:list|show)\s+(?:the\s+)?"
            r"(?:reports?|report\s+archive|weekly\s+reports?)"
            r"(?:\s+(?:limit\s+)?(\d+))?\??$",
            report_text, re.IGNORECASE)
        if not list_match:
            list_match = re.match(
                r"^(?:list|show)\s+(?:the\s+)?(?:last\s+)?(\d+)\s+"
                r"(?:reports?|weekly\s+reports?)\??$",
                report_text, re.IGNORECASE)
        if list_match:
            return self._list_weekly_reports(
                int(list_match.group(1)) if list_match.group(1) else 10,
                order=report_sort)
        report_query = re.match(
            r"^(?:find|search|show)\s+(?:the\s+)?report\s+archive\s+"
            r"(?:for\s+)?(.+?)\??$", report_text, re.IGNORECASE)
        if not report_query:
            report_query = re.match(
                r"^(?:find|search|show|list)\s+(?:the\s+)?"
                r"(?:weekly\s+)?reports?\s+(?:for\s+)?(.+?)\??$",
                report_text, re.IGNORECASE)
        if not report_query:
            report_query = re.match(
                r"^(?:rapor|raporlar[ıi])\s+(?:ara|bul)\s+(.+?)\??$",
                report_text, re.IGNORECASE)
        if not report_query:
            report_query = re.match(
                r"^(?:haftal[ıi]k\s+)?rapor\s+ar[şs]ivinde\s+"
                r"(.+?)\s+ara\??$", report_text, re.IGNORECASE)
        if report_query:
            return self._search_weekly_reports(
                report_query.group(1).strip(), order=report_sort)
        repair_match = re.match(
            r"^(?:repair|fix)\s+(?:the\s+)?"
            r"(?:report\s+archive|reports?|report\s+metadata)"
            r"(?:\s+metadata)?(?:\s+(w\d+))?\??$",
            text, re.IGNORECASE)
        if not repair_match:
            repair_match = re.match(
                r"^(?:repair|fix)\s+(w\d+)\??$",
                text, re.IGNORECASE)
        if not repair_match:
            repair_match = re.match(
                r"^(?:rapor|raporlar[ıi])\s+ar[şs]ivini\s+"
                r"onar\??$", text, re.IGNORECASE)
        if not repair_match:
            repair_match = re.match(
                r"^(?:rapor|raporu)\s+(w\d+)\s+onar\??$",
                text, re.IGNORECASE)
        if repair_match:
            report_id = (repair_match.group(1)
                         if repair_match.lastindex else "")
            return self._preview_repair_reports(report_id or "")
        if low in ("confirm reports", "confirm report changes", "apply reports",
                   "yes reports", "raporları onayla"):
            return self._apply_report_action()
        if low in ("cancel reports", "cancel report changes",
                   "discard reports", "rapor işlemini iptal et"):
            return self._cancel_report_action()
        prune_match = re.match(
            r"^(?:prune|clean(?:\s+up)?|remove\s+missing)\s+(?:the\s+)?"
            r"(?:missing\s+)?reports?(?:\s+archive)?\??$",
            text, re.IGNORECASE)
        if prune_match:
            return self._preview_prune_reports()
        dry_run = bool(re.search(
            r"\bdry[- ]?run\??$", text, re.IGNORECASE))
        keep_text = re.sub(
            r"\s+dry[- ]?run\??$", "", text,
            flags=re.IGNORECASE).strip()
        keep_match = re.match(
            r"^(?:keep|retain)\s+(?:the\s+)?(?:newest\s+)?"
            r"(?:reports?|report\s+archive)\s+(?:for\s+)?(\d+)\??$",
            keep_text, re.IGNORECASE)
        if not keep_match:
            keep_match = re.match(
                r"^(?:keep|retain)\s+(?:the\s+)?(\d+)\s+"
                r"(?:newest\s+)?reports?\??$", keep_text, re.IGNORECASE)
        if keep_match:
            return self._preview_keep_reports(
                int(keep_match.group(1)), dry_run=dry_run)
        delete_match = re.match(
            r"^(?:delete|remove)\s+(?:the\s+)?"
            r"(?:report|report\s+archive|archive\s+entry)\s+"
            r"(w\d+)\??$", text, re.IGNORECASE)
        if not delete_match:
            delete_match = re.match(
                r"^(?:delete|remove)\s+(w\d+)\s+"
                r"(?:report|archive\s+entry)\??$", text, re.IGNORECASE)
        if delete_match:
            return self._preview_delete_report(delete_match.group(1))
        graph_query = re.match(
            r"^(?:what is related to|what relates to|connections (?:for|of)|"
            r"graph)\s*:?\s*(.+?)\??$", text, re.IGNORECASE)
        if graph_query:
            return self._graph_entity(graph_query.group(1).strip())
        path_query = re.match(
            r"^(?:how is|how are)\s+(.+?)\s+connected to\s+(.+?)\??$",
            text, re.IGNORECASE)
        if not path_query:
            path_query = re.match(
                r"^(?:connection between|path from)\s+(.+?)\s+"
                r"(?:and|to)\s+(.+?)\??$", text, re.IGNORECASE)
        if not path_query:
            path_query = re.match(
                r"^connect\s+(.+?)\s+to\s+(.+?)\??$", text,
                re.IGNORECASE)
        if not path_query:
            path_query = re.match(
                r"^(.+?)\s+ile\s+(.+?)\s+(?:nasıl|ne)\s+"
                r"bağlantılı\??$", text, re.IGNORECASE)
        if path_query:
            return self._graph_path(
                path_query.group(1).strip(),
                path_query.group(2).strip())
        compare_match = re.match(
            r"^(?:compare sources|compare)\s+(.+?)\s+"
            r"(?:and|to|with|vs\.?|versus)\s+(.+?)\??$",
            text, re.IGNORECASE)
        if not compare_match:
            compare_match = re.match(
                r"^(.+?)\s+(?:ve|ile)\s+(.+?)\s+"
                r"(?:kaynaklar[ıi]n[ıi]\s+)?karşılaştır\??$",
                text, re.IGNORECASE)
        if compare_match:
            return self._compare_sources(
                compare_match.group(1).strip(),
                compare_match.group(2).strip())
        brief_match = re.match(
            r"^(?:brief me|daily brief|what should i do today)"
            r"(?:\s+(?P<sort>priority))?"
            r"(?:\s+tag\s+(?P<tag>\S+))?\??$", text, re.IGNORECASE)
        if brief_match:
            groups = brief_match.groupdict()
            return self._briefing(
                groups.get("sort") or "default",
                tag=groups.get("tag") or "all")
        if (re.match(r"^(?:list|show)\s+(?:task\s+)?tags\??$", text,
                     re.IGNORECASE)
                or low.rstrip("?") in ("tags", "task tags")):
            return self._list_task_tags()
        if "list tasks" in low:
            return self._list_tasks(text)
        task_search = re.match(
            r"^(?:search|find)\s+tasks?\s+(.+?)\??$",
            text, re.IGNORECASE)
        if not task_search:
            task_search = re.match(
                r"^task\s+search\s+(.+?)\??$", text, re.IGNORECASE)
        if task_search:
            return self._search_tasks(task_search.group(1).strip())
        overdue_match = re.match(
            r"^(?:overdue(?: tasks?)?|tasks overdue|"
            r"show overdue(?: tasks?)?)"
            r"(?:\s+(?:sort\s+)?(?P<sort>default|priority))?\??$",
            text, re.IGNORECASE)
        if overdue_match:
            return self._overdue(
                overdue_match.groupdict().get("sort") or "default")
        due_match = re.match(
            r"^(?:due soon|tasks due soon|what(?:'s| is) due soon|"
            r"show due soon)(?:\s+(?:in\s+)?(?P<days>\d+)\s+days?)?"
            r"(?:\s+(?:sort\s+)?(?P<sort>default|priority))?\??$",
            text, re.IGNORECASE)
        if due_match:
            groups = due_match.groupdict()
            days_raw = groups.get("days")
            return self._due_soon(
                int(days_raw) if days_raw else 7,
                sort=groups.get("sort") or "default")
        weekly_tag_match = re.match(
            r"^(?:(?:weekly|week)\s+(?:review|summary)|"
            r"haftal[ıi]k\s+(?:özet|rapor|değerlendirme))"
            r"(?:\s+(?:for|last|son)?\s*(?P<days>\d+)\s*"
            r"(?:days?|gün))?"
            r"\s+tag\s+(?P<tag>\S+)"
            r"(?:\s+(?P<sort>priority))?\??$",
            text, re.IGNORECASE)
        if weekly_tag_match:
            groups = weekly_tag_match.groupdict()
            days_raw = groups.get("days")
            return self._weekly_review(
                int(days_raw) if days_raw else 7,
                sort=groups.get("sort") or "default",
                tag=groups.get("tag") or "all")
        weekly_match = re.match(
            r"^(?:(?:weekly|week)\s+(?:review|summary)|"
            r"haftal[ıi]k\s+(?:özet|rapor|değerlendirme))"
            r"(?:\s+(?:for|last|son)?\s*(?P<days>\d+)\s*"
            r"(?:days?|gün))?"
            r"(?:\s+(?P<sort>priority))?\??$",
            text, re.IGNORECASE)
        if not weekly_match:
            weekly_match = re.match(
                r"^son\s+(?P<days>\d+)\s+gün(?:ün)?\s+özeti\??$",
                text, re.IGNORECASE)
        if not weekly_match:
            weekly_match = re.match(
                r"^bu\s+haftan[ıi]n\s+özeti\??$", text, re.IGNORECASE)
        if weekly_match:
            groups = weekly_match.groupdict()
            days_raw = groups.get("days")
            return self._weekly_review(
                int(days_raw) if days_raw else 7,
                sort=groups.get("sort") or "default")
        plan_match = re.match(
            r"^(?:plan goal|goal plan)\s*:?\s*(.+)$", text, re.IGNORECASE)
        if plan_match:
            return self._plan_goal(plan_match.group(1).strip())
        if low in ("confirm plan", "create plan tasks", "yes plan"):
            return self._plan_goal("", confirm=True)
        if low in ("cancel plan", "discard plan"):
            had_plan = self.pending_plan is not None
            self.pending_plan = None
            return {
                "reply": ("Plan cancelled." if had_plan
                          else "There is no pending goal plan."),
                "tool_calls": [], "sources": [], "evidence": [],
            }
        if self.skills is not None:
            skill = self.skills.match(text)
            if skill is not None:
                return self._run_skill(skill, text)
        reply = self.human.respond(text)
        return {"reply": reply, "tool_calls": [],
                "sources": list(getattr(self.human, "last_sources", [])),
                "evidence": list(getattr(self.human, "last_evidence", [])),
                "confidence": getattr(
                    self.human, "last_confidence",
                    {"score": 0.0, "label": "none",
                     "coverage": 0.0, "margin": 0.0})}
