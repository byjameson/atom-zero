"""Session-based agent runtime around the tiny model and local tools."""
from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from atomzero.briefing import build_briefing, build_weekly_review, plan_goal
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

    def _briefing(self) -> dict:
        brief = build_briefing(
            self.human.data, self.gap_store, self.docs)
        sources = brief.get("documents", {}).get("items", [])
        return {
            "reply": brief["text"],
            "tool_calls": [{"name": "briefing"}],
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

    def _weekly_review(self, days: int = 7) -> dict:
        review = build_weekly_review(
            self.human.data, self.gap_store, self.docs, days=days)
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
        return {
            "reply": review["text"],
            "tool_calls": [{
                "name": "weekly_review",
                "days": review.get("period_days", days),
            }],
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
        if low in ("brief me", "brief me?", "daily brief", "daily brief?",
                   "what should i do today", "what should i do today?"):
            return self._briefing()
        weekly_match = re.match(
            r"^(?:(?:weekly|week)\s+(?:review|summary)|"
            r"haftal[ıi]k\s+(?:özet|rapor|değerlendirme))"
            r"(?:\s+(?:for|last|son)?\s*(?P<days>\d+)\s*"
            r"(?:days?|gün))?\??$",
            text, re.IGNORECASE)
        if not weekly_match:
            weekly_match = re.match(
                r"^son\s+(?P<days>\d+)\s+gün(?:ün)?\s+özeti\??$",
                text, re.IGNORECASE)
        if not weekly_match:
            weekly_match = re.match(
                r"^bu\s+haftan[ıi]n\s+özeti\??$", text, re.IGNORECASE)
        if weekly_match:
            days_raw = weekly_match.groupdict().get("days")
            return self._weekly_review(int(days_raw) if days_raw else 7)
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
