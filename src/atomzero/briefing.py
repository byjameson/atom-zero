"""Daily briefing and local goal planning for Atom Zero.

This module is deterministic by default: it reads local tasks, pending gaps,
document stats and the second-brain graph, then produces a short brief. Goal
planning adds cited local context and a few concrete next actions. It never
changes model weights and never calls a provider on its own.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from atomzero import __version__
from atomzero.doc_memory import (
    DocMemory, best_snippet, format_citation, retrieval_confidence)
from atomzero.knowledge_graph import build_graph
from atomzero.weekly_reports import (
    DEFAULT_REPORTS_PATH, WeeklyReportArchive, WeeklyReportError)


def _open_tasks(session_data: dict | None) -> list[dict]:
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks
            if isinstance(task, dict) and task.get("status") == "open"]


def _today_from_now(now=None) -> str:
    if isinstance(now, datetime):
        return now.date().isoformat()
    text = str(now or "").strip()
    try:
        return date.fromisoformat(text[:10]).isoformat()
    except (TypeError, ValueError):
        return date.today().isoformat()


def _task_due_label(task: dict, today: str) -> str:
    due = str(task.get("due", "") or "").strip()
    if not due:
        return ""
    is_open = task.get("status", "open") == "open"
    marker = " [overdue]" if due < today and is_open else ""
    return f" (due {due}{marker})"


def _overdue_tasks(session_data: dict | None,
                   today: str | None = None) -> list[dict]:
    today = today or date.today().isoformat()
    return [task for task in _open_tasks(session_data)
            if str(task.get("due", "") or "").strip()
            and str(task["due"]) < today]


def overdue_view(session_data: dict | None, now=None) -> dict:
    tasks = _open_tasks(session_data)
    today = _today_from_now(now)
    overdue = _overdue_tasks(session_data, today)
    return {
        "generated": (now.strftime("%Y-%m-%d %H:%M")
                      if isinstance(now, datetime)
                      else str(now or time.strftime("%Y-%m-%d %H:%M"))),
        "today": today,
        "open_task_count": len(tasks),
        "overdue_task_count": len(overdue),
        "open_tasks": tasks,
        "overdue_tasks": overdue,
        "weights_changed": False,
    }


def render_overdue_view(view: dict) -> str:
    lines = [
        f"Overdue tasks ({view.get('overdue_task_count', 0)} of "
        f"{view.get('open_task_count', 0)} open) - {view.get('today', '')}"
    ]
    tasks = view.get("overdue_tasks", [])
    if not tasks:
        lines.append("- None")
        return "\n".join(lines)
    for task in tasks:
        lines.append(
            f"- {task.get('id')}: {task.get('text', '')} | "
            f"due {task.get('due', '')} | overdue")
    return "\n".join(lines)


def due_soon_tasks(session_data: dict | None, days: int = 7,
                   today: str | None = None) -> list[dict]:
    try:
        days = max(1, min(int(days), 90))
    except (TypeError, ValueError):
        days = 7
    today = today or date.today().isoformat()
    end = (date.fromisoformat(today) + timedelta(days=days)).isoformat()
    return [task for task in _open_tasks(session_data)
            if str(task.get("due", "") or "").strip()
            and today <= str(task["due"]) <= end]


def due_soon_view(session_data: dict | None, days: int = 7,
                  now=None) -> dict:
    try:
        days = max(1, min(int(days), 90))
    except (TypeError, ValueError):
        days = 7
    today = _today_from_now(now)
    end = (date.fromisoformat(today) + timedelta(days=days)).isoformat()
    tasks = _open_tasks(session_data)
    due = due_soon_tasks(session_data, days=days, today=today)
    return {
        "generated": (now.strftime("%Y-%m-%d %H:%M")
                      if isinstance(now, datetime)
                      else str(now or time.strftime("%Y-%m-%d %H:%M"))),
        "today": today,
        "window_end": end,
        "days": days,
        "open_task_count": len(tasks),
        "due_soon_count": len(due),
        "open_tasks": tasks,
        "due_soon_tasks": due,
        "weights_changed": False,
    }


def render_due_soon_view(view: dict) -> str:
    lines = [
        f"Due soon ({view.get('due_soon_count', 0)} of "
        f"{view.get('open_task_count', 0)} open, next "
        f"{view.get('days', 7)} days) - {view.get('today', '')} "
        f"to {view.get('window_end', '')}"
    ]
    tasks = view.get("due_soon_tasks", [])
    if not tasks:
        lines.append("- None")
        return "\n".join(lines)
    for task in tasks:
        lines.append(
            f"- {task.get('id')}: {task.get('text', '')} | "
            f"due {task.get('due', '')}")
    return "\n".join(lines)


def search_tasks(session_data: dict | None, query: str) -> list[dict]:
    tokens = [token for token in str(query or "").strip().lower().split()
              if token]
    if not tokens:
        return []
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    matches = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        text = str(task.get("text", "")).lower()
        if all(token in text for token in tokens):
            matches.append(task)
    return matches


def task_search_view(session_data: dict | None, query: str,
                     now=None) -> dict:
    query = str(query or "").strip()
    matches = search_tasks(session_data, query)
    return {
        "generated": (now.strftime("%Y-%m-%d %H:%M")
                      if isinstance(now, datetime)
                      else str(now or time.strftime("%Y-%m-%d %H:%M"))),
        "query": query,
        "count": len(matches),
        "tasks": matches,
        "open_task_count": len(_open_tasks(session_data)),
        "weights_changed": False,
    }


def render_task_search_view(view: dict) -> str:
    query = view.get("query", "")
    lines = [f'Task search "{query}" ({view.get("count", 0)} match(es))']
    tasks = view.get("tasks", [])
    if not tasks:
        lines.append("- None")
        return "\n".join(lines)
    today = _today_from_now(view.get("generated"))
    for task in tasks:
        lines.append(
            f"- {task.get('id')} [{task.get('status', '?')}] "
            f"{task.get('text', '')}{_task_due_label(task, today)}")
    return "\n".join(lines)


def _pending_gaps(gaps) -> list[dict]:
    if gaps is None:
        return []
    try:
        items = gaps.list()
    except Exception:
        return []
    return [gap for gap in items if gap.get("status") == "pending"]


def _graph_snapshot(docs) -> dict:
    if docs is None:
        return {"nodes": [], "edges": [], "stats": {}}
    try:
        return build_graph(docs, limit_chunks=200, max_entities=20)
    except Exception:
        return {"nodes": [], "edges": [], "stats": {}}


def build_briefing(session_data: dict | None, gaps, docs,
                   now=None) -> dict:
    """Build a compact local brief from tasks, gaps, docs and the graph."""
    tasks = _open_tasks(session_data)
    today = _today_from_now(now)
    overdue = _overdue_tasks(session_data, today)
    due_soon = due_soon_tasks(session_data, days=7, today=today)
    pending = _pending_gaps(gaps)
    sources = docs.sources() if docs is not None else []
    chunks = docs.stats().get("chunks", 0) if docs is not None else 0
    graph = _graph_snapshot(docs)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if overdue:
        focus = (f"overdue task {overdue[0].get('id')}: "
                 f"{overdue[0].get('text', '')}")
    elif due_soon:
        focus = (f"due soon: task {due_soon[0].get('id')}: "
                 f"{due_soon[0].get('text', '')}")
    elif tasks:
        focus = f"task {tasks[0].get('id')}: {tasks[0].get('text', '')}"
    elif pending:
        focus = f"gap: {pending[0].get('query', '')}"
    elif nodes:
        focus = f"explore: {nodes[0].get('name', '')}"
    elif not sources:
        focus = "add a local document or sync a folder"
    else:
        focus = "ask a question about your local notes"

    actions = []
    for task in overdue[:3]:
        actions.append(
            f"Handle overdue task {task.get('id')}: "
            f"{task.get('text', '')} (due {task.get('due', '')})")
    for task in due_soon[:3]:
        actions.append(
            f"Finish due-soon task {task.get('id')}: "
            f"{task.get('text', '')} (due {task.get('due', '')})")
    for task in tasks[:3]:
        if task in overdue or task in due_soon:
            continue
        actions.append(f"Finish task {task.get('id')}: {task.get('text', '')}")
    for gap in pending[:2]:
        actions.append(f"Review the open gap: {gap.get('query', '')}")
    if not sources:
        actions.append("Sync a folder with `atomzero-docs sync .\\notes`")
    if nodes:
        actions.append(
            f"Explore the graph: what is related to {nodes[0].get('name')}?")
    actions.append("Ask a grounded question or run a custom skill")
    stamp = (now or time.strftime("%Y-%m-%d %H:%M"))
    task_summary = f"Tasks: {len(tasks)} open"
    if overdue:
        task_summary += f" | {len(overdue)} overdue"
    if due_soon:
        task_summary += f" | {len(due_soon)} due soon"
    lines = [
        f"Daily brief - {stamp}",
        f"Focus: {focus}",
        task_summary,
    ]
    lines.extend(
        f"- {task.get('id')}: {task.get('text', '')}"
        f"{_task_due_label(task, today)}"
        for task in tasks[:5])
    lines.append(f"Gaps: {len(pending)} pending")
    lines.extend(f"- {gap.get('query', '')}" for gap in pending[:5])
    lines.append(f"Index: {chunks} chunk(s) / {len(sources)} source(s)")
    lines.append(
        f"Graph: {len(nodes)} entities / {len(edges)} connections")
    if nodes:
        lines.append("Top entities: " + ", ".join(
            node.get("name", "") for node in nodes[:5]))
    lines.append("Next actions:")
    lines.extend(f"- {action}" for action in actions[:6])
    return {
        "generated": stamp,
        "focus": focus,
        "tasks": tasks[:5],
        "open_task_count": len(tasks),
        "overdue_tasks": overdue[:5],
        "overdue_task_count": len(overdue),
        "due_soon_tasks": due_soon[:5],
        "due_soon_count": len(due_soon),
        "pending_gaps": pending[:5],
        "pending_gap_count": len(pending),
        "documents": {
            "chunks": chunks,
            "sources": len(sources),
            "titles": [item.get("title", item.get("source", ""))
                       for item in sources[:5]],
            "items": [{
                "source": item.get("source", ""),
                "title": item.get("title", ""),
                "kind": item.get("kind", "document"),
                "verified": bool(item.get("verified", True)),
            } for item in sources[:5]],
        },
        "graph": {
            "entities": nodes[:8],
            "connections": edges[:8],
        },
        "next_actions": actions[:6],
        "text": "\n".join(lines),
        "weights_changed": False,
    }


def _done_tasks(session_data: dict | None) -> list[dict]:
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks
            if isinstance(task, dict) and task.get("status") == "done"]


def _task_recent(value, cutoff: datetime) -> bool:
    if not value:
        return False
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except (TypeError, ValueError):
        return False
    if parsed.tzinfo is not None:
        parsed = parsed.replace(tzinfo=None)
    return parsed >= cutoff


def _source_recent(item: dict, cutoff_ts: float) -> bool:
    value = item.get("created", "")
    if not value:
        return True
    try:
        return float(value) >= cutoff_ts
    except (TypeError, ValueError):
        try:
            parsed = datetime.fromisoformat(str(value))
            return parsed >= datetime.fromtimestamp(cutoff_ts)
        except (TypeError, ValueError):
            return True


def build_weekly_review(session_data: dict | None, gaps, docs,
                        days: int = 7, now=None) -> dict:
    """Aggregate tasks, gaps, recent sources and graph highlights."""
    try:
        days = max(1, min(int(days), 90))
    except (TypeError, ValueError):
        days = 7
    stamp = (now.strftime("%Y-%m-%d %H:%M")
             if isinstance(now, datetime)
             else str(now or time.strftime("%Y-%m-%d %H:%M")))
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = time.time() - days * 86400
    open_tasks = _open_tasks(session_data)
    today = _today_from_now(now)
    overdue = _overdue_tasks(session_data, today)
    due_soon = due_soon_tasks(session_data, days=7, today=today)
    done_tasks = _done_tasks(session_data)
    completed_recent = [task for task in done_tasks
                        if _task_recent(task.get("completed"), cutoff)]
    completed_undated = [task for task in done_tasks
                         if not task.get("completed")]
    pending = _pending_gaps(gaps)
    sources = docs.sources() if docs is not None else []
    recent_sources = [source for source in sources
                      if _source_recent(source, cutoff_ts)]
    graph = _graph_snapshot(docs)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    highlights = []
    if recent_sources:
        first = recent_sources[0]
        highlights.append(
            f"Recently indexed: {first.get('title', first.get('source', ''))}")
    if nodes:
        highlights.append(
            "Top graph entity: " + nodes[0].get("name", ""))
    if edges:
        edge = edges[0]
        highlights.append(
            f"Strongest connection: {edge.get('source')} ↔ "
            f"{edge.get('target')}")
    next_week = []
    for task in due_soon[:3]:
        next_week.append(
            f"Finish due-soon task {task.get('id')}: "
            f"{task.get('text', '')} (due {task.get('due', '')})")
    for task in open_tasks[:3]:
        if task in due_soon:
            continue
        next_week.append(
            f"Close task {task.get('id')}: {task.get('text', '')}")
    for gap in pending[:2]:
        next_week.append(f"Resolve the gap: {gap.get('query', '')}")
    if recent_sources:
        next_week.append(
            f"Review the new source: "
            f"{recent_sources[0].get('title', recent_sources[0].get('source', ''))}")
    if nodes:
        next_week.append(
            f"Explore the graph around {nodes[0].get('name', '')}")
    next_week.append("Run a skill or named recipe for a recurring task")
    period_label = ("this week" if days == 7
                    else f"in the last {days} days")
    completed_label = ("Completed this week:" if days == 7
                       else f"Completed in the last {days} days:")
    lines = [
        f"Weekly review - {stamp} (last {days} days)",
        f"Tasks: {len(open_tasks)} open | {len(completed_recent)} "
        f"completed {period_label}",
    ]
    if overdue:
        lines.append(f"Overdue: {len(overdue)}")
    if due_soon:
        lines.append(f"Due soon: {len(due_soon)} (next 7 days)")
    lines.append("Open:")
    lines.extend(
        f"- {task.get('id')}: {task.get('text', '')}"
        f"{_task_due_label(task, today)}"
        for task in open_tasks[:8])
    if completed_recent:
        lines.append(completed_label)
        lines.extend(f"- {task.get('id')}: {task.get('text', '')}"
                     for task in completed_recent[:8])
    if completed_undated:
        lines.append("Completed (undated):")
        lines.extend(f"- {task.get('id')}: {task.get('text', '')}"
                     for task in completed_undated[:5])
    lines.append(f"Gaps: {len(pending)} pending")
    lines.extend(f"- {gap.get('query', '')}" for gap in pending[:5])
    lines.append(f"Recent sources: {len(recent_sources)}")
    lines.extend(f"- {source.get('title', source.get('source', ''))} "
                 f"({source.get('source', '')})"
                 for source in recent_sources[:8])
    lines.append(
        f"Graph: {len(nodes)} entities / {len(edges)} connections")
    lines.append("Next week:")
    lines.extend(f"- {action}" for action in next_week[:7])
    return {
        "generated": stamp,
        "period_days": days,
        "open_tasks": open_tasks[:8],
        "overdue_tasks": overdue[:8],
        "overdue_task_count": len(overdue),
        "due_soon_tasks": due_soon[:8],
        "due_soon_count": len(due_soon),
        "completed_tasks": completed_recent[:8],
        "completed_undated": completed_undated[:5],
        "task_counts": {
            "open": len(open_tasks),
            "completed_recent": len(completed_recent),
            "completed_total": len(done_tasks),
        },
        "pending_gaps": pending[:5],
        "pending_gap_count": len(pending),
        "recent_sources": recent_sources[:8],
        "recent_source_count": len(recent_sources),
        "graph": {
            "entities": nodes[:8],
            "connections": edges[:8],
        },
        "highlights": highlights,
        "next_week": next_week[:7],
        "text": "\n".join(lines),
        "weights_changed": False,
    }


def plan_goal(goal: str, docs, gaps=None, session_data: dict | None = None,
              max_steps: int = 3) -> dict:
    """Create a cited, deterministic plan skeleton for a local goal."""
    goal = re.sub(r"\s+", " ", str(goal or "")).strip()
    if len(goal) < 3:
        raise ValueError("goal is required")
    if len(goal) > 500:
        raise ValueError("goal is too long (max 500 chars)")
    try:
        max_steps = max(1, min(int(max_steps), 6))
    except (TypeError, ValueError):
        max_steps = 3
    hits = docs.search(goal, k=6, diverse=True) if docs is not None else []
    context = []
    for hit in hits[:5]:
        context.append({
            "source": hit.get("source", ""),
            "title": hit.get("title", ""),
            "page": hit.get("page"),
            "section": hit.get("section", ""),
            "snippet": best_snippet(goal, hit.get("chunk", "")),
            "citation": format_citation(hit),
        })
    confidence = (retrieval_confidence(goal, hits) if hits
                  else {"score": 0.0, "label": "none",
                        "coverage": 0.0, "margin": 0.0})
    pending = _pending_gaps(gaps)
    tasks = _open_tasks(session_data)
    suggestions = []
    if context:
        suggestions.append(
            f"Review the evidence from {context[0]['citation']} "
            f"for: {goal}")
    else:
        suggestions.append(f"Find one local source for: {goal}")
    if pending:
        suggestions.append(f"Resolve the open question: "
                           f"{pending[0].get('query', '')}")
    else:
        suggestions.append(f"Write down one assumption to test for: {goal}")
    if tasks:
        suggestions.append(f"Connect the goal to task {tasks[0].get('id')}: "
                           f"{tasks[0].get('text', '')}")
    else:
        suggestions.append(f"Define the first concrete step for: {goal}")
    return {
        "goal": goal,
        "context": context,
        "confidence": confidence,
        "suggested_tasks": suggestions[:max_steps],
        "pending_gaps": pending[:3],
        "open_tasks": tasks[:3],
        "weights_changed": False,
        "text": ("Goal: " + goal + "\n\nSuggested plan:\n" +
                 "\n".join(f"{index}. {item}"
                           for index, item in enumerate(
                               suggestions[:max_steps], 1))),
    }


def load_session(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read session: {exc}") from exc
    return data if isinstance(data, dict) else {}


def render_briefing(brief: dict) -> str:
    return str(brief.get("text", ""))


def render_plan(plan: dict) -> str:
    lines = [str(plan.get("text", ""))]
    if plan.get("context"):
        lines.append("\nEvidence:")
        for item in plan["context"]:
            lines.append(f"- {item.get('snippet', '')} "
                         f"(source: {item.get('citation', '')})")
    return "\n".join(lines)


def _weekly_report_line(item: dict) -> str:
    counts = item.get("task_counts", {})
    marker = "" if item.get("exists", True) else " [missing]"
    return (
        f"- {item.get('id', '?')} | {item.get('exported_at', '')} | "
        f"last {item.get('period_days', 7)} days | "
        f"{counts.get('open', 0)} open | "
        f"{item.get('recent_source_count', 0)} source(s) | "
        f"{item.get('path', '')}{marker}")


def render_weekly_report_index(entries: list[dict],
                               index_path: str | Path,
                               query: str = "",
                               matched: int | None = None,
                               total: int | None = None,
                               sort: str = "newest") -> str:
    heading = f"Weekly report archive: {index_path}"
    if query:
        heading += (f" | query: {query} | "
                    f"{matched if matched is not None else len(entries)} "
                    f"match(es) of {total if total is not None else '?'}")
    heading += f" | sort: {sort}"
    lines = [heading]
    if not entries:
        lines.append("No archived weekly reports." if not query
                     else f'No archived reports match "{query}".')
        return "\n".join(lines)
    lines.extend(_weekly_report_line(item) for item in entries)
    return "\n".join(lines)


def render_weekly_report_verification(result: dict) -> str:
    summary = result.get("summary", {})
    lines = [
        f"Report integrity: {summary.get('ok', 0)} ok, "
        f"{summary.get('failed', 0)} failed, "
        f"{summary.get('missing', 0)} missing "
        f"({summary.get('total', 0)} total)"
    ]
    for item in result.get("entries", []):
        problems = ",".join(item.get("problems", []))
        suffix = f" [{problems}]" if problems else ""
        lines.append(
            f"- {item.get('id', '?')} | {item.get('status', '?')} | "
            f"{item.get('path', '')}{suffix}")
    return "\n".join(lines)


def render_weekly_report_repair(result: dict) -> str:
    summary = result.get("summary", {})
    lines = [
        f"Report metadata repair: {summary.get('repaired', 0)} repaired, "
        f"{summary.get('ok', 0)} unchanged, "
        f"{summary.get('missing', 0)} missing "
        f"({summary.get('total', 0)} total)"
    ]
    for item in result.get("entries", []):
        status = item.get("status", "?")
        if status == "repaired":
            lines.append(
                f"- {item.get('id', '?')} | repaired | "
                f"{item.get('old_size_bytes')} -> "
                f"{item.get('actual_size_bytes')} bytes | "
                f"{item.get('path', '')}")
        else:
            lines.append(
                f"- {item.get('id', '?')} | {status} | "
                f"{item.get('path', '')}")
    return "\n".join(lines)


def render_weekly_report_comparison(result: dict) -> str:
    left_id = result.get("left", {}).get("id", "?")
    right_id = result.get("right", {}).get("id", "?")
    lines = [f"Report comparison: {left_id} -> {right_id}"]
    for metric in result.get("metrics", []):
        delta = metric.get("delta", 0)
        sign = f"{delta:+d}" if isinstance(delta, int) else str(delta)
        lines.append(
            f"- {metric.get('label', metric.get('key', '?'))}: "
            f"{metric.get('left', 0)} -> {metric.get('right', 0)} "
            f"({sign})")
    if result.get("hash_available"):
        lines.append(
            "- SHA-256: " + ("equal" if result.get("hash_equal")
                             else "different"))
    else:
        lines.append("- SHA-256: unavailable")
    return "\n".join(lines)


def render_weekly_report_comparison_markdown(result: dict) -> str:
    left = result.get("left", {})
    right = result.get("right", {})
    left_id = left.get("id", "?")
    right_id = right.get("id", "?")
    lines = [
        f"# Report comparison: {left_id} -> {right_id}",
        "",
        "| Report | Exported | Path |",
        "| --- | --- | --- |",
        f"| {left_id} | {left.get('exported_at', '')} | "
        f"`{left.get('path', '')}` |",
        f"| {right_id} | {right.get('exported_at', '')} | "
        f"`{right.get('path', '')}` |",
        "",
        "## Metrics",
        "",
        "| Metric | Left | Right | Delta |",
        "| --- | ---: | ---: | ---: |",
    ]
    for metric in result.get("metrics", []):
        delta = metric.get("delta", 0)
        sign = f"{delta:+d}" if isinstance(delta, int) else str(delta)
        lines.append(
            f"| {metric.get('label', metric.get('key', '?'))} | "
            f"{metric.get('left', 0)} | {metric.get('right', 0)} | "
            f"{sign} |")
    lines.extend([
        "",
        "## Content hash",
        "",
    ])
    if result.get("hash_available"):
        lines.append(
            "- Equal: " + ("yes" if result.get("hash_equal") else "no"))
        lines.append(f"- {left_id}: `{left.get('sha256', '')}`")
        lines.append(f"- {right_id}: `{right.get('sha256', '')}`")
    else:
        lines.append("- SHA-256 metadata is unavailable.")
    lines.extend([
        "",
        "Generated locally by Atom Zero. No model weights were changed.",
    ])
    return "\n".join(lines) + "\n"


def render_weekly_report_preview(entries: list[dict], keep: int) -> str:
    lines = [
        f"Dry run: would remove {len(entries)} older report(s) to keep "
        f"{keep}."
    ]
    if entries:
        lines.extend(_weekly_report_line(item) for item in entries)
    else:
        lines.append("- None")
    return "\n".join(lines)


def write_markdown_atomic(path: str | Path, markdown: str) -> Path:
    """Write a sibling temp file, then replace the target atomically."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(markdown)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, target)
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
    return target


def render_weekly_markdown(review: dict) -> str:
    """Render a weekly review as a portable Markdown report."""
    days = review.get("period_days", 7)
    counts = review.get("task_counts", {})
    lines = [
        f"# Weekly review - {review.get('generated', '')}",
        "",
        f"Period: last {days} days",
        "",
        "## Tasks",
        "",
        f"- Open: {counts.get('open', 0)}",
        f"- Completed in the last {days} days: "
        f"{counts.get('completed_recent', 0)}",
        f"- Completed total: {counts.get('completed_total', 0)}",
        "",
        "### Open",
        "",
    ]
    open_tasks = review.get("open_tasks", [])
    if open_tasks:
        today = _today_from_now(review.get("generated"))
        lines.extend(
            f"- [ ] **#{task.get('id')}** {task.get('text', '')}"
            f"{_task_due_label(task, today)}"
            for task in open_tasks)
    else:
        lines.append("- None")
    overdue = review.get("overdue_tasks", [])
    due_soon = review.get("due_soon_tasks", [])
    if overdue:
        lines.extend(["", f"### Overdue ({len(overdue)})", ""])
        lines.extend(
            f"- [ ] **#{task.get('id')}** {task.get('text', '')} "
            f"(due {task.get('due', '')})"
            for task in overdue)
    if due_soon:
        lines.extend(["", f"### Due soon ({len(due_soon)})", ""])
        lines.extend(
            f"- [ ] **#{task.get('id')}** {task.get('text', '')} "
            f"(due {task.get('due', '')})"
            for task in due_soon)
    lines.extend([
        "",
        f"### Completed in the last {days} days",
        "",
    ])
    completed = review.get("completed_tasks", [])
    if completed:
        lines.extend(
            f"- [x] **#{task.get('id')}** {task.get('text', '')}"
            for task in completed)
    else:
        lines.append("- None")
    undated = review.get("completed_undated", [])
    if undated:
        lines.extend(["", "### Completed (undated)", ""])
        lines.extend(
            f"- [x] **#{task.get('id')}** {task.get('text', '')}"
            for task in undated)
    gaps = review.get("pending_gaps", [])
    lines.extend([
        "",
        f"## Pending gaps ({review.get('pending_gap_count', len(gaps))})",
        "",
    ])
    if gaps:
        lines.extend(f"- {gap.get('query', '')}" for gap in gaps)
    else:
        lines.append("- None")
    sources = review.get("recent_sources", [])
    lines.extend([
        "",
        f"## Recent sources "
        f"({review.get('recent_source_count', len(sources))})",
        "",
    ])
    if sources:
        for source in sources:
            title = source.get("title") or source.get("source", "")
            lines.append(
                f"- **{title}** - `{source.get('source', '')}` "
                f"({source.get('chunks', 0)} chunk(s))")
    else:
        lines.append("- None")
    graph = review.get("graph", {})
    entities = graph.get("entities", [])
    connections = graph.get("connections", [])
    lines.extend([
        "",
        "## Graph",
        "",
        f"- Entities shown: {len(entities)}",
        f"- Connections shown: {len(connections)}",
    ])
    if entities:
        lines.extend(["", "### Top entities", ""])
        lines.extend(
            f"- **{entity.get('name', '')}** - "
            f"{entity.get('mentions', 0)} mention(s), "
            f"{entity.get('sources', 0)} source(s)"
            for entity in entities)
    if connections:
        lines.extend(["", "### Connections", ""])
        lines.extend(
            f"- `{edge.get('source', '')}` <-> `{edge.get('target', '')}` - "
            f"{edge.get('sources', 0)} shared chunk(s)"
            for edge in connections)
    highlights = review.get("highlights", [])
    lines.extend(["", "## Highlights", ""])
    if highlights:
        lines.extend(f"- {item}" for item in highlights)
    else:
        lines.append("- None")
    next_week = review.get("next_week", [])
    lines.extend(["", "## Next week", ""])
    if next_week:
        lines.extend(f"- [ ] {item}" for item in next_week)
    else:
        lines.append("- None")
    lines.extend([
        "",
        "---",
        "",
        "Generated locally from Atom Zero tasks, gaps, sources and graph "
        "co-occurrence.",
        "No model weights were changed. This review is a local summary, not "
        "an autonomous decision.",
    ])
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="atomzero-brief",
        description="Local daily brief or goal plan for Atom Zero.")
    parser.add_argument("--version", action="version",
                        version=f"atomzero-brief {__version__}")
    parser.add_argument("--session", default="")
    parser.add_argument("--sessions-dir", default="data/sessions")
    parser.add_argument("--session-id", default="default")
    parser.add_argument("--docs", default="data/atomzero_demo.db")
    parser.add_argument("--gaps", default="data/knowledge_gaps.json")
    parser.add_argument("--goal", default="")
    parser.add_argument("--weekly", action="store_true",
                        help="build a weekly review instead of a daily brief")
    parser.add_argument("--overdue", action="store_true",
                        help="list overdue tasks from the session")
    parser.add_argument("--due-soon", action="store_true",
                        help="list tasks due in the next N days")
    parser.add_argument("--due-days", type=int, default=None,
                        help="due-soon window in days (default 7)")
    parser.add_argument("--search-tasks", default=None,
                        help="search task text in the session")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--markdown", action="store_true",
                        help="render the weekly review as Markdown")
    parser.add_argument("--out", default="",
                        help="write the Markdown review to PATH")
    parser.add_argument("--reports", action="store_true",
                        help="list recent archived weekly reports")
    parser.add_argument("--prune", action="store_true",
                        help="remove archived reports whose files are missing")
    parser.add_argument("--keep", type=int, default=None,
                        help="keep only the N newest archived reports")
    parser.add_argument("--dry-run", action="store_true",
                        help="preview --keep removals without changing the archive")
    parser.add_argument("--delete", metavar="ID", default=None,
                        help="remove one archive entry by report id")
    parser.add_argument("--query", default="",
                        help="filter archived reports by text")
    parser.add_argument("--sort", choices=("newest", "oldest", "largest"),
                        default=None,
                        help="sort archived reports")
    parser.add_argument("--verify", action="store_true",
                        help="verify archived report sizes and SHA-256 values")
    parser.add_argument("--repair", action="store_true",
                        help="recompute changed archived report metadata")
    parser.add_argument("--compare", nargs=2, metavar=("LEFT", "RIGHT"),
                        default=None,
                        help="compare metadata from two archived reports")
    parser.add_argument("--reports-index", default=DEFAULT_REPORTS_PATH,
                        help="weekly report archive JSON path")
    parser.add_argument("--limit", type=int, default=10,
                        help="number of archived reports to list")
    args = parser.parse_args(argv)
    if (args.markdown and not args.weekly
            and not (args.reports and args.compare)):
        parser.error("--markdown requires --weekly or --reports --compare")
    if args.markdown and args.json:
        parser.error("--markdown cannot be combined with --json")
    if args.out and not (args.weekly and args.markdown):
        parser.error("--out requires --weekly --markdown")
    if args.overdue and (args.weekly or args.goal or args.reports
                         or args.markdown or args.out):
        parser.error("--overdue cannot be combined with other views")
    if args.due_soon and (args.weekly or args.goal or args.reports
                          or args.markdown or args.out or args.overdue):
        parser.error("--due-soon cannot be combined with other views")
    if args.due_days is not None and not args.due_soon:
        parser.error("--due-days requires --due-soon")
    if args.due_days is not None and args.due_days < 1:
        parser.error("--due-days must be at least 1")
    if args.search_tasks is not None:
        if not args.search_tasks.strip():
            parser.error("--search-tasks needs a query")
        if (args.weekly or args.goal or args.reports or args.markdown
                or args.out or args.overdue or args.due_soon):
            parser.error("--search-tasks cannot be combined with other views")
    if args.reports and (
            args.weekly or (args.markdown and not args.compare)
            or args.out or args.goal):
        parser.error("--reports cannot be combined with brief/plan options")
    if args.prune and not args.reports:
        parser.error("--prune requires --reports")
    if args.keep is not None and not args.reports:
        parser.error("--keep requires --reports")
    if args.keep is not None and args.keep < 1:
        parser.error("--keep must be at least 1")
    if args.dry_run and args.keep is None:
        parser.error("--dry-run requires --keep")
    if args.dry_run and args.prune:
        parser.error("--dry-run cannot be combined with --prune")
    if args.delete is not None and not args.reports:
        parser.error("--delete requires --reports")
    if args.delete is not None and (
            args.prune or args.keep is not None or args.dry_run):
        parser.error("--delete cannot be combined with --prune/--keep")
    if args.query and not args.reports:
        parser.error("--query requires --reports")
    if args.query and (args.prune or args.keep is not None
                       or args.delete is not None or args.dry_run):
        parser.error("--query cannot be combined with maintenance options")
    if args.sort and not args.reports:
        parser.error("--sort requires --reports")
    if args.verify and not args.reports:
        parser.error("--verify requires --reports")
    if args.verify and (args.query or args.sort or args.prune
                        or args.keep is not None
                        or args.delete is not None or args.dry_run):
        parser.error("--verify cannot be combined with other report actions")
    if args.repair and not args.reports:
        parser.error("--repair requires --reports")
    if args.repair and (args.query or args.sort or args.prune
                        or args.keep is not None
                        or args.delete is not None or args.dry_run
                        or args.verify):
        parser.error("--repair cannot be combined with other report actions")
    if args.compare and not args.reports:
        parser.error("--compare requires --reports")
    if args.compare and (args.query or args.sort or args.prune
                         or args.keep is not None or args.delete is not None
                         or args.dry_run or args.verify or args.repair):
        parser.error("--compare cannot be combined with other report actions")
    if args.reports:
        index_path = args.reports_index or DEFAULT_REPORTS_PATH
        query = str(args.query or "").strip()
        sort = args.sort or "newest"
        removed = 0
        trimmed = 0
        preview = []
        deleted = None
        try:
            archive = WeeklyReportArchive(index_path)
            if args.compare:
                comparison = archive.compare(
                    args.compare[0], args.compare[1])
                if args.markdown:
                    print(render_weekly_report_comparison_markdown(
                        comparison), end="")
                elif args.json:
                    print(json.dumps({
                        **comparison,
                        "index": str(archive.path),
                    }, indent=2, ensure_ascii=False))
                else:
                    print(render_weekly_report_comparison(comparison))
                return 0
            if args.verify:
                verification = archive.verify()
                if args.json:
                    print(json.dumps({
                        "data": verification["entries"],
                        "count": len(verification["entries"]),
                        "total": verification["summary"]["total"],
                        "index": str(archive.path),
                        "verified_at": verification["verified_at"],
                        "verified": verification["ok"],
                        "verification": verification["summary"],
                    }, indent=2, ensure_ascii=False))
                else:
                    print(render_weekly_report_verification(verification))
                return 0
            if args.repair:
                repair = archive.repair()
                if args.json:
                    print(json.dumps({
                        "data": repair["entries"],
                        "count": len(repair["entries"]),
                        "total": repair["summary"]["total"],
                        "index": str(archive.path),
                        "repaired_at": repair["repaired_at"],
                        "repaired": repair["summary"]["repaired"],
                        "ok": repair["ok"],
                        "repair": repair["summary"],
                    }, indent=2, ensure_ascii=False))
                else:
                    print(render_weekly_report_repair(repair))
                return 0
            if args.delete is not None:
                deleted = archive.delete(args.delete)
            if args.prune:
                removed = archive.prune()
            if args.keep is not None:
                if args.dry_run:
                    preview = archive.keep_preview(args.keep)
                else:
                    trimmed = archive.keep(args.keep)
            entries = archive.list(
                args.limit, query=query, order=sort)
            total = archive.count()
            matched = archive.count(query=query)
        except (WeeklyReportError, OSError) as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)},
                                 indent=2))
            else:
                print(f"briefing error: {exc}")
            return 1
        if args.json:
            payload = {
                "data": entries,
                "count": len(entries),
                "total": total,
                "index": str(archive.path),
                "sort": sort,
            }
            if query:
                payload["query"] = query
                payload["matched"] = matched
            if args.delete is not None:
                payload["deleted"] = deleted
            if args.prune:
                payload["removed"] = removed
            if args.keep is not None:
                payload["keep"] = args.keep
                if args.dry_run:
                    payload["dry_run"] = True
                    payload["would_remove"] = preview
                    payload["would_remove_count"] = len(preview)
                else:
                    payload["trimmed"] = trimmed
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif args.dry_run:
            print(render_weekly_report_preview(preview, args.keep))
            print(f"Archive unchanged: {total} report(s).")
        else:
            if args.delete is not None:
                print(f"Deleted archive entry {deleted.get('id', '')}: "
                      f"{deleted.get('path', '')}")
                print("Report file was left on disk.")
            if args.prune:
                print(f"Removed {removed} missing report(s).")
            if args.keep is not None:
                print(f"Removed {trimmed} older report(s) to keep "
                      f"{args.keep}.")
            print(render_weekly_report_index(
                entries, archive.path, query=query,
                matched=matched, total=total, sort=sort))
        return 0
    session_path = Path(args.session) if args.session else (
        Path(args.sessions_dir) / f"{args.session_id}.json")
    docs = None
    try:
        session = load_session(session_path)
        if Path(args.docs).exists():
            docs = DocMemory(args.docs)
        gaps = None
        if Path(args.gaps).exists():
            from atomzero.knowledge_gaps import KnowledgeGapStore
            gaps = KnowledgeGapStore(args.gaps)
        if args.search_tasks is not None:
            result = task_search_view(session, args.search_tasks)
        elif args.overdue:
            result = overdue_view(session)
        elif args.due_soon:
            result = due_soon_view(
                session, days=args.due_days or 7)
        elif args.weekly:
            result = build_weekly_review(
                session, gaps, docs, days=args.days)
        elif args.goal:
            result = plan_goal(
                args.goal, docs, gaps, session, args.max_steps)
        else:
            result = build_briefing(session, gaps, docs)
    except (ValueError, OSError) as exc:
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, indent=2))
        else:
            print(f"briefing error: {exc}")
        return 1
    finally:
        if docs is not None:
            docs.close()
    if args.json:
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.search_tasks is not None:
        print(render_task_search_view(result))
    elif args.overdue:
        print(render_overdue_view(result))
    elif args.due_soon:
        print(render_due_soon_view(result))
    elif args.markdown:
        markdown = render_weekly_markdown(result)
        if args.out:
            try:
                target = write_markdown_atomic(args.out, markdown)
            except OSError as exc:
                print(f"briefing error: {exc}")
                return 1
            try:
                archive = WeeklyReportArchive(
                    args.reports_index or DEFAULT_REPORTS_PATH)
                entry = archive.record(target, result)
            except (WeeklyReportError, OSError) as exc:
                print(f"weekly review written to {target}")
                print(f"weekly report index error: {exc}")
                return 1
            print(f"weekly review written to {target} "
                  f"(archived {entry['id']})")
        else:
            print(markdown, end="")
    elif args.goal:
        print(render_plan(result))
    else:
        print(render_briefing(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
