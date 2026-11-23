"""Daily briefing and local goal planning for Atom Zero.

This module is deterministic by default: it reads local tasks, pending gaps,
document stats and the second-brain graph, then produces a short brief. Goal
planning adds cited local context and a few concrete next actions. It never
changes model weights and never calls a provider on its own.
"""
from __future__ import annotations

import argparse
import csv
import io
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


TASK_SEARCH_STATUSES = ("all", "open", "done")
TASK_SEARCH_DUE_FILTERS = ("all", "overdue", "today", "week", "none")
TASK_PRIORITIES = ("low", "normal", "high")
TASK_SEARCH_PRIORITIES = ("all",) + TASK_PRIORITIES
TASK_SORTS = ("default", "priority")
TASK_EXPORT_FORMATS = ("text", "csv", "markdown")


def _normalize_search_filters(status="all", due="all",
                              priority="all") -> tuple[str, str, str]:
    status = str(status or "all").strip().lower()
    due = str(due or "all").strip().lower()
    priority = str(priority or "all").strip().lower()
    if status not in TASK_SEARCH_STATUSES:
        raise ValueError(
            "status must be one of: " + ", ".join(TASK_SEARCH_STATUSES))
    if due not in TASK_SEARCH_DUE_FILTERS:
        raise ValueError(
            "due must be one of: " + ", ".join(TASK_SEARCH_DUE_FILTERS))
    if priority not in TASK_SEARCH_PRIORITIES:
        raise ValueError(
            "priority must be one of: " + ", ".join(TASK_SEARCH_PRIORITIES))
    return status, due, priority


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


def _task_priority_label(task: dict) -> str:
    priority = str(task.get("priority", "normal") or "normal").lower()
    return f" [{priority}]" if priority != "normal" else ""


def _task_tag_label(task: dict) -> str:
    tags = _task_tags(task)
    return f" [tags: {', '.join(tags)}]" if tags else ""


def _task_tags(task: dict) -> list[str]:
    values = task.get("tags", [])
    if isinstance(values, str):
        values = values.split(",")
    if not isinstance(values, (list, tuple)):
        return []
    return [str(value).strip().lower().lstrip("#")
            for value in values if str(value).strip()]


def _tag_counts(tasks) -> list[dict]:
    counts = {}
    for task in tasks or []:
        if not isinstance(task, dict):
            continue
        is_done = str(task.get("status", "open")) == "done"
        for tag in _task_tags(task):
            entry = counts.setdefault(
                tag, {"tag": tag, "open": 0, "done": 0, "total": 0})
            entry["done" if is_done else "open"] += 1
            entry["total"] += 1
    return sorted(counts.values(),
                  key=lambda item: (-item["total"], item["tag"]))


def _task_list_label(task: dict, today: str) -> str:
    due = str(task.get("due", "") or "").strip()
    priority = str(task.get("priority", "normal") or "normal").lower()
    priority_tag = f" [{priority}]" if priority != "normal" else ""
    tags = _task_tags(task)
    tag_text = f" [tags: {', '.join(tags)}]" if tags else ""
    suffix = ""
    if due:
        is_open = task.get("status", "open") == "open"
        overdue = ", overdue" if due < today and is_open else ""
        suffix = f" (due {due}{overdue})"
    return (f"{task.get('id')}{priority_tag} {task.get('text', '')}"
            f"{tag_text}{suffix}")


def _normalize_task_sort(sort="default") -> str:
    sort = str(sort or "default").strip().lower()
    if sort not in TASK_SORTS:
        raise ValueError("sort must be one of: " + ", ".join(TASK_SORTS))
    return sort


def sort_tasks(tasks, sort="default") -> list[dict]:
    """Return tasks in priority-first order when requested."""
    sort = _normalize_task_sort(sort)
    items = list(tasks or [])
    if sort == "default":
        return items
    ranks = {"high": 0, "normal": 1, "low": 2}

    def key(task: dict):
        priority = str(
            task.get("priority", "normal") or "normal").lower()
        due = str(task.get("due", "") or "").strip() or "9999-12-31"
        try:
            task_id = int(task.get("id", 0))
        except (TypeError, ValueError):
            task_id = 0
        return ranks.get(priority, 1), due, task_id

    return sorted(items, key=key)


def task_list_view(session_data: dict | None, status="open", due="all",
                   priority="all", sort="default", now=None,
                   tag="all") -> dict:
    """Filter and render a local task list for chat, HTTP and UI."""
    status = str(status or "open").strip().lower()
    status, due, priority = _normalize_search_filters(
        status, due, priority)
    tag = str(tag or "all").strip().lower().lstrip("#")
    sort = _normalize_task_sort(sort)
    today = _today_from_now(now)
    week_end = (date.fromisoformat(today) + timedelta(days=7)).isoformat()
    data = session_data if isinstance(session_data, dict) else {}
    rows = data.get("tasks", [])
    tasks = ([task for task in rows if isinstance(task, dict)]
             if isinstance(rows, list) else [])
    if status != "all":
        tasks = [task for task in tasks
                 if str(task.get("status", "open")) == status]
    if due == "overdue":
        tasks = [task for task in tasks
                 if task.get("status") == "open"
                 and str(task.get("due", "") or "").strip()
                 and str(task["due"]) < today]
    elif due == "today":
        tasks = [task for task in tasks
                 if str(task.get("due", "") or "").strip() == today]
    elif due == "week":
        tasks = [task for task in tasks
                 if str(task.get("due", "") or "").strip()
                 and today <= str(task["due"]) <= week_end]
    elif due == "none":
        tasks = [task for task in tasks
                 if not str(task.get("due", "") or "").strip()]
    if priority != "all":
        tasks = [task for task in tasks
                 if str(task.get("priority", "normal")
                        or "normal").lower() == priority]
    if tag != "all":
        tasks = [task for task in tasks if tag in _task_tags(task)]
    tasks = sort_tasks(tasks, sort)
    base = {"open": "Open tasks", "done": "Done tasks",
            "all": "All tasks"}[status]
    descriptors = []
    if due != "all":
        descriptors.append(f"due {due}")
    if priority != "all":
        descriptors.append(f"priority {priority}")
    if tag != "all":
        descriptors.append(f"tag {tag}")
    if sort != "default":
        descriptors.append("priority first")
    heading = base + (f" ({', '.join(descriptors)})"
                      if descriptors else "") + ": "
    filtered = (status != "open" or due != "all"
                or priority != "all" or tag != "all"
                or sort != "default")
    text = (heading + "; ".join(_task_list_label(task, today)
                                for task in tasks)
            if tasks else ("No tasks match." if filtered
                           else "There are no open tasks."))
    stamp = (now.strftime("%Y-%m-%d %H:%M")
             if isinstance(now, datetime)
             else str(now or time.strftime("%Y-%m-%d %H:%M")))
    return {
        "generated": stamp,
        "status": status,
        "due": due,
        "priority": priority,
        "tag": tag,
        "sort": sort,
        "count": len(tasks),
        "heading": heading,
        "tasks": tasks,
        "text": text,
        "weights_changed": False,
    }


def _task_overdue(task: dict, today: str) -> bool:
    due = str(task.get("due", "") or "").strip()
    return (task.get("status", "open") == "open"
            and bool(due) and due < today)


def render_task_list_csv(view: dict) -> str:
    today = _today_from_now(view.get("generated"))
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow([
        "id", "status", "priority", "tags", "text", "due", "overdue",
        "created", "completed",
    ])
    for task in view.get("tasks", []):
        writer.writerow([
            task.get("id", ""),
            task.get("status", ""),
            task.get("priority", "normal"),
            ",".join(_task_tags(task)),
            task.get("text", ""),
            task.get("due", ""),
            "true" if _task_overdue(task, today) else "false",
            task.get("created", ""),
            task.get("completed", ""),
        ])
    return output.getvalue()


def _markdown_cell(value) -> str:
    return str(value or "").replace("|", "\\|").replace("\n", " ")


def render_task_list_markdown(view: dict) -> str:
    today = _today_from_now(view.get("generated"))
    lines = [
        f"# Task list - {view.get('generated', '')}",
        "",
        f"- Status: {view.get('status', 'open')}",
        f"- Due: {view.get('due', 'all')}",
        f"- Priority: {view.get('priority', 'all')}",
        f"- Tag: {view.get('tag', 'all')}",
        f"- Sort: {view.get('sort', 'default')}",
        f"- Count: {view.get('count', 0)}",
        "",
        "| ID | Status | Priority | Tags | Task | Due | Overdue |",
        "|---:|---|---|---|---|---|---|",
    ]
    for task in view.get("tasks", []):
        lines.append(
            f"| {_markdown_cell(task.get('id'))} "
            f"| {_markdown_cell(task.get('status'))} "
            f"| {_markdown_cell(task.get('priority', 'normal'))} "
            f"| {_markdown_cell(','.join(_task_tags(task)))} "
            f"| {_markdown_cell(task.get('text'))} "
            f"| {_markdown_cell(task.get('due'))} "
            f"| {'yes' if _task_overdue(task, today) else 'no'} |")
    if not view.get("tasks"):
        lines.extend(["", "_No tasks match._"])
    return "\n".join(lines) + "\n"


def task_tag_view(session_data: dict | None, now=None) -> dict:
    """Count task tags across open and done tasks."""
    data = session_data if isinstance(session_data, dict) else {}
    rows = data.get("tasks", [])
    tasks = ([task for task in rows if isinstance(task, dict)]
             if isinstance(rows, list) else [])
    tagged_tasks = 0
    open_tagged = 0
    done_tagged = 0
    for task in tasks:
        tags = _task_tags(task)
        is_done = str(task.get("status", "open")) == "done"
        if tags:
            tagged_tasks += 1
            if is_done:
                done_tagged += 1
            else:
                open_tagged += 1
    tags = _tag_counts(tasks)
    stamp = (now.strftime("%Y-%m-%d %H:%M")
             if isinstance(now, datetime)
             else str(now or time.strftime("%Y-%m-%d %H:%M")))
    if not tasks:
        text = "There are no tasks."
    elif not tags:
        text = ("Task tags: none | 0 tagged tasks. Add one with "
                "'add task ... tags work'.")
    else:
        text = (f"Task tags: {len(tags)} distinct | {tagged_tasks} tagged "
                f"task(s) ({open_tagged} open, {done_tagged} done) | "
                f"{len(tasks) - tagged_tasks} untagged")
        text += "\n" + "\n".join(
            f"- {item['tag']}: {item['open']} open, {item['done']} done, "
            f"{item['total']} total" for item in tags)
    return {
        "generated": stamp,
        "tag_count": len(tags),
        "task_count": len(tasks),
        "tagged_task_count": tagged_tasks,
        "untagged_task_count": len(tasks) - tagged_tasks,
        "open_tagged": open_tagged,
        "done_tagged": done_tagged,
        "tags": tags,
        "text": text,
        "weights_changed": False,
    }


def render_task_tag_view(view: dict) -> str:
    return str(view.get("text", ""))


def normalize_task_tag(value) -> str:
    tag = str(value or "").strip().lower().lstrip("#")
    if not tag:
        raise ValueError("tag is required")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,23}", tag):
        raise ValueError(
            "tags must use letters, numbers, - or _ (max 24 chars)")
    return tag


def normalize_tag_rename(old, new) -> tuple[str, str]:
    old = normalize_task_tag(old)
    new = normalize_task_tag(new)
    if old == new:
        raise ValueError("new tag must be different from old tag")
    return old, new


def rename_task_tag(session_data: dict | None, old, new) -> dict:
    """Rename one tag on every task, preserving other tags and order."""
    old, new = normalize_tag_rename(old, new)
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("session tasks must be a list")
    updated = []
    stamp = time.strftime("%Y-%m-%dT%H:%M:%S")
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tags = _task_tags(task)
        if old not in tags:
            continue
        replaced = []
        for tag in tags:
            value = new if tag == old else tag
            if value not in replaced:
                replaced.append(value)
        task["tags"] = replaced
        task["renamed"] = stamp
        try:
            updated.append(int(task.get("id")))
        except (TypeError, ValueError):
            continue
    if not updated:
        raise ValueError(f"tag '{old}' is not used")
    return {
        "old": old,
        "new": new,
        "updated": len(updated),
        "task_ids": updated,
    }


def set_task_priority(session_data: dict | None, tid, priority) -> dict:
    """Update one task's priority in a session dict."""
    try:
        tid = int(tid)
    except (TypeError, ValueError) as exc:
        raise ValueError("task ID must be an integer") from exc
    priority = str(priority or "").strip().lower()
    if priority not in TASK_PRIORITIES:
        raise ValueError(
            "priority must be one of: " + ", ".join(TASK_PRIORITIES))
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        raise ValueError("session tasks must be a list")
    for task in tasks:
        if not isinstance(task, dict):
            continue
        try:
            task_id = int(task.get("id", 0))
        except (TypeError, ValueError):
            continue
        if task_id == tid:
            task["priority"] = priority
            task["prioritized"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            return task
    raise ValueError(f"task {tid} not found")


def _overdue_tasks(session_data: dict | None,
                   today: str | None = None) -> list[dict]:
    today = today or date.today().isoformat()
    return [task for task in _open_tasks(session_data)
            if str(task.get("due", "") or "").strip()
            and str(task["due"]) < today]


def overdue_view(session_data: dict | None, now=None,
                 sort="default", tag="all") -> dict:
    sort = _normalize_task_sort(sort)
    tag = str(tag or "all").strip().lower().lstrip("#")
    if tag in ("", "all"):
        tag = "all"
    else:
        tag = normalize_task_tag(tag)
    tasks = _open_tasks(session_data)
    if tag != "all":
        tasks = [task for task in tasks if tag in _task_tags(task)]
    today = _today_from_now(now)
    overdue = sort_tasks(
        [task for task in tasks
         if str(task.get("due", "") or "").strip()
         and str(task["due"]) < today], sort)
    return {
        "generated": (now.strftime("%Y-%m-%d %H:%M")
                      if isinstance(now, datetime)
                      else str(now or time.strftime("%Y-%m-%d %H:%M"))),
        "today": today,
        "sort": sort,
        "tag": tag,
        "open_task_count": len(tasks),
        "overdue_task_count": len(overdue),
        "open_tasks": tasks,
        "overdue_tasks": overdue,
        "weights_changed": False,
    }


def render_overdue_view(view: dict) -> str:
    header = (
        f"Overdue tasks ({view.get('overdue_task_count', 0)} of "
        f"{view.get('open_task_count', 0)} open) - {view.get('today', '')}")
    if view.get("tag", "all") != "all":
        header += f" | tag={view['tag']}"
    if view.get("sort", "default") != "default":
        header += f" | sort={view['sort']}"
    lines = [header]
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
                  now=None, sort="default", tag="all") -> dict:
    sort = _normalize_task_sort(sort)
    tag = str(tag or "all").strip().lower().lstrip("#")
    if tag in ("", "all"):
        tag = "all"
    else:
        tag = normalize_task_tag(tag)
    try:
        days = max(1, min(int(days), 90))
    except (TypeError, ValueError):
        days = 7
    today = _today_from_now(now)
    end = (date.fromisoformat(today) + timedelta(days=days)).isoformat()
    tasks = _open_tasks(session_data)
    if tag != "all":
        tasks = [task for task in tasks if tag in _task_tags(task)]
    due = sort_tasks(
        [task for task in tasks
         if str(task.get("due", "") or "").strip()
         and today <= str(task["due"]) <= end], sort)
    return {
        "generated": (now.strftime("%Y-%m-%d %H:%M")
                      if isinstance(now, datetime)
                      else str(now or time.strftime("%Y-%m-%d %H:%M"))),
        "today": today,
        "window_end": end,
        "days": days,
        "sort": sort,
        "tag": tag,
        "open_task_count": len(tasks),
        "due_soon_count": len(due),
        "open_tasks": tasks,
        "due_soon_tasks": due,
        "weights_changed": False,
    }


def render_due_soon_view(view: dict) -> str:
    header = (
        f"Due soon ({view.get('due_soon_count', 0)} of "
        f"{view.get('open_task_count', 0)} open, next "
        f"{view.get('days', 7)} days) - {view.get('today', '')} "
        f"to {view.get('window_end', '')}")
    if view.get("tag", "all") != "all":
        header += f" | tag={view['tag']}"
    if view.get("sort", "default") != "default":
        header += f" | sort={view['sort']}"
    lines = [header]
    tasks = view.get("due_soon_tasks", [])
    if not tasks:
        lines.append("- None")
        return "\n".join(lines)
    for task in tasks:
        lines.append(
            f"- {task.get('id')}: {task.get('text', '')} | "
            f"due {task.get('due', '')}")
    return "\n".join(lines)


def search_tasks(session_data: dict | None, query: str,
                 status: str = "all", due: str = "all",
                 priority: str = "all", tag: str = "all",
                 today=None) -> list[dict]:
    tokens = [token for token in str(query or "").strip().lower().split()
              if token]
    if not tokens:
        return []
    status, due, priority = _normalize_search_filters(status, due, priority)
    tag = str(tag or "all").strip().lower().lstrip("#")
    today = _today_from_now(today)
    week_end = (date.fromisoformat(today) + timedelta(days=7)).isoformat()
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    matches = []
    for task in tasks:
        if not isinstance(task, dict):
            continue
        text = str(task.get("text", "")).lower()
        if not all(token in text for token in tokens):
            continue
        task_status = str(task.get("status", "open") or "open")
        if status != "all" and task_status != status:
            continue
        task_due = str(task.get("due", "") or "").strip()
        if due == "overdue":
            if task_status != "open" or not task_due or task_due >= today:
                continue
        elif due == "today":
            if task_due != today:
                continue
        elif due == "week":
            if not task_due or task_due < today or task_due > week_end:
                continue
        elif due == "none":
            if task_due:
                continue
        task_priority = str(
            task.get("priority", "normal") or "normal").lower()
        if priority != "all" and task_priority != priority:
            continue
        if tag != "all" and tag not in _task_tags(task):
            continue
        matches.append(task)
    return matches


def task_search_view(session_data: dict | None, query: str,
                     status: str = "all", due: str = "all",
                     priority: str = "all", sort="default",
                     tag: str = "all", now=None) -> dict:
    query = str(query or "").strip()
    status, due, priority = _normalize_search_filters(
        status, due, priority)
    tag = str(tag or "all").strip().lower().lstrip("#")
    sort = _normalize_task_sort(sort)
    today = _today_from_now(now)
    matches = sort_tasks(search_tasks(
        session_data, query, status=status, due=due,
        priority=priority, tag=tag, today=today), sort)
    return {
        "generated": (now.strftime("%Y-%m-%d %H:%M")
                      if isinstance(now, datetime)
                      else str(now or time.strftime("%Y-%m-%d %H:%M"))),
        "today": today,
        "query": query,
        "status": status,
        "due": due,
        "priority": priority,
        "tag": tag,
        "sort": sort,
        "count": len(matches),
        "tasks": matches,
        "open_task_count": len(_open_tasks(session_data)),
        "weights_changed": False,
    }


def render_task_search_view(view: dict) -> str:
    query = view.get("query", "")
    header = f'Task search "{query}" ({view.get("count", 0)} match(es))'
    filters = []
    if view.get("status", "all") != "all":
        filters.append(f"status={view['status']}")
    if view.get("due", "all") != "all":
        filters.append(f"due={view['due']}")
    if view.get("priority", "all") != "all":
        filters.append(f"priority={view['priority']}")
    if view.get("tag", "all") != "all":
        filters.append(f"tag={view['tag']}")
    if view.get("sort", "default") != "default":
        filters.append(f"sort={view['sort']}")
    if filters:
        header += " | " + " | ".join(filters)
    lines = [header]
    tasks = view.get("tasks", [])
    if not tasks:
        lines.append("- None")
        return "\n".join(lines)
    today = _today_from_now(view.get("generated"))
    for task in tasks:
        lines.append(
            f"- {task.get('id')} [{task.get('status', '?')}] "
            f"{task.get('text', '')}{_task_priority_label(task)}"
            f"{_task_tag_label(task)}"
            f"{_task_due_label(task, today)}")
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
                   now=None, sort="default", tag="all") -> dict:
    """Build a compact local brief from tasks, gaps, docs and the graph."""
    sort = _normalize_task_sort(sort)
    tag = str(tag or "all").strip().lower().lstrip("#")
    if tag in ("", "all"):
        tag = "all"
    else:
        tag = normalize_task_tag(tag)
    tasks = sort_tasks(_open_tasks(session_data), sort)
    if tag != "all":
        tasks = [task for task in tasks if tag in _task_tags(task)]
    today = _today_from_now(now)
    week_end = (date.fromisoformat(today) + timedelta(days=7)).isoformat()
    overdue = sort_tasks(
        [task for task in tasks
         if str(task.get("due", "") or "").strip()
         and str(task["due"]) < today], sort)
    due_soon = sort_tasks(
        [task for task in tasks
         if str(task.get("due", "") or "").strip()
         and today <= str(task["due"]) <= week_end], sort)
    brief_tags = _tag_counts(tasks)
    tagged_tasks = sum(1 for task in tasks if _task_tags(task))
    pending = _pending_gaps(gaps)
    sources = docs.sources() if docs is not None else []
    chunks = docs.stats().get("chunks", 0) if docs is not None else 0
    graph = _graph_snapshot(docs)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if tag != "all" and not tasks:
        focus = f"no open tasks tagged #{tag}"
    elif overdue:
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
    title = f"Daily brief - {stamp}"
    if sort != "default":
        title += f" | sort={sort}"
    if tag != "all":
        title += f" | tag={tag}"
    lines = [
        title,
        f"Focus: {focus}",
        task_summary,
    ]
    tag_scope = f" for #{tag}" if tag != "all" else ""
    if brief_tags:
        preview = " | ".join(
            f"{item['tag']} ({item['open']} open)"
            for item in brief_tags[:5])
        lines.append(f"Tags{tag_scope}: {len(brief_tags)} distinct | "
                     f"{preview}")
    elif tag != "all":
        lines.append(f"Tags{tag_scope}: none")
    lines.extend(
        f"- {task.get('id')}: {task.get('text', '')}"
        f"{_task_priority_label(task)}"
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
        "sort": sort,
        "tag": tag,
        "focus": focus,
        "tasks": tasks[:5],
        "open_task_count": len(tasks),
        "overdue_tasks": overdue[:5],
        "overdue_task_count": len(overdue),
        "due_soon_tasks": due_soon[:5],
        "due_soon_count": len(due_soon),
        "tags": brief_tags[:8],
        "tag_count": len(brief_tags),
        "tagged_task_count": tagged_tasks,
        "untagged_task_count": len(tasks) - tagged_tasks,
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
                        days: int = 7, now=None, sort="default",
                        tag="all") -> dict:
    """Aggregate tasks, gaps, recent sources and graph highlights."""
    sort = _normalize_task_sort(sort)
    tag = str(tag or "all").strip().lower().lstrip("#")
    if tag in ("", "all"):
        tag = "all"
    else:
        tag = normalize_task_tag(tag)
    try:
        days = max(1, min(int(days), 90))
    except (TypeError, ValueError):
        days = 7
    stamp = (now.strftime("%Y-%m-%d %H:%M")
             if isinstance(now, datetime)
             else str(now or time.strftime("%Y-%m-%d %H:%M")))
    cutoff = datetime.now() - timedelta(days=days)
    cutoff_ts = time.time() - days * 86400
    open_tasks = sort_tasks(_open_tasks(session_data), sort)
    done_tasks = _done_tasks(session_data)
    if tag != "all":
        open_tasks = [task for task in open_tasks
                      if tag in _task_tags(task)]
        done_tasks = [task for task in done_tasks
                      if tag in _task_tags(task)]
    today = _today_from_now(now)
    week_end = (date.fromisoformat(today) + timedelta(days=7)).isoformat()
    overdue = sort_tasks(
        [task for task in open_tasks
         if str(task.get("due", "") or "").strip()
         and str(task["due"]) < today], sort)
    due_soon = sort_tasks(
        [task for task in open_tasks
         if str(task.get("due", "") or "").strip()
         and today <= str(task["due"]) <= week_end], sort)
    completed_recent = [task for task in done_tasks
                        if _task_recent(task.get("completed"), cutoff)]
    completed_undated = [task for task in done_tasks
                         if not task.get("completed")]
    active_tasks = open_tasks + completed_recent
    weekly_tags = _tag_counts(active_tasks)
    tagged_active = sum(1 for task in active_tasks if _task_tags(task))
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
    title = f"Weekly review - {stamp} (last {days} days)"
    if sort != "default":
        title += f" | sort={sort}"
    if tag != "all":
        title += f" | tag={tag}"
    lines = [
        title,
        f"Tasks: {len(open_tasks)} open | {len(completed_recent)} "
        f"completed {period_label}",
    ]
    if overdue:
        lines.append(f"Overdue: {len(overdue)}")
    if due_soon:
        lines.append(f"Due soon: {len(due_soon)} (next 7 days)")
    tag_scope = f" for #{tag}" if tag != "all" else ""
    if weekly_tags:
        preview = " | ".join(
            f"{item['tag']} ({item['open']} open, {item['done']} done)"
            for item in weekly_tags[:5])
        lines.append(f"Tags{tag_scope}: {len(weekly_tags)} distinct | "
                     f"{preview}")
    else:
        lines.append(f"Tags{tag_scope}: none")
    lines.append("Open:")
    lines.extend(
        f"- {task.get('id')}: {task.get('text', '')}"
        f"{_task_priority_label(task)}"
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
        "sort": sort,
        "period_days": days,
        "tag": tag,
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
        "tags": weekly_tags[:8],
        "tag_count": len(weekly_tags),
        "tagged_task_count": tagged_active,
        "untagged_task_count": len(active_tasks) - tagged_active,
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
    tag = str(item.get("tag", "all") or "all")
    tag_part = f" | tag={tag}" if tag != "all" else ""
    return (
        f"- {item.get('id', '?')} | {item.get('exported_at', '')} | "
        f"last {item.get('period_days', 7)} days{tag_part} | "
        f"{counts.get('open', 0)} open | "
        f"{item.get('recent_source_count', 0)} source(s) | "
        f"{item.get('path', '')}{marker}")


def render_weekly_report_index(entries: list[dict],
                               index_path: str | Path,
                               query: str = "",
                               matched: int | None = None,
                               total: int | None = None,
                               sort: str = "newest",
                               tag: str = "all") -> str:
    heading = f"Weekly report archive: {index_path}"
    filters = []
    if query:
        filters.append(f"query: {query}")
    if tag != "all":
        filters.append(f"tag: {tag}")
    if filters:
        heading += " | " + " | ".join(filters)
        heading += (f" | {matched if matched is not None else len(entries)} "
                    f"match(es) of {total if total is not None else '?'}")
    heading += f" | sort: {sort}"
    lines = [heading]
    if not entries:
        if query:
            lines.append(f'No archived reports match "{query}".')
        elif tag != "all":
            lines.append(f"No archived reports tagged '{tag}'.")
        else:
            lines.append("No archived weekly reports.")
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


def write_text_atomic(path: str | Path, text: str) -> Path:
    """Write a sibling temp file, then replace the target atomically."""
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            fd = -1
            handle.write(text)
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


def write_markdown_atomic(path: str | Path, markdown: str) -> Path:
    return write_text_atomic(path, markdown)


def render_weekly_markdown(review: dict) -> str:
    """Render a weekly review as a portable Markdown report."""
    days = review.get("period_days", 7)
    counts = review.get("task_counts", {})
    title = f"# Weekly review - {review.get('generated', '')}"
    if review.get("sort", "default") != "default":
        title += f" | sort={review['sort']}"
    tag = str(review.get("tag", "all") or "all")
    if tag != "all":
        title += f" | tag={tag}"
    lines = [
        title,
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
            f"{_task_priority_label(task)}"
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
    tags = review.get("tags", [])
    untagged = review.get("untagged_task_count", 0)
    tag_count = review.get("tag_count", len(tags))
    tag_word = "tag" if tag_count == 1 else "tags"
    tag_label = f" for #{tag}" if tag != "all" else ""
    if tag != "all":
        heading = f"## Tags{tag_label} ({tag_count} {tag_word})"
    else:
        heading = f"## Tags ({tag_count} {tag_word}, {untagged} untagged)"
    lines.extend(["", heading, ""])
    if tags:
        lines.extend([
            "| Tag | Open | Done | Total |",
            "| --- | ---: | ---: | ---: |",
        ])
        lines.extend(
            f"| {_markdown_cell(item.get('tag'))} | {item.get('open', 0)} "
            f"| {item.get('done', 0)} | {item.get('total', 0)} |"
            for item in tags)
        lines.extend([
            "",
            f"- Tagged tasks: {review.get('tagged_task_count', 0)}",
        ])
    else:
        lines.append("- None")
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
    parser.add_argument("--weekly-tag", default=None,
                        help="filter a weekly review to one task tag")
    parser.add_argument("--brief-tag", default=None,
                        help="filter the daily brief to one task tag")
    parser.add_argument("--overdue", action="store_true",
                        help="list overdue tasks from the session")
    parser.add_argument("--due-soon", action="store_true",
                        help="list tasks due in the next N days")
    parser.add_argument("--due-days", type=int, default=None,
                        help="due-soon window in days (default 7)")
    parser.add_argument("--overdue-tag", default=None,
                        help="filter overdue tasks by tag")
    parser.add_argument("--due-soon-tag", default=None,
                        help="filter due-soon tasks by tag")
    parser.add_argument("--search-tasks", default=None,
                        help="search task text in the session")
    parser.add_argument("--search-status", choices=TASK_SEARCH_STATUSES,
                        default="all",
                        help="filter task search by status")
    parser.add_argument("--search-due", choices=TASK_SEARCH_DUE_FILTERS,
                        default="all",
                        help="filter task search by due date")
    parser.add_argument("--search-priority", choices=TASK_SEARCH_PRIORITIES,
                        default="all",
                        help="filter task search by priority")
    parser.add_argument("--search-tag", default=None,
                        help="filter task search by tag")
    parser.add_argument("--set-task-priority", nargs=2,
                        metavar=("ID", "LEVEL"), default=None,
                        help="set one task priority to low, normal or high")
    parser.add_argument("--task-sort", choices=TASK_SORTS,
                        default="default",
                        help="sort tasks in brief, review and task views")
    parser.add_argument("--task-list", action="store_true",
                        help="render a filtered task list")
    parser.add_argument("--list-tags", action="store_true",
                        help="count task tags across open and done tasks")
    parser.add_argument("--rename-tag", nargs=2, metavar=("OLD", "NEW"),
                        default=None,
                        help="preview renaming one task tag everywhere")
    parser.add_argument("--confirm-rename-tag", action="store_true",
                        help="apply the --rename-tag preview")
    parser.add_argument("--task-status", choices=TASK_SEARCH_STATUSES,
                        default=None, help="task-list status filter")
    parser.add_argument("--task-due", choices=TASK_SEARCH_DUE_FILTERS,
                        default=None, help="task-list due-date filter")
    parser.add_argument("--task-priority", choices=TASK_SEARCH_PRIORITIES,
                        default=None, help="task-list priority filter")
    parser.add_argument("--task-tag", default=None,
                        help="task-list tag filter")
    parser.add_argument("--task-format", choices=TASK_EXPORT_FORMATS,
                        default="text",
                        help="task-list output: text, csv or markdown")
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
    parser.add_argument("--tag", default=None,
                        help="filter archived reports by exact tag")
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
    if args.overdue_tag is not None and not args.overdue:
        parser.error("--overdue-tag requires --overdue")
    if args.due_soon_tag is not None and not args.due_soon:
        parser.error("--due-soon-tag requires --due-soon")
    if args.weekly_tag is not None and not args.weekly:
        parser.error("--weekly-tag requires --weekly")
    if args.brief_tag is not None and (
            args.weekly or args.task_list or args.list_tags
            or args.rename_tag is not None
            or args.search_tasks is not None
            or args.set_task_priority is not None
            or args.overdue or args.due_soon or args.goal
            or args.reports or args.markdown or args.out):
        parser.error("--brief-tag cannot be combined with other views")
    if args.task_sort != "default" and (args.goal or args.reports):
        parser.error("--task-sort cannot be combined with goal/report views")
    if not args.task_list:
        if (args.task_status is not None or args.task_due is not None
                or args.task_priority is not None
                or args.task_tag is not None
                or args.task_format != "text"):
            parser.error(
                "--task-status/--task-due/--task-priority/--task-tag/"
                "--task-format require --task-list")
    else:
        if (args.weekly or args.goal or args.reports or args.markdown
                or args.out or args.overdue or args.due_soon
                or args.search_tasks is not None
                or args.set_task_priority is not None):
            parser.error("--task-list cannot be combined with other views")
        if args.json and args.task_format != "text":
            parser.error(
                "--json cannot be combined with --task-format csv/markdown")
    if args.list_tags and (
            args.task_list or args.weekly or args.goal or args.reports
            or args.markdown or args.out or args.overdue or args.due_soon
            or args.search_tasks is not None
            or args.set_task_priority is not None
            or args.task_sort != "default"):
        parser.error("--list-tags cannot be combined with other views")
    if args.confirm_rename_tag and args.rename_tag is None:
        parser.error("--confirm-rename-tag requires --rename-tag")
    if args.rename_tag is not None and (
            args.task_list or args.list_tags or args.weekly or args.goal
            or args.reports or args.markdown or args.out or args.overdue
            or args.due_soon or args.search_tasks is not None
            or args.set_task_priority is not None
            or args.task_sort != "default"):
        parser.error("--rename-tag cannot be combined with other views")
    if args.set_task_priority is not None:
        raw_id, raw_priority = args.set_task_priority
        try:
            int(raw_id)
        except ValueError:
            parser.error("--set-task-priority ID must be an integer")
        if raw_priority.strip().lower() not in TASK_PRIORITIES:
            parser.error(
                "--set-task-priority level must be low, normal or high")
        if (args.weekly or args.goal or args.reports or args.markdown
                or args.out or args.overdue or args.due_soon
                or args.search_tasks is not None
                or args.task_sort != "default"):
            parser.error(
                "--set-task-priority cannot be combined with other views")
    if args.search_tasks is None and (
            args.search_status != "all" or args.search_due != "all"
            or args.search_priority != "all"
            or args.search_tag is not None):
        parser.error(
            "--search-status/--search-due/--search-priority/--search-tag "
            "require --search-tasks")
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
    if args.tag and not args.reports:
        parser.error("--tag requires --reports")
    if args.tag and (
            args.prune or args.keep is not None or args.delete is not None
            or args.dry_run or args.verify or args.repair or args.compare):
        parser.error("--tag cannot be combined with maintenance options")
    if args.tag:
        try:
            normalize_task_tag(args.tag)
        except ValueError as exc:
            parser.error(str(exc))
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
        tag = str(args.tag or "all").strip().lower().lstrip("#") or "all"
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
                args.limit, query=query, order=sort, tag=tag)
            total = archive.count()
            matched = archive.count(query=query, tag=tag)
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
            if tag != "all":
                payload["tag"] = tag
            if query or tag != "all":
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
                matched=matched, total=total, sort=sort, tag=tag))
        return 0
    session_path = Path(args.session) if args.session else (
        Path(args.sessions_dir) / f"{args.session_id}.json")
    if args.rename_tag is not None:
        old_raw, new_raw = args.rename_tag
        try:
            old, new = normalize_tag_rename(old_raw, new_raw)
            session = load_session(session_path)
            if not session:
                raise ValueError(f"cannot read session: {session_path}")
            if args.confirm_rename_tag:
                result = rename_task_tag(session, old, new)
                write_text_atomic(
                    session_path,
                    json.dumps(session, indent=2, ensure_ascii=False) + "\n")
                payload = {
                    "applied": True, **result,
                    "session": str(session_path),
                }
            else:
                view = task_tag_view(session)
                entry = next((item for item in view["tags"]
                              if item["tag"] == old), None)
                if entry is None:
                    raise ValueError(f"tag '{old}' is not used")
                payload = {
                    "applied": False,
                    "old": old,
                    "new": new,
                    "would_update": entry["total"],
                    "open": entry["open"],
                    "done": entry["done"],
                    "session": str(session_path),
                }
        except (ValueError, OSError) as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)},
                                 indent=2, ensure_ascii=False))
            else:
                print(f"briefing error: {exc}")
            return 1
        if args.json:
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        elif payload["applied"]:
            print(f"Renamed tag '{old}' to '{new}' on "
                  f"{payload['updated']} task(s).")
        else:
            print(f"Would rename tag '{old}' to '{new}' on "
                  f"{payload['would_update']} task(s) "
                  f"({payload['open']} open, {payload['done']} done); "
                  "rerun with --confirm-rename-tag to apply.")
        return 0
    if args.list_tags:
        try:
            session = load_session(session_path)
            result = task_tag_view(session)
        except (ValueError, OSError) as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)},
                                 indent=2, ensure_ascii=False))
            else:
                print(f"briefing error: {exc}")
            return 1
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(result["text"])
        return 0
    if args.task_list:
        status = args.task_status or "open"
        due = args.task_due or "all"
        priority = args.task_priority or "all"
        tag = args.task_tag or "all"
        try:
            session = load_session(session_path)
            result = task_list_view(
                session, status=status, due=due, priority=priority,
                tag=tag, sort=args.task_sort)
        except (ValueError, OSError) as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)},
                                 indent=2, ensure_ascii=False))
            else:
                print(f"briefing error: {exc}")
            return 1
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        elif args.task_format == "csv":
            print(render_task_list_csv(result), end="")
        elif args.task_format == "markdown":
            print(render_task_list_markdown(result), end="")
        else:
            print(result["text"])
        return 0
    if args.set_task_priority is not None:
        raw_id, raw_priority = args.set_task_priority
        try:
            session = load_session(session_path)
            if not session:
                raise ValueError(f"cannot read session: {session_path}")
            task = set_task_priority(session, raw_id, raw_priority)
            write_text_atomic(
                session_path,
                json.dumps(session, indent=2, ensure_ascii=False) + "\n")
        except (ValueError, OSError) as exc:
            if args.json:
                print(json.dumps({"ok": False, "error": str(exc)},
                                 indent=2, ensure_ascii=False))
            else:
                print(f"briefing error: {exc}")
            return 1
        if args.json:
            print(json.dumps({"task": task}, indent=2, ensure_ascii=False))
        else:
            print(f"Task {task['id']} priority set to {task['priority']}.")
        return 0
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
            result = task_search_view(
                session, args.search_tasks,
                status=args.search_status, due=args.search_due,
                priority=args.search_priority, sort=args.task_sort,
                tag=args.search_tag or "all")
        elif args.overdue:
            result = overdue_view(
                session, sort=args.task_sort,
                tag=args.overdue_tag or "all")
        elif args.due_soon:
            result = due_soon_view(
                session, days=args.due_days or 7, sort=args.task_sort,
                tag=args.due_soon_tag or "all")
        elif args.weekly:
            result = build_weekly_review(
                session, gaps, docs, days=args.days,
                sort=args.task_sort, tag=args.weekly_tag or "all")
        elif args.goal:
            result = plan_goal(
                args.goal, docs, gaps, session, args.max_steps)
        else:
            result = build_briefing(
                session, gaps, docs, sort=args.task_sort,
                tag=args.brief_tag or "all")
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
