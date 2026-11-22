"""Daily briefing and local goal planning for Atom Zero.

This module is deterministic by default: it reads local tasks, pending gaps,
document stats and the second-brain graph, then produces a short brief. Goal
planning adds cited local context and a few concrete next actions. It never
changes model weights and never calls a provider on its own.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from atomzero import __version__
from atomzero.doc_memory import (
    DocMemory, best_snippet, format_citation, retrieval_confidence)
from atomzero.knowledge_graph import build_graph


def _open_tasks(session_data: dict | None) -> list[dict]:
    data = session_data if isinstance(session_data, dict) else {}
    tasks = data.get("tasks", [])
    if not isinstance(tasks, list):
        return []
    return [task for task in tasks
            if isinstance(task, dict) and task.get("status") == "open"]


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
    pending = _pending_gaps(gaps)
    sources = docs.sources() if docs is not None else []
    chunks = docs.stats().get("chunks", 0) if docs is not None else 0
    graph = _graph_snapshot(docs)
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if tasks:
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
    for task in tasks[:3]:
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
    lines = [
        f"Daily brief - {stamp}",
        f"Focus: {focus}",
        f"Tasks: {len(tasks)} open",
    ]
    lines.extend(f"- {task.get('id')}: {task.get('text', '')}"
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
        data = json.loads(p.read_text(encoding="utf-8"))
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
    parser.add_argument("--max-steps", type=int, default=3)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)
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
        if args.goal:
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
    elif args.goal:
        print(render_plan(result))
    else:
        print(render_briefing(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
