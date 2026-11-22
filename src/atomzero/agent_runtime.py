"""Session-based agent runtime around the tiny model and local tools."""
from __future__ import annotations

import ast
import operator
import re
from datetime import datetime
from pathlib import Path

from atomzero.briefing import plan_goal
from atomzero.doc_memory import (
    best_snippet, format_citation, grounded_answer)
from atomzero.knowledge_gaps import KnowledgeGapError
from atomzero.knowledge_graph import entity_sources
from atomzero.skills import render_skill
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
                 generator=None, allow_general_chat=False, skills=None):
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", session_id):
            raise ValueError("invalid session id")
        self.id = session_id
        self.docs = docs
        self.wiki_lang = wiki_lang
        self.gap_store = gap_store
        self.safety = safety
        self.skills = skills
        self.pending_plan = None
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
        hits = []
        if skill.get("use_context", True) and self.docs is not None:
            hits = self.docs.search(
                text, k=int(skill.get("max_chunks", 4)), diverse=True)
        prompt = render_skill(skill, text, hits)
        system = str(skill.get("system") or "").strip()
        if system:
            prompt = f"{system}\n\n{prompt}"
        sources = []
        evidence = []
        for hit in hits:
            source = {
                "source": hit.get("source", ""),
                "title": hit.get("title", ""),
                "kind": hit.get("kind", "document"),
                "verified": bool(hit.get("verified", True)),
            }
            if source not in sources:
                sources.append(source)
            evidence.append({
                "source": hit.get("source", ""),
                "title": hit.get("title", ""),
                "page": hit.get("page"),
                "section": hit.get("section", ""),
                "snippet": best_snippet(text, hit.get("chunk", "")),
            })
        base = {
            "tool_calls": [{"name": "skill", "skill": skill["name"]}],
            "sources": sources,
            "evidence": evidence,
            "skill": skill["name"],
        }
        if not skill.get("provider", True):
            return {"reply": prompt, **base}
        generator = getattr(self.human, "generator", None)
        if generator is None:
            return {
                "reply": (f"Skill '{skill['name']}' needs a local provider. "
                          "Start Atom Zero with --provider or configure an "
                          "OpenAI-compatible endpoint."),
                **base,
            }
        try:
            answer = generator(prompt)
        except Exception as exc:
            return {"reply": f"Skill provider failed: {exc}", **base}
        if not answer:
            return {"reply": f"Skill '{skill['name']}' returned no answer.",
                    **base}
        return {"reply": str(answer), **base}

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
                          "search your local documents, look up Wikipedia "
                          "with sources, calculate simple arithmetic, and "
                          "tell the time. I am a small local model, so I am "
                          "not a general reasoning or coding assistant."),
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
        graph_query = re.match(
            r"^(?:what is related to|what relates to|connections (?:for|of)|"
            r"graph)\s*:?\s*(.+?)\??$", text, re.IGNORECASE)
        if graph_query:
            return self._graph_entity(graph_query.group(1).strip())
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
