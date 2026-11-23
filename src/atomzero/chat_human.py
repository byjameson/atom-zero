"""A small virtual-human runtime for the from-scratch chat model.

The language model supplies wording; this module supplies the parts that a
21M model should not be asked to memorize: persona, facts, episodes, tasks.

    python research/chat_human.py --model checkpoints/spiking_chat_en_21m
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import time
from datetime import date, timedelta
from pathlib import Path

import torch

from atomzero.doc_memory import grounded_answer
from atomzero.transformer import TransformerLM, generate as transformer_generate

ROOT = Path.cwd()
DEFAULT_MEMORY = ROOT / "data/chat_human_memory.json"


def extract_facts(text: str) -> list[tuple[str, str]]:
    patterns = [
        ("name", r"\bmy name is ([A-Za-z][\w .'-]{0,40})"),
        ("likes", r"\bi (?:like|love) ([^.!?\n]{1,80})"),
        ("location", r"\bi live in ([^.!?\n]{1,80})"),
        ("work", r"\bi work (?:as|at) ([^.!?\n]{1,80})"),
        ("note", r"\bremember that ([^.!?\n]{1,120})"),
    ]
    out = []
    for key, pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            value = match.group(1).strip(" .,!?:;")
            if value:
                out.append((key, value))
    return out


def _tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1}


_QUESTION_STARTS = (
    "what", "who", "where", "when", "why", "how", "which", "whose", "whom",
    "is ", "are ", "was ", "were ", "do ", "does ", "did ", "can ", "could ",
    "should ", "would ", "will ", "tell me about",
)


def _parse_due(value: str) -> str:
    value = str(value or "").strip().lower()
    if value in ("today", "bugün"):
        return date.today().isoformat()
    if value in ("tomorrow", "yarın"):
        return (date.today() + timedelta(days=1)).isoformat()
    match = re.fullmatch(r"(?:in\s+)?(\d+)\s+days?", value)
    if match:
        return (date.today() + timedelta(days=int(match.group(1)))).isoformat()
    try:
        return date.fromisoformat(value).isoformat()
    except ValueError:
        return ""


TASK_PRIORITIES = ("low", "normal", "high")
TASK_TAG_LIMIT = 5


def _normalize_priority(value: str) -> str:
    priority = str(value or "normal").strip().lower()
    if priority not in TASK_PRIORITIES:
        raise ValueError(
            "priority must be one of: " + ", ".join(TASK_PRIORITIES))
    return priority


def _normalize_tags(values) -> list[str]:
    if values is None:
        return []
    if isinstance(values, str):
        values = values.split(",")
    tags = []
    for raw in values:
        tag = str(raw or "").strip().lower().lstrip("#")
        if not tag:
            continue
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,23}", tag):
            raise ValueError(
                "tags must use letters, numbers, - or _ (max 24 chars)")
        if tag not in tags:
            tags.append(tag)
    if len(tags) > TASK_TAG_LIMIT:
        raise ValueError(f"at most {TASK_TAG_LIMIT} tags are allowed")
    return tags


def _task_label(task: dict) -> str:
    due = str(task.get("due", "") or "").strip()
    priority = str(task.get("priority", "normal") or "normal").lower()
    priority_tag = f" [{priority}]" if priority != "normal" else ""
    try:
        tags = _normalize_tags(task.get("tags", []))
    except ValueError:
        tags = []
    tag_text = f" [tags: {', '.join(tags)}]" if tags else ""
    suffix = ""
    if due:
        overdue = due < date.today().isoformat()
        suffix = f" (due {due}{', overdue' if overdue else ''})"
    return f"{task['id']}{priority_tag} {task['text']}{tag_text}{suffix}"


def _parse_list_task_filters(text: str) -> tuple[str, str, str, str, str]:
    match = re.search(r"list\s+tasks\b", text, re.IGNORECASE)
    tail = text[match.end():] if match else ""
    tokens = tail.split()
    status = "open"
    due = "all"
    priority = "all"
    sort = "default"
    tag = "all"
    status_words = {"open", "done", "all"}
    due_words = {"overdue", "today", "week", "none"}
    priority_words = {"low", "normal", "high"}
    sort_words = {"default", "priority"}
    while tokens:
        last = tokens[-1].lower().strip("?!.")
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
        if (len(tokens) >= 2 and tokens[-2].lower() == "priority"
                and last in priority_words):
            priority = last
            tokens = tokens[:-2]
            continue
        if (len(tokens) >= 2 and tokens[-2].lower() in {"tag", "tags"}):
            tag = last.lstrip("#")
            tokens = tokens[:-2]
            continue
        if last in status_words:
            status = last
            tokens.pop()
            continue
        if last in due_words:
            due = last
            tokens.pop()
            continue
        if last in priority_words:
            priority = last
            tokens.pop()
            continue
        if last in sort_words:
            sort = last
            tokens.pop()
            continue
        if last.startswith("#") and len(last) > 1:
            tag = last[1:]
            tokens.pop()
            continue
        break
    return status, due, priority, sort, tag


def looks_like_question(text: str) -> bool:
    low = str(text or "").strip().lower()
    return low.endswith("?") or low.startswith(_QUESTION_STARTS)


class ChatHuman:
    def __init__(self, memory_path: str | Path = DEFAULT_MEMORY,
                 model=None, tokenizer=None, device=None, arch="transformer",
                 docs=None, on_gap=None, generator=None,
                 allow_general_chat=False):
        self.path = Path(memory_path)
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.arch = arch
        self.docs = docs
        self.on_gap = on_gap
        self.generator = generator
        self.allow_general_chat = bool(allow_general_chat)
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8-sig"))
        return {
            "persona": {"name": "Tiny", "style": "warm, concise, curious"},
            "facts": [],
            "episodes": [],
            "tasks": [],
            "next_task": 1,
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2, ensure_ascii=False)
                             + "\n", encoding="utf-8")

    def export_memory(self) -> dict:
        return copy.deepcopy(self.data)

    def import_memory(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise ValueError("memory data must be an object")
        persona = data.get("persona")
        if not isinstance(persona, dict):
            persona = self.data.get("persona", {})
        clean = {
            "persona": copy.deepcopy(persona),
            "facts": [],
            "episodes": [],
            "tasks": [],
            "next_task": 1,
        }
        for key, limit in (("facts", 1000), ("episodes", 1000),
                           ("tasks", 1000)):
            rows = data.get(key, [])
            if not isinstance(rows, list):
                raise ValueError(f"{key} must be a list")
            if len(rows) > limit:
                raise ValueError(f"{key} is too large")
            clean[key] = [copy.deepcopy(row) for row in rows
                          if isinstance(row, dict)]
        try:
            clean["next_task"] = max(1, int(data.get("next_task", 1)))
        except (TypeError, ValueError):
            clean["next_task"] = 1
        self.data = clean
        self.save()
        return {
            "facts": len(clean["facts"]),
            "episodes": len(clean["episodes"]),
            "tasks": len(clean["tasks"]),
        }

    def forget(self, text: str = "") -> dict:
        """Delete facts/episodes/tasks matching ``text``; empty clears all."""
        needle = str(text or "").strip().lower()
        before = {
            "facts": len(self.data.get("facts", [])),
            "episodes": len(self.data.get("episodes", [])),
            "tasks": len(self.data.get("tasks", [])),
        }

        def matches(value) -> bool:
            return bool(needle) and needle in str(value).lower()

        if needle:
            self.data["facts"] = [
                fact for fact in self.data.get("facts", [])
                if not matches(f"{fact.get('key', '')} {fact.get('value', '')}")
            ]
            self.data["episodes"] = [
                episode for episode in self.data.get("episodes", [])
                if not matches(episode.get("text", ""))
            ]
            self.data["tasks"] = [
                task for task in self.data.get("tasks", [])
                if not matches(task.get("text", ""))
            ]
        else:
            self.data["facts"] = []
            self.data["episodes"] = []
            self.data["tasks"] = []
            self.data["next_task"] = 1
        self.save()
        return {
            "text": needle,
            "facts_deleted": before["facts"] - len(self.data.get("facts", [])),
            "episodes_deleted": (before["episodes"]
                                 - len(self.data.get("episodes", []))),
            "tasks_deleted": before["tasks"] - len(self.data.get("tasks", [])),
            "cleared_all": not bool(needle),
        }

    def add_fact(self, key: str, value: str):
        value = value.strip()
        for fact in self.data["facts"]:
            if fact["key"] == key and fact["value"].lower() == value.lower():
                fact["ts"] = int(time.time())
                self.save()
                return
        self.data["facts"].append({"key": key, "value": value,
                                   "ts": int(time.time())})
        self.data["facts"] = self.data["facts"][-100:]
        self.save()

    def add_episode(self, role: str, text: str):
        self.data["episodes"].append({"role": role, "text": text,
                                      "ts": int(time.time())})
        self.data["episodes"] = self.data["episodes"][-200:]
        self.save()

    def add_task(self, text: str, due: str = "",
                 priority: str = "normal", tags=None) -> int:
        priority = _normalize_priority(priority)
        tags = _normalize_tags(tags)
        tid = int(self.data.get("next_task", 1))
        self.data["next_task"] = tid + 1
        self.data["tasks"].append({
            "id": tid, "text": text, "status": "open",
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "completed": "", "due": due, "priority": priority,
            "tags": tags,
        })
        self.save()
        return tid

    def build_task_list(self, text: str) -> dict:
        status, due, priority, sort, tag = _parse_list_task_filters(text)
        from atomzero.briefing import task_list_view

        return task_list_view(
            self.data, status=status, due=due, priority=priority,
            tag=tag, sort=sort)

    def set_task_priority(self, tid: int, priority: str) -> dict | None:
        normalized = _normalize_priority(priority)
        for task in self.data["tasks"]:
            if int(task["id"]) == int(tid):
                task["priority"] = normalized
                task["prioritized"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self.save()
                return task
        return None

    def set_task_tags(self, tid: int, tags) -> dict | None:
        normalized = _normalize_tags(tags)
        for task in self.data["tasks"]:
            if int(task["id"]) == int(tid):
                task["tags"] = normalized
                task["tagged"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self.save()
                return task
        return None

    def rename_task_tag(self, old, new) -> dict:
        from atomzero.briefing import rename_task_tag

        result = rename_task_tag(self.data, old, new)
        self.save()
        return result

    def reschedule_task(self, tid: int, due: str) -> dict | None:
        normalized = _parse_due(due)
        if not normalized:
            raise ValueError(
                "due date must be YYYY-MM-DD, today, tomorrow or in N days")
        for task in self.data["tasks"]:
            if int(task["id"]) == int(tid):
                task["due"] = normalized
                task["rescheduled"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self.save()
                return task
        return None

    def delete_task(self, tid: int) -> dict | None:
        for index, task in enumerate(self.data["tasks"]):
            if int(task["id"]) == int(tid):
                deleted = self.data["tasks"].pop(index)
                self.save()
                return deleted
        return None

    def complete_task(self, tid: int) -> bool:
        for task in self.data["tasks"]:
            if int(task["id"]) == int(tid):
                task["status"] = "done"
                task["completed"] = time.strftime("%Y-%m-%dT%H:%M:%S")
                self.save()
                return True
        return False

    def retrieve(self, query: str, k: int = 5) -> list[str]:
        query_tokens = _tokens(query)
        scored = []
        for fact in self.data["facts"]:
            text = f"{fact['key']}: {fact['value']}"
            scored.append((len(query_tokens & _tokens(text)), text))
        for task in self.data["tasks"]:
            if task["status"] == "open":
                text = f"task {task['id']}: {task['text']}"
                scored.append((len(query_tokens & _tokens(text)), text))
        for episode in reversed(self.data["episodes"][-20:]):
            text = f"{episode['role']}: {episode['text']}"
            scored.append((len(query_tokens & _tokens(text)), text))
        scored = [item for item in scored if item[0] > 0]
        scored.sort(key=lambda item: -item[0])
        out = []
        for _, text in scored:
            if text not in out:
                out.append(text)
            if len(out) >= k:
                break
        return out

    def persona_line(self) -> str:
        persona = self.data["persona"]
        return f"{persona.get('name', 'Tiny')} ({persona.get('style', '')})"

    def memory_context(self) -> str:
        lines = [f"Persona: {self.persona_line()}"]
        facts = self.data["facts"][-6:]
        if facts:
            lines.append("Known facts: " + "; ".join(
                f"{fact['key']} is {fact['value']}" for fact in facts))
        open_tasks = [task for task in self.data["tasks"]
                      if task["status"] == "open"]
        if open_tasks:
            lines.append("Open tasks: " + "; ".join(
                _task_label(task) for task in open_tasks))
        return "\n".join(lines)

    def format_skill(self, text: str) -> str | None:
        """Deterministic handlers for exact formats the 20M model is weak at."""
        low = text.lower().strip()
        numbers = ["one", "two", "three", "four", "five", "six", "seven",
                   "eight", "nine", "ten", "eleven", "twelve"]
        match = re.search(
            r"count (?:from one )?to (\d+|one|two|three|four|five|six|"
            r"seven|eight|nine|ten|eleven|twelve)", low)
        if match:
            value = match.group(1)
            n = int(value) if value.isdigit() else numbers.index(value) + 1
            if 1 <= n <= len(numbers):
                return ", ".join(numbers[:n]).capitalize() + "."
        if low.startswith("be very brief:"):
            rest = low.split(":", 1)[1]
            if any(word in rest for word in ("hello", "hi", "hey")):
                return "Hi."
            if "thank" in rest:
                return "Thanks."
            if "bye" in rest or "goodbye" in rest:
                return "Bye."
            return "Okay."
        if low.startswith("make it polite:"):
            action = text.split(":", 1)[1].strip(" .!?")
            return f"Could you {action}, please?"
        if "give me two colors" in low or "name two colors" in low:
            return "Blue and green."
        if "give me three fruits" in low:
            return "Apple, banana, and orange."
        if low.startswith(("repeat after me:", "say exactly:", "echo:")):
            return text.split(":", 1)[1].strip()
        if low.startswith("summarize in one sentence:"):
            return text.split(":", 1)[1].strip()
        return None

    def _generate(self, user_text: str) -> str:
        if self.generator is not None:
            try:
                answer = self.generator(user_text)
                if answer:
                    return str(answer)
            except Exception:
                pass
        if self.model is None or self.tokenizer is None:
            return "I do not have a language model loaded yet."
        # The 20M model was trained on plain chat turns; injecting a long
        # persona/fact block into the user turn is out of distribution.
        prompt = user_text
        if self.arch == "transformer":
            user_id = self.tokenizer.token_to_id("<|user|>")
            assistant_id = self.tokenizer.token_to_id("<|assistant|>")
            text, _ = transformer_generate(
                self.model, self.tokenizer,
                f"<|user|> {prompt} <|assistant|>",
                max_new=64, temperature=0.7, top_k=20,
                repetition_penalty=1.1,
                banned_ids=(user_id, assistant_id),
                device=self.device)
        else:
            try:
                from atomzero.chat_eval import chat_generate
            except ImportError as exc:
                raise RuntimeError(
                    "spiking chat is not part of the atomzero package; "
                    "use the transformer checkpoint") from exc
            text, _ = chat_generate(
                self.model, self.tokenizer, prompt, max_new_tokens=64,
                temperature=0.7, top_k=20, repetition_penalty=1.1,
                device=self.device)
        return text or "I am not sure what to say yet."

    def respond(self, text: str) -> str:
        self.last_sources = []
        self.last_evidence = []
        self.last_confidence = {"score": 0.0, "label": "none",
                                "coverage": 0.0, "margin": 0.0}
        self.add_episode("user", text)
        extracted = extract_facts(text)
        for key, value in extracted:
            self.add_fact(key, value)

        low = text.lower().strip()
        if "what do you remember" in low or "what do you know about me" in low:
            facts = self.data["facts"]
            reply = ("I remember: " + "; ".join(
                f"{fact['key']} is {fact['value']}" for fact in facts[-8:])
                if facts else "I do not know anything about you yet.")
        elif "what is my name" in low or "who am i" in low:
            name = next((f["value"] for f in self.data["facts"]
                         if f["key"] == "name"), None)
            reply = (f"Your name is {name}." if name
                     else "I do not know your name yet.")
        elif "what do i like" in low or "what do you know i like" in low:
            likes = [f["value"] for f in self.data["facts"]
                     if f["key"] == "likes"]
            reply = ("You like " + ", ".join(likes) + "." if likes
                     else "I do not know what you like yet.")
        elif low.startswith("add task"):
            task = text.split("add task", 1)[1].strip(" :")
            due = ""
            priority = "normal"
            tags = []
            due_error = ""
            priority_error = ""
            tags_error = ""
            priority_seen = False
            due_seen = False
            tags_seen = False
            while task:
                if not tags_seen:
                    tags_match = re.search(
                        r"\s+tags?\s+((?:[a-z0-9][a-z0-9_-]{0,23}"
                        r"\s*,\s*)*[a-z0-9][a-z0-9_-]{0,23})\s*$",
                        task, re.IGNORECASE)
                    if tags_match:
                        try:
                            tags = _normalize_tags(tags_match.group(1))
                        except ValueError as exc:
                            tags_error = str(exc)
                            break
                        task = task[:tags_match.start()].strip(" :")
                        tags_seen = True
                        continue
                    invalid_tags = re.search(
                        r"\s+tags?\s+(.+)$", task, re.IGNORECASE)
                    if (invalid_tags and not re.search(
                            r"\s+(?:due|priority)\s+",
                            invalid_tags.group(1), re.IGNORECASE)):
                        try:
                            _normalize_tags(invalid_tags.group(1))
                        except ValueError as exc:
                            tags_error = str(exc)
                            break
                if not priority_seen:
                    priority_match = re.search(
                        r"\s+priority\s+(\S+)\s*$", task, re.IGNORECASE)
                    if priority_match:
                        raw_priority = (
                            priority_match.group(1).strip().lower())
                        if raw_priority not in TASK_PRIORITIES:
                            priority_error = (
                                f"I could not read the priority "
                                f"'{raw_priority}'. Use low, normal or high.")
                            break
                        priority = raw_priority
                        task = task[:priority_match.start()].strip(" :")
                        priority_seen = True
                        continue
                if not due_seen:
                    due_match = re.search(
                        r"\s+due\s+(.+)$", task, re.IGNORECASE)
                    if due_match:
                        due_value = due_match.group(1).strip()
                        parsed_due = _parse_due(due_value)
                        if not parsed_due:
                            due_error = (
                                f"I could not read the due date "
                                f"'{due_value}'. Use YYYY-MM-DD, today, "
                                "tomorrow or in N days.")
                            break
                        due = parsed_due
                        task = task[:due_match.start()].strip(" :")
                        due_seen = True
                        continue
                break
            if tags_error:
                reply = tags_error
            elif priority_error:
                reply = priority_error
            elif due_error:
                reply = due_error
            elif task:
                tid = self.add_task(
                    task, due=due, priority=priority, tags=tags)
                suffix = f" (due {due})" if due else ""
                if priority != "normal":
                    suffix += f" (priority {priority})"
                if tags:
                    suffix += f" (tags {', '.join(tags)})"
                reply = f"Added task {tid}: {task}{suffix}"
            else:
                reply = "Tell me the task after 'add task'."
        elif low.startswith("reschedule task"):
            match = re.match(
                r"reschedule task\s*(\d+)\s+(?:due|to)\s+(.+)$",
                text, re.IGNORECASE)
            if not match:
                reply = ("Use 'reschedule task ID due YYYY-MM-DD' "
                         "or 'reschedule task ID due tomorrow'.")
            else:
                try:
                    task = self.reschedule_task(
                        int(match.group(1)), match.group(2).strip())
                except ValueError as exc:
                    task = None
                    reply = str(exc)
                else:
                    if task is None:
                        reply = "I could not find that task."
                    else:
                        reply = (f"Rescheduled task {task['id']} to "
                                 f"{task['due']}.")
        elif low.startswith("tag task"):
            match = re.match(
                r"tag\s+task\s*(\d+)\s+(.+)$", text, re.IGNORECASE)
            if not match:
                reply = "Use 'tag task ID tag1,tag2' to set task tags."
            else:
                value = match.group(2).strip()
                tags = [] if value.lower() in ("clear", "none", "-") else value
                try:
                    task = self.set_task_tags(int(match.group(1)), tags)
                except ValueError as exc:
                    task = None
                    reply = str(exc)
                else:
                    if task is None:
                        reply = "I could not find that task."
                    elif not task["tags"]:
                        reply = f"Cleared tags on task {task['id']}."
                    else:
                        reply = (f"Set tags on task {task['id']}: "
                                 f"{', '.join(task['tags'])}.")
        elif low.startswith("set task"):
            match = re.match(
                r"set task\s*(\d+)\s+(?:priority\s+)?"
                r"(low|normal|high)\??$", text, re.IGNORECASE)
            if not match:
                reply = "Use 'set task ID priority low|normal|high'."
            else:
                try:
                    task = self.set_task_priority(
                        int(match.group(1)), match.group(2))
                except ValueError as exc:
                    task = None
                    reply = str(exc)
                else:
                    if task is None:
                        reply = "I could not find that task."
                    else:
                        reply = (f"Set task {task['id']} priority to "
                                 f"{task['priority']}.")
        elif "list tasks" in low:
            reply = self.build_task_list(text)["text"]
        elif "complete task" in low:
            match = re.search(r"complete task\s*(\d+)", low)
            if match and self.complete_task(int(match.group(1))):
                reply = f"Completed task {match.group(1)}."
            else:
                reply = "I could not find that task."
        elif "how are you" in low:
            reply = "I am doing well, thank you. How are you?"
        elif low.strip(" ?!.") in ("hello", "hi", "hey"):
            reply = "Hello! How can I help you today?"
        elif "thank you" in low:
            reply = "You are welcome."
        elif low.startswith("teach:"):
            fact = text.split(":", 1)[1].strip()
            if fact:
                self.add_fact("note", fact)
                reply = "I learned that."
            else:
                reply = "Tell me what to learn after 'teach:'."
        elif extracted:
            reply = "I will remember that."
        else:
            hits = self.docs.search(text, k=6) if self.docs else []
            if hits:
                answer = grounded_answer(text, hits)
                self.last_sources = answer["sources"]
                self.last_evidence = answer["evidence"]
                self.last_confidence = answer["confidence"]
                reply = answer["text"]
            else:
                formatted = self.format_skill(text)
                if formatted is not None:
                    reply = formatted
                else:
                    question = looks_like_question(text)
                    if question and self.on_gap is not None:
                        self.on_gap(text)
                    general = self.allow_general_chat and self.generator is not None
                    if question and self.docs is not None and not general:
                        reply = ("I do not have a source for that yet. I can "
                                 "add it to the research queue; approve the "
                                 "topic and I will fetch it.")
                    else:
                        reply = self._generate(text)

        self.add_episode("assistant", reply)
        return reply


def load_model(path: str, device, arch: str = "spiking",
               tokenizer_path: str = "data/chat_en_bpe8k.json",
               adapter: str = ""):
    if arch == "transformer":
        ck = torch.load(Path(path) / "model.pt", map_location="cpu",
                        weights_only=False)
        cfg = ck["config"]
        model = TransformerLM(cfg["vocab"], cfg["dim"], cfg["layers"],
                              cfg["heads"], cfg["ff"]).to(device)
        model.load_state_dict(ck["state"])
        if adapter:
            from atomzero.lora import apply_adapter
            apply_adapter(model, ROOT / adapter)
        model.eval()
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(ROOT / tokenizer_path))
        return model, tokenizer
    from atomzero.spiking import load_checkpoint
    model, tokenizer, _ = load_checkpoint(path)
    model.to(device)
    return model, tokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", default=str(DEFAULT_MEMORY))
    ap.add_argument("--model", default="")
    ap.add_argument("--arch", choices=["spiking", "transformer"],
                    default="spiking")
    ap.add_argument("--tokenizer", default="data/chat_en_bpe8k.json")
    ap.add_argument("--docs", default="")
    ap.add_argument("--adapter", default="")
    ap.add_argument("--message", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = tokenizer = None
    if args.model:
        model, tokenizer = load_model(args.model, device, args.arch,
                                      args.tokenizer, args.adapter)
    docs = None
    if args.docs:
        from atomzero.doc_memory import DocMemory
        docs = DocMemory(args.docs)
    human = ChatHuman(args.memory, model, tokenizer, device, args.arch, docs)

    if args.message:
        print(human.respond(args.message))
        return
    print("Chat human ready. Ctrl-C to exit.")
    while True:
        try:
            text = input("you> ")
        except (EOFError, KeyboardInterrupt):
            break
        if text.strip():
            print("tiny>", human.respond(text))


if __name__ == "__main__":
    main()
