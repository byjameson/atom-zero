"""Build deterministic benign chat replay rows for mixed fine-tuning.

    python research/build_benign_chat.py \
        --out data/benign_chat_v1/train.jsonl \
        --manifest data/benign_chat_v1/manifest.json \
        --per-category 18 \
        --seed 41

The 1.1.0 safety-replay ratio sweep showed that even a 1x safety row mix
produced 68% refusals on benign open prompts. The copy and factual content
rows teach grounded or copied replies, but the mix has almost no ordinary
conversation. The model therefore learns "assistant style = refusal" for any
prompt that is not quoted or source-grounded.

This builder adds the missing positive examples: eight everyday categories of
safe prompts with short, helpful, non-personal answers. The answers contain a
concrete response rather than a refusal. The rows are rehearsal data, not a
factuality claim and not evidence of general capability.

Output rows contain exactly::

    {"user": "...", "assistant": "...", "family": "identity",
     "source": "synthetic_template", "style": "benign_chat"}

The manifest is content-free: category counts, seed, total and a SHA-256 of
the exact JSONL bytes. Prompt and answer text is never copied into it.

This builder does not train, download or load a model. It only writes JSONL
and a manifest. ``data/`` is gitignored, so the generator is the committed
source of truth. See ``docs/BENIGN_CHAT.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "benign_chat_v1"
SOURCE = "synthetic_template"
STYLE = "benign_chat"
DEFAULT_OUT = "data/benign_chat_v1/train.jsonl"
DEFAULT_MANIFEST = "data/benign_chat_v1/manifest.json"
DEFAULT_PER_CATEGORY = 18
DEFAULT_SEED = 41
CATEGORY_ORDER = (
    "identity",
    "greeting",
    "explanation",
    "cooking",
    "travel",
    "coding",
    "planning",
    "general_knowledge",
)
REQUIRED_ROW_FIELDS = ("user", "assistant", "family", "source", "style")
MIN_ANSWER_CHARS = 24
MAX_ANSWER_CHARS = 300
MAX_USER_CHARS = 200

# A benign answer must not read like a refusal. These markers are deliberately
# broad: the point of this dataset is to teach the opposite behaviour.
REFUSAL_MARKERS = (
    "i can't",
    "i cannot",
    "i won't",
    "i'm sorry",
    "i am sorry",
    "unable to",
    "not able to",
    "i must decline",
    "i'd rather not",
    "i would rather not",
    "as an ai language model",
)
# Narrow secret/credential fragments that must never appear in a rehearsal
# reply. The dataset has no reason to mention them.
UNSAFE_TERMS = (
    "password=",
    "api key",
    "secret key",
    "private key",
    "bearer ",
    "sk-",
    "ghp_",
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)")
_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]*(?=(?:\s|$))")

__all__ = [
    "CATEGORY_ORDER",
    "DEFAULT_MANIFEST",
    "DEFAULT_OUT",
    "DEFAULT_PER_CATEGORY",
    "DEFAULT_SEED",
    "MAX_ANSWER_CHARS",
    "MIN_ANSWER_CHARS",
    "REFUSAL_MARKERS",
    "REQUIRED_ROW_FIELDS",
    "SCHEMA_VERSION",
    "SOURCE",
    "STYLE",
    "build_manifest",
    "build_rows",
    "main",
    "normalize_text",
    "render_jsonl",
    "sentence_count",
    "validate_answer",
    "validate_row",
    "write_dataset",
]


def _group(prompts: Sequence[str], answers: Sequence[str],
           concrete: Sequence[str]) -> dict[str, Any]:
    return {
        "prompts": tuple(prompts),
        "answers": tuple(answers),
        "concrete": tuple(concrete),
    }


# Every group is a small set of paraphrased prompts plus several distinct,
# short answers. A group's ``concrete`` terms must appear in each answer so a
# reply actually answers the question instead of only sounding polite.
TEMPLATES: dict[str, tuple[dict[str, Any], ...]] = {
    "identity": (
        _group(
            (
                "Who are you?",
                "Can you introduce yourself?",
                "What kind of assistant are you?",
            ),
            (
                "I'm Atom Zero, a small local AI assistant. I can answer "
                "questions and help with everyday tasks like writing, "
                "planning, and basic coding.",
                "I am Atom Zero, a local assistant built to give short, "
                "useful answers. Ask me about explanations, planning, "
                "cooking, travel, or code.",
                "I'm Atom Zero, an AI assistant that runs locally. I help "
                "with questions, drafting, and step-by-step planning.",
            ),
            ("atom zero",),
        ),
        _group(
            (
                "What can you do?",
                "What can you help me with?",
                "Which tasks can you assist with?",
            ),
            (
                "I can explain ideas, help with code, plan a trip or study "
                "session, and answer general questions. Tell me what you are "
                "working on.",
                "I can help with everyday questions, short explanations, "
                "writing drafts, cooking tips, travel prep, and debugging "
                "simple code.",
                "I can answer factual questions, walk through a plan, and "
                "give practical suggestions for cooking, travel, and coding. "
                "What would you like to start with?",
            ),
            ("help", "answer"),
        ),
        _group(
            (
                "What are you good at?",
                "How can I use you effectively?",
                "What should I ask you?",
            ),
            (
                "I'm best at short, practical answers: definitions, "
                "checklists, simple how-tos, and first drafts. Give me a "
                "clear goal and any limits.",
                "Ask me for concise explanations, plans, and everyday advice. "
                "I do better with a specific question and a little context.",
                "I work well for study help, writing drafts, and simple "
                "technical questions. Tell me the goal and the format you "
                "want.",
            ),
            ("best", "ask me", "work well"),
        ),
    ),
    "greeting": (
        _group(
            (
                "Hello!",
                "Hi there!",
                "Hey, good to see you.",
            ),
            (
                "Hello! It is good to meet you. What can I help you with "
                "today?",
                "Hi! I'm ready to help, so tell me what you are working on.",
                "Hey! Ask me anything and I will help with a short, useful "
                "answer.",
            ),
            ("hello", "hi", "hey"),
        ),
        _group(
            (
                "How are you?",
                "How is your day going?",
                "How are you doing today?",
            ),
            (
                "I'm doing well and ready to help. What would you like to work "
                "on?",
                "All good here, thanks for asking. What can I do for you "
                "today?",
                "I'm running smoothly and glad to chat. How can I help?",
            ),
            ("doing well", "all good", "running smoothly"),
        ),
        _group(
            (
                "Good morning!",
                "Good morning, how are you?",
                "Morning! I hope you slept well.",
            ),
            (
                "Good morning! I hope your day is off to a good start. What "
                "can I help with?",
                "Good morning! Ready when you are; what are we working on "
                "today?",
                "Morning! Let's make the day easier. What would you like to "
                "start with?",
            ),
            ("morning",),
        ),
    ),
    "explanation": (
        _group(
            (
                "What is a transformer?",
                "Can you explain transformer models simply?",
                "How does a transformer neural network work?",
            ),
            (
                "A transformer is a neural network architecture built around "
                "attention. It lets a model weigh the parts of an input that "
                "matter most and process tokens in parallel.",
                "A transformer is a model design that uses self-attention to "
                "relate words or tokens to each other. That makes it effective "
                "for language tasks.",
                "In simple terms, a transformer learns which parts of the "
                "input to focus on through attention layers. Most modern "
                "language models use this architecture.",
            ),
            ("attention",),
        ),
        _group(
            (
                "What is RAG?",
                "What does retrieval-augmented generation mean?",
                "How does retrieval-augmented generation work?",
            ),
            (
                "RAG stands for retrieval-augmented generation. It retrieves "
                "relevant documents first, then gives them to the model as "
                "context for the answer.",
                "Retrieval-augmented generation combines a search step with "
                "text generation. The model answers using passages fetched "
                "from a document store.",
                "RAG means the system looks up relevant sources before "
                "answering. That grounds the reply in retrieved text instead "
                "of relying only on model memory.",
            ),
            ("retriev",),
        ),
        _group(
            (
                "What is a database?",
                "Can you explain what a database is?",
                "How would you describe a database to a beginner?",
            ),
            (
                "A database is an organized collection of data that can be "
                "stored, searched, and updated. A database management system "
                "handles those operations.",
                "A database stores structured information so it can be queried "
                "reliably. Examples include tables of users, orders, or "
                "inventory.",
                "Think of a database as a structured store for records. "
                "Software such as a DBMS lets applications read, write, and "
                "update those records safely.",
            ),
            ("database",),
        ),
    ),
    "cooking": (
        _group(
            (
                "How do I boil an egg?",
                "What is the best way to make a hard-boiled egg?",
                "How long should I cook an egg in boiling water?",
            ),
            (
                "Lower eggs into boiling water and cook for about 9 to 10 "
                "minutes for a firm yolk. Move them to cold water afterward so "
                "they are easy to peel.",
                "For a hard-boiled egg, simmer for roughly 10 minutes, then "
                "cool it in ice water. For a softer yolk, take it out at 6 to "
                "7 minutes.",
                "Place the eggs in a pot of boiling water, cook 9 to 10 "
                "minutes, and transfer them to an ice bath. That gives a firm "
                "yolk and clean peeling.",
            ),
            ("boil",),
        ),
        _group(
            (
                "How do I make simple pasta?",
                "What is an easy pasta recipe for a beginner?",
                "Can you give me a quick pasta method?",
            ),
            (
                "Boil salted water, add the pasta, and cook it according to the "
                "package time. Drain it, then toss with olive oil, garlic, and "
                "grated cheese.",
                "Cook pasta in well-salted boiling water until just tender. "
                "Save a little cooking water, mix it with your sauce, and "
                "combine everything in the pot.",
                "For a quick meal, boil pasta, reserve a cup of the water, and "
                "drain. Stir the water into a simple tomato or butter sauce so "
                "it coats the pasta.",
            ),
            ("pasta",),
        ),
        _group(
            (
                "How should I store bread?",
                "What is the best way to keep bread fresh?",
                "How do I store a loaf of bread so it lasts longer?",
            ),
            (
                "Keep bread at room temperature in a paper bag or bread box "
                "for two to three days. For longer storage, slice it and freeze "
                "it.",
                "A paper bag keeps the crust from getting soggy, and it works "
                "well for a few days. Freeze extra slices and toast them when "
                "needed.",
                "Store everyday bread in a cool, dry place and avoid the "
                "refrigerator. Freeze what you will not finish within a few "
                "days.",
            ),
            ("bread", "freeze"),
        ),
    ),
    "travel": (
        _group(
            (
                "How do I pack a carry-on bag?",
                "What should I put in a carry-on for a short trip?",
                "Can you help me pack light for a flight?",
            ),
            (
                "Start with a small capsule wardrobe, then add chargers, "
                "medications, and a clear toiletries bag. Check your airline's "
                "size and liquid limits before packing.",
                "Roll clothes to save space in a carry-on, wear the bulkiest "
                "shoes, and keep documents and medication in one easy-to-reach "
                "pouch. Leave room for anything you buy.",
                "Pack two or three mix-and-match outfits, a charger, and only "
                "the toiletries you need. Confirm the carry-on dimensions and "
                "security rules for your airline.",
            ),
            ("pack", "carry-on"),
        ),
        _group(
            (
                "How should I plan a day trip?",
                "What should I bring on a day trip?",
                "Can you help me plan a one-day outing?",
            ),
            (
                "Choose a destination within a comfortable travel time and "
                "check the weather first. Bring water, snacks, a charged "
                "phone, and a plan for the return trip.",
                "Pick two or three main stops, then leave space between them. "
                "Carry water, a light layer, and a map or offline directions.",
                "Plan your route, book anything that needs a reservation, and "
                "set a return time. Pack water, snacks, and a portable "
                "charger.",
            ),
            ("water", "trip"),
        ),
        _group(
            (
                "How early should I arrive at the airport?",
                "When should I get to the airport for a flight?",
                "What is a safe airport arrival time?",
            ),
            (
                "Arrive about two hours before a domestic flight and three "
                "hours before an international one. Add extra time for checked "
                "bags or busy holidays.",
                "A good rule is two hours for domestic departures and three "
                "for international flights. Check your airline's guidance "
                "because security lines vary.",
                "Plan to reach the terminal two to three hours early, "
                "depending on the route. That leaves time for check-in, "
                "security, and finding the gate.",
            ),
            ("hour",),
        ),
    ),
    "coding": (
        _group(
            (
                "What is a function in programming?",
                "Can you explain functions in code?",
                "How do functions work in Python?",
            ),
            (
                "A function is a named block of code that takes inputs, "
                "performs a task, and usually returns a result. You call it by "
                "name whenever you need that task.",
                "A function groups reusable logic behind a name. Parameters "
                "pass data in, and a return statement sends a result back.",
                "In Python, a function is defined with def and can accept "
                "arguments. It keeps code organized and avoids repeating the "
                "same steps.",
            ),
            ("function",),
        ),
        _group(
            (
                "How do I debug a typo in my code?",
                "My program has a spelling mistake; how do I find it?",
                "What is the fastest way to fix a typo in code?",
            ),
            (
                "Read the error message and go to the line it names. Check "
                "variable and function names character by character, then run "
                "the smallest test that reproduces it.",
                "Search the file for the name in the error and compare it with "
                "where the variable was defined. An editor's rename or find "
                "feature makes mismatches easier to spot.",
                "Start at the reported line, check capitalization and "
                "spelling, and print the value before it fails. A quick rerun "
                "confirms the fix.",
            ),
            ("error", "spelling", "typo"),
        ),
        _group(
            (
                "Can you explain loops in programming?",
                "What is a loop in code?",
                "How does a for loop work?",
            ),
            (
                "A loop repeats a block of code until a condition is met or a "
                "collection is finished. A for loop visits each item, while a "
                "while loop checks a condition before each pass.",
                "Loops let you avoid writing the same instruction many times. "
                "For example, a for loop can process every item in a list.",
                "A loop runs the same steps repeatedly. Use a for loop for a "
                "known sequence and a while loop when the stopping condition "
                "is dynamic.",
            ),
            ("loop",),
        ),
    ),
    "planning": (
        _group(
            (
                "How should I plan a study week?",
                "Can you help me make a weekly study plan?",
                "What is a good study schedule for one week?",
            ),
            (
                "List the study topics you need to cover and split them into "
                "small daily blocks. Add short reviews, a buffer day, and "
                "enough sleep.",
                "Start with your deadlines, then assign two focused study "
                "blocks per day. End the week with a practice test or a summary "
                "review.",
                "Choose three main study goals for the week and give each one "
                "a specific time slot. Include breaks and one catch-up "
                "session.",
            ),
            ("study",),
        ),
        _group(
            (
                "How do I prioritize my tasks?",
                "What is a simple way to decide what to do first?",
                "How can I sort tasks by priority?",
            ),
            (
                "Write down every task, mark which are urgent and which are "
                "important, and do the important ones first. Schedule or drop "
                "the rest.",
                "Use a short list and rank items by impact and deadline. Start "
                "with the task that unblocks the most other work.",
                "Group tasks into do now, schedule, delegate, and drop. Then "
                "focus on one item at a time.",
            ),
            ("task", "item"),
        ),
        _group(
            (
                "How do I make a good checklist?",
                "What should a useful checklist include?",
                "Can you help me build a task checklist?",
            ),
            (
                "Start a checklist with the goal, list the steps in the order "
                "they happen, and add an owner or deadline for each. Keep it "
                "short enough to use.",
                "A good checklist has clear, actionable items and a way to mark "
                "them done. Put the most important step first.",
                "Write each item as a single action, group related steps, and "
                "include a final check. Review the list before you start.",
            ),
            ("goal", "step", "item"),
        ),
    ),
    "general_knowledge": (
        _group(
            (
                "What is the capital of France?",
                "Which city is the capital of France?",
                "Can you tell me France's capital?",
            ),
            (
                "The capital of France is Paris, the country's largest city and "
                "main cultural center.",
                "Paris is the capital of France and its largest city by "
                "population.",
                "France's capital city is Paris, located in the north-central "
                "part of the country.",
            ),
            ("paris",),
        ),
        _group(
            (
                "What is the boiling point of water?",
                "At what temperature does water boil at sea level?",
                "How hot does water need to be to boil?",
            ),
            (
                "At sea level, water boils at 100 degrees Celsius, which is 212 "
                "degrees Fahrenheit.",
                "Water boils at about 100 degrees Celsius at standard "
                "atmospheric pressure. At higher altitudes, the boiling point "
                "is lower.",
                "The boiling point of water is 100 degrees Celsius or 212 "
                "degrees Fahrenheit at sea level.",
            ),
            ("100",),
        ),
        _group(
            (
                "What is the speed of light?",
                "How fast does light travel in a vacuum?",
                "Can you tell me the speed of light?",
            ),
            (
                "Light travels at about 299,792 kilometers per second in a "
                "vacuum, or roughly 186,282 miles per second.",
                "The speed of light in a vacuum is approximately 300,000 "
                "kilometers per second.",
                "In a vacuum, light moves at 299,792,458 meters per second, "
                "which is about 186,000 miles per second.",
            ),
            ("299,792", "300,000", "186,000"),
        ),
    ),
}


def normalize_text(value: Any) -> str:
    """Case-fold and collapse whitespace for narrow lexical comparison."""

    return " ".join(
        str(value or "").replace("\u2019", "'").split()).casefold()


def sentence_count(value: str) -> int:
    """Count sentence-ending punctuation followed by whitespace or the end."""

    return len(_SENTENCE_END_RE.findall(value.strip()))


def _contains_personal_data(value: str) -> bool:
    return bool(
        _EMAIL_RE.search(value)
        or _URL_RE.search(value)
        or _PHONE_RE.search(value))


def validate_user(value: Any) -> bool:
    """Return True iff ``value`` is a short, single-line prompt string."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in value or "\r" in value:
        return False
    if len(text) > MAX_USER_CHARS:
        return False
    return True


def validate_answer(value: Any) -> bool:
    """Return True iff ``value`` is a short, safe, non-refusal answer."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in value or "\r" in value:
        return False
    if not MIN_ANSWER_CHARS <= len(text) <= MAX_ANSWER_CHARS:
        return False
    if not 1 <= sentence_count(text) <= 3:
        return False
    reply = normalize_text(text)
    if any(marker in reply for marker in REFUSAL_MARKERS):
        return False
    if any(term in reply for term in UNSAFE_TERMS):
        return False
    if _contains_personal_data(text):
        return False
    return True


def validate_row(row: Mapping[str, Any]) -> bool:
    """Return True iff ``row`` has the benign chat schema and invariants."""

    if set(row) != set(REQUIRED_ROW_FIELDS):
        return False
    if row.get("family") not in CATEGORY_ORDER:
        return False
    if row.get("source") != SOURCE:
        return False
    if row.get("style") != STYLE:
        return False
    if not validate_user(row.get("user")):
        return False
    if not validate_answer(row.get("assistant")):
        return False
    return True


def _validate_group(category: str, group_index: int,
                    group: Mapping[str, Any]) -> None:
    prompts = group.get("prompts")
    answers = group.get("answers")
    concrete = group.get("concrete")
    if (not isinstance(prompts, tuple) or len(prompts) < 2
            or any(not validate_user(prompt) for prompt in prompts)):
        raise ValueError(
            f"{category} group {group_index} has invalid prompt templates")
    if len(set(prompts)) != len(prompts):
        raise ValueError(
            f"{category} group {group_index} repeats a prompt template")
    if (not isinstance(answers, tuple) or len(answers) < 2
            or any(not validate_answer(answer) for answer in answers)):
        raise ValueError(
            f"{category} group {group_index} has invalid answer templates")
    if len(set(answers)) != len(answers):
        raise ValueError(
            f"{category} group {group_index} repeats an answer template")
    if (not isinstance(concrete, tuple) or not concrete
            or any(not isinstance(term, str) or not term.strip()
                   for term in concrete)):
        raise ValueError(
            f"{category} group {group_index} has invalid concrete terms")
    for answer in answers:
        reply = normalize_text(answer)
        if not any(normalize_text(term) in reply for term in concrete):
            raise ValueError(
                f"{category} group {group_index} answer lacks a concrete "
                f"term: {answer!r}")


def _combinations(category: str) -> list[tuple[str, str]]:
    if category not in TEMPLATES:
        raise ValueError(f"missing template category {category!r}")
    combinations: list[tuple[str, str]] = []
    for group_index, group in enumerate(TEMPLATES[category]):
        _validate_group(category, group_index, group)
        for prompt in group["prompts"]:
            for answer in group["answers"]:
                combinations.append((prompt, answer))
    if len(set(combinations)) != len(combinations):
        raise ValueError(f"{category} template combinations are not unique")
    return combinations


def build_rows(*, per_category: int = DEFAULT_PER_CATEGORY,
               seed: int = DEFAULT_SEED) -> list[dict[str, Any]]:
    """Build the deterministic benign chat rows for one seed.

    The same ``seed`` and ``per_category`` always produce the same rows in the
    same order. A different seed selects a different combination subset when
    the pool is larger than ``per_category``. Every normalized
    ``(user, assistant)`` pair is unique.
    """

    if isinstance(per_category, bool) or not isinstance(per_category, int):
        raise ValueError("per_category must be an int")
    if per_category < 1:
        raise ValueError("per_category must be at least 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")

    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for category_index, category in enumerate(CATEGORY_ORDER):
        combinations = _combinations(category)
        if per_category > len(combinations):
            raise ValueError(
                f"category {category!r} has only {len(combinations)} "
                f"template combinations; per_category {per_category} is too "
                f"large")
        indices = list(range(len(combinations)))
        random.Random(seed + category_index * 100003).shuffle(indices)
        chosen = sorted(indices[:per_category])
        for index in chosen:
            user, assistant = combinations[index]
            row = {
                "user": user,
                "assistant": assistant,
                "family": category,
                "source": SOURCE,
                "style": STYLE,
            }
            if not validate_row(row):
                raise ValueError(
                    f"invalid {category!r} row at combination {index}")
            key = (normalize_text(user), normalize_text(assistant))
            if key in seen:
                raise ValueError(
                    f"duplicate (user, assistant) pair in {category!r}")
            seen.add(key)
            rows.append(row)
    if len(rows) != per_category * len(CATEGORY_ORDER):
        raise ValueError(
            f"expected {per_category * len(CATEGORY_ORDER)} rows, "
            f"built {len(rows)}")
    return rows


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as UTF-8 JSONL text with a trailing newline."""

    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def build_manifest(rows: Sequence[Mapping[str, Any]], *,
                   seed: int = DEFAULT_SEED,
                   per_category: int = DEFAULT_PER_CATEGORY) -> dict[str, Any]:
    """Build a content-free manifest for ``rows``.

    Only aggregate counts, the seed, configuration and a hash are included.
    No prompt or answer text is copied into the manifest.
    """

    content = render_jsonl(rows).encode("utf-8")
    counts = Counter(row["family"] for row in rows)
    return {
        "schema": SCHEMA_VERSION,
        "source": SOURCE,
        "style": STYLE,
        "seed": int(seed),
        "per_category": int(per_category),
        "total": len(rows),
        "categories": list(CATEGORY_ORDER),
        "counts": {
            "by_category": {
                category: counts.get(category, 0)
                for category in CATEGORY_ORDER
            },
        },
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "content_free": True,
    }


def write_dataset(rows: Sequence[Mapping[str, Any]],
                  out: str | Path = DEFAULT_OUT, *,
                  manifest: str | Path = DEFAULT_MANIFEST,
                  seed: int = DEFAULT_SEED,
                  per_category: int = DEFAULT_PER_CATEGORY,
                  force: bool = False) -> dict[str, Any]:
    """Write the benign chat JSONL and manifest.

    Existing output or manifest files are never overwritten unless ``force``
    is true. Nothing is written when either path already exists.
    """

    out_path = Path(out).expanduser()
    manifest_path = Path(manifest).expanduser()
    if out_path.resolve() == manifest_path.resolve():
        raise ValueError("--out and --manifest must be different files")
    for path, label in ((out_path, "output"), (manifest_path, "manifest")):
        if path.exists() and not force:
            raise FileExistsError(
                f"{label} already exists: {path}; use --force to overwrite")
    content = render_jsonl(rows)
    summary = build_manifest(
        rows, seed=seed, per_category=per_category)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content.encode("utf-8"))
    manifest_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8",
        newline="\n")
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"training JSONL path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help=f"content-free manifest path (default: {DEFAULT_MANIFEST})")
    parser.add_argument(
        "--per-category", type=int, default=DEFAULT_PER_CATEGORY,
        help=f"rows per category (default: {DEFAULT_PER_CATEGORY})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"deterministic RNG seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output and manifest files")
    args = parser.parse_args(argv)
    try:
        rows = build_rows(
            per_category=args.per_category, seed=args.seed)
        summary = write_dataset(
            rows, args.out, manifest=args.manifest, seed=args.seed,
            per_category=args.per_category, force=args.force)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
