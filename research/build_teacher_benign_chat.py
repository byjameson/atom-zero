"""Build teacher-generated benign chat rows for a later training mix.

    python research/build_teacher_benign_chat.py \
        --model Qwen/Qwen2.5-0.5B-Instruct \
        --device cuda \
        --rows 400 \
        --batch-size 8 \
        --max-new-tokens 96 \
        --seed 41 \
        --out data/teacher_benign_v1/train.jsonl \
        --manifest data/teacher_benign_v1/manifest.json \
        --local-files-only

The 1.5.0 default ships 480 authored benign chat rows from
``research/build_benign_chat.py``. That pool is small and visible: every answer
is a template. This builder keeps authored, non-personal user prompts but asks a
larger local teacher for the answers, so the replay mix can grow beyond the
canned wording while still being filtered for safety and style.

Pipeline::

    authored benign prompt (eight categories)
      -> teacher chat template, greedy decoding
      -> clean answer text
      -> lexical + safety validation
      -> canonical benign_chat row
      -> train.jsonl + content-free manifest.json

Output rows contain exactly::

    {"user": "...", "assistant": "...", "family": "identity",
     "source": "teacher_generated",
     "teacher_model": "Qwen/Qwen2.5-0.5B-Instruct",
     "style": "benign_chat"}

Every accepted answer must be 20-300 characters, at least five words, free of
refusal markers, credential-like fragments, personal data and operational harm.
A prompt and a normalized ``(user, assistant)`` pair may appear at most once.
The local ``atomzero.safety.SafetyFilter`` must accept the answer.

The teacher callable is injectable as ``generator(prompts) -> list[str]``. The
tests use a fake generator and never download or load weights. This builder
itself only writes JSONL and a manifest; it does not train a model. Any output
must pass the product suite before it is added to a training mix. See
``docs/TEACHER_BENIGN_CHAT.md``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from atomzero.safety import SafetyFilter

SCHEMA_VERSION = "teacher_benign_chat_v1"
SOURCE = "teacher_generated"
STYLE = "benign_chat"
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_DEVICE = "cuda"
DEFAULT_ROWS = 400
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_NEW_TOKENS = 96
DEFAULT_SEED = 41
DEFAULT_OUT = "data/teacher_benign_v1/train.jsonl"
DEFAULT_MANIFEST = "data/teacher_benign_v1/manifest.json"
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
REQUIRED_ROW_FIELDS = (
    "user",
    "assistant",
    "family",
    "source",
    "teacher_model",
    "style",
)
MIN_ANSWER_CHARS = 20
MAX_ANSWER_CHARS = 300
MIN_ANSWER_WORDS = 5
MAX_USER_CHARS = 200
PROMPTS_PER_CATEGORY = 96

TEACHER_SYSTEM_PROMPT = (
    "You write short training conversations for Atom Zero, a small local AI "
    "assistant. Reply to the user's benign message with a helpful answer of "
    "one to three sentences. Be concrete, friendly and neutral. Do not refuse, "
    "do not lecture, and do not mention that you are generating data. Return "
    "only the answer text."
)

# Broad refusal markers: a benign reply must not read like a refusal.
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
    "cannot assist",
    "can't assist",
)
# Credential-like fragments that have no place in a benign rehearsal answer.
UNSAFE_TERMS = (
    "password=",
    "api key",
    "secret key",
    "private key",
    "bearer ",
    "sk-",
    "ghp_",
)
# Narrow operational-harm phrases. The SafetyFilter covers explicit requests;
# this list also rejects affirmative answers that describe the same harm.
OPERATIONAL_HARM = (
    "how to make a bomb",
    "how to build a bomb",
    "make a bomb",
    "build a bomb",
    "make explosives",
    "build explosives",
    "how to steal",
    "steal a car",
    "break into a house",
    "hack into",
    "hurt someone",
    "kill someone",
    "poison someone",
    "commit fraud",
    "launder money",
    "buy drugs",
    "hide a body",
)
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_URL_RE = re.compile(r"(?:https?://|www\.)", re.IGNORECASE)
_PHONE_RE = re.compile(r"(?<!\d)\+?\d[\d ().-]{7,}\d(?!\d)")
_LONG_DIGITS_RE = re.compile(r"\d{7,}")
_PERSONAL_CLAIM_RE = re.compile(
    r"\b(?:my|your)\s+home\s+address\b"
    r"|\bsocial\s+security\s+number\b"
    r"|\bcredit\s+card\s+number\b",
    re.IGNORECASE,
)
_FENCE_RE = re.compile(r"^```[a-zA-Z0-9_-]*\s*")
_FENCE_END_RE = re.compile(r"\s*```$")
_PREFIX_RE = re.compile(r"^(?:assistant|answer|response)\s*:\s*", re.IGNORECASE)
_BULLET_RE = re.compile(r"^[-*]\s+")
_QUOTES = "\"'`\u201c\u201d\u2018\u2019"

Teacher = Callable[[Sequence[str]], Sequence[str]]

__all__ = [
    "CATEGORY_ORDER",
    "DEFAULT_BATCH_SIZE",
    "DEFAULT_DEVICE",
    "DEFAULT_MANIFEST",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_MODEL",
    "DEFAULT_OUT",
    "DEFAULT_ROWS",
    "DEFAULT_SEED",
    "MAX_ANSWER_CHARS",
    "MIN_ANSWER_CHARS",
    "MIN_ANSWER_WORDS",
    "OPERATIONAL_HARM",
    "PROMPTS_PER_CATEGORY",
    "REFUSAL_MARKERS",
    "REQUIRED_ROW_FIELDS",
    "SCHEMA_VERSION",
    "SOURCE",
    "STYLE",
    "TEACHER_SYSTEM_PROMPT",
    "UNSAFE_TERMS",
    "available_prompts",
    "build_manifest",
    "build_rows",
    "clean_answer",
    "coerce_answers",
    "load_teacher",
    "main",
    "normalize_text",
    "prompt_pool",
    "render_jsonl",
    "rejection_reason",
    "validate_answer",
    "validate_row",
    "validate_user",
    "write_dataset",
]

# 16 topics x 6 frames = 96 authored prompts for each of these categories.
_TOPIC_FRAMES = (
    "Can you give me a short, practical answer about {topic}?",
    "What is the most useful thing to know about {topic}?",
    "I'm curious about {topic}. Where should I start?",
    "Can you explain {topic} in simple terms?",
    "What should a beginner know about {topic}?",
    "Any quick tips for {topic}?",
)
_TOPICS: dict[str, tuple[str, ...]] = {
    "explanation": (
        "transformer models",
        "retrieval-augmented generation",
        "application programming interfaces",
        "caching",
        "version control with Git",
        "encryption",
        "the JSON data format",
        "relational databases",
        "operating systems",
        "internet routing",
        "machine learning",
        "search indexes",
        "public-key cryptography",
        "the difference between RAM and disk storage",
        "compilers",
        "the client-server model",
    ),
    "cooking": (
        "boiling an egg",
        "making simple pasta",
        "storing bread",
        "cooking white rice",
        "roasting vegetables",
        "brewing a good cup of tea",
        "making pancakes",
        "mixing a salad dressing",
        "baking a simple cake",
        "preparing a quick breakfast",
        "keeping herbs fresh",
        "making vegetable soup",
        "cooking dried beans",
        "seasoning food without salt",
        "packing a lunch",
        "washing fresh produce",
    ),
    "travel": (
        "packing a carry-on bag",
        "planning a day trip",
        "arriving at the airport on time",
        "reducing jet lag",
        "budgeting for a trip",
        "using public transit in a new city",
        "preparing travel documents",
        "choosing a hotel",
        "staying safe in an unfamiliar city",
        "traveling with only hand luggage",
        "planning a train journey",
        "keeping travel costs down",
        "visiting a museum",
        "dealing with a flight delay",
        "choosing travel insurance",
        "packing for rainy weather",
    ),
    "coding": (
        "Python functions",
        "debugging a typo",
        "for loops and while loops",
        "lists and dictionaries",
        "writing a good git commit",
        "unit testing",
        "naming variables",
        "reading a Python traceback",
        "handling exceptions",
        "writing clear comments",
        "refactoring small functions",
        "using version control safely",
        "writing a README file",
        "choosing data types",
        "reading command-line arguments",
        "splitting code into modules",
    ),
    "planning": (
        "planning a study week",
        "prioritizing tasks",
        "making a useful checklist",
        "building a morning routine",
        "breaking a project into steps",
        "writing a meeting agenda",
        "doing a weekly review",
        "starting a new habit",
        "planning a busy day",
        "setting realistic goals",
        "scheduling focused work",
        "keeping a simple task list",
        "preparing for an exam",
        "planning a small event",
        "managing interruptions",
        "estimating how long a task will take",
    ),
    "general_knowledge": (
        "the capital of France",
        "the boiling point of water",
        "the speed of light",
        "the largest planet in the solar system",
        "the largest ocean on Earth",
        "the number of continents",
        "photosynthesis",
        "the color of the sky",
        "the water cycle",
        "the function of the heart",
        "the difference between weather and climate",
        "the life cycle of a butterfly",
        "the Great Wall of China",
        "the Amazon rainforest",
        "the Moon's phases",
        "the layers of the Earth",
    ),
}
# 8 identity topics x 12 frames = 96 prompts.
_IDENTITY_TOPICS = (
    "who you are",
    "what you can do",
    "how you run locally",
    "your limits",
    "which languages you support",
    "how you handle privacy",
    "what kind of model you are",
    "how you differ from a search engine",
)
_IDENTITY_FRAMES = (
    "Can you tell me about {topic}?",
    "I'd like to know about {topic}.",
    "Please explain {topic}.",
    "I'm wondering about {topic}.",
    "Could you describe {topic}?",
    "Help me understand {topic}.",
    "What should I know about {topic}?",
    "Give me a brief overview of {topic}.",
    "Can you answer a question about {topic}?",
    "I have a question about {topic}.",
    "What is the short version of {topic}?",
    "Could you walk me through {topic}?",
)
# 8 salutations x 12 contexts = 96 conversational turns.
_SALUTATIONS = (
    "Hello!",
    "Hi!",
    "Hey!",
    "Good morning!",
    "Good afternoon!",
    "Good evening!",
    "Greetings!",
    "Hi there!",
)
_GREETING_CONTEXTS = (
    "How are you?",
    "Can you help me with something?",
    "I have a quick question.",
    "I hope you are having a good day.",
    "What can you do?",
    "Are you ready to help?",
    "I'm glad to meet you.",
    "Could we talk for a moment?",
    "Thanks for being here.",
    "I have a small task for you.",
    "How is your day going?",
    "Let's get started.",
)


def normalize_text(value: Any) -> str:
    """Case-fold and collapse whitespace for narrow lexical comparison."""

    return " ".join(
        str(value or "").replace("\u2019", "'").split()).casefold()


def clean_answer(value: Any) -> str:
    """Turn one raw teacher reply into one normalized answer line.

    Fences, role prefixes, matching outer quotes and internal line breaks are
    removed. The validator still applies the length, word and safety rules to
    the cleaned text.
    """

    text = str(value or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = _FENCE_RE.sub("", text)
    text = _FENCE_END_RE.sub("", text).strip()
    text = text.strip(_QUOTES).strip()
    text = _BULLET_RE.sub("", text).strip()
    text = _PREFIX_RE.sub("", text).strip()
    return " ".join(text.split())


def coerce_answers(values: Any, expected: int) -> list[str]:
    """Normalize a generator return value to exactly ``expected`` raw answers.

    A string is treated as a single answer. Missing answers become empty
    strings and are rejected by validation; extra answers are ignored.
    """

    if isinstance(values, str):
        items: list[Any] = [values]
    elif values is None:
        items = []
    else:
        try:
            items = list(values)
        except TypeError:
            items = []
    normalized = [
        item if isinstance(item, str) else ("" if item is None else str(item))
        for item in items[:expected]
    ]
    return normalized + [""] * (expected - len(normalized))


def _contains_personal_data(value: str) -> bool:
    return bool(
        _EMAIL_RE.search(value)
        or _URL_RE.search(value)
        or _PHONE_RE.search(value)
        or _LONG_DIGITS_RE.search(value)
        or _PERSONAL_CLAIM_RE.search(value)
    )


def validate_user(value: Any) -> bool:
    """Return True iff ``value`` is a short, safe, single-line prompt."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in text or "\r" in text:
        return False
    if len(text) > MAX_USER_CHARS:
        return False
    if _contains_personal_data(text):
        return False
    return SafetyFilter().check(text) is None


def validate_answer(value: Any, *, safety: SafetyFilter | None = None) -> bool:
    """Return True iff ``value`` is a short, safe, non-refusal answer.

    Rules: non-empty, 20-300 characters, at least five words, no refusal
    marker, no credential-like fragment, no personal data, no operational-harm
    phrase, and no ``atomzero.safety.SafetyFilter`` verdict.
    """

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text or "\n" in text or "\r" in text:
        return False
    if not MIN_ANSWER_CHARS <= len(text) <= MAX_ANSWER_CHARS:
        return False
    if len(text.split()) < MIN_ANSWER_WORDS:
        return False
    reply = normalize_text(text)
    if any(marker in reply for marker in REFUSAL_MARKERS):
        return False
    if any(term in reply for term in UNSAFE_TERMS):
        return False
    if any(term in reply for term in OPERATIONAL_HARM):
        return False
    if _contains_personal_data(text):
        return False
    checker = safety or SafetyFilter()
    return checker.check(text) is None


def validate_row(row: Mapping[str, Any]) -> bool:
    """Return True iff ``row`` has the canonical teacher benign chat schema."""

    if set(row) != set(REQUIRED_ROW_FIELDS):
        return False
    if row.get("family") not in CATEGORY_ORDER:
        return False
    if row.get("source") != SOURCE:
        return False
    if row.get("style") != STYLE:
        return False
    if not isinstance(row.get("teacher_model"), str):
        return False
    if not str(row.get("teacher_model") or "").strip():
        return False
    if not validate_user(row.get("user")):
        return False
    if not validate_answer(row.get("assistant")):
        return False
    return True


def rejection_reason(prompt: str, answer: str, *,
                     safety: SafetyFilter | None = None,
                     seen_users: set[str] | None = None,
                     seen_pairs: set[tuple[str, str]] | None = None
                     ) -> str | None:
    """Return a short rejection reason, or None when the candidate is valid."""

    if not validate_user(prompt):
        return "invalid_prompt"
    if not validate_answer(answer, safety=safety):
        return "invalid_answer"
    user_key = normalize_text(prompt)
    if seen_users is not None and user_key in seen_users:
        return "duplicate_prompt"
    pair_key = (user_key, normalize_text(answer))
    if seen_pairs is not None and pair_key in seen_pairs:
        return "duplicate_pair"
    return None


def _category_prompts(category: str) -> tuple[str, ...]:
    if category == "identity":
        prompts = tuple(
            frame.format(topic=topic)
            for topic in _IDENTITY_TOPICS
            for frame in _IDENTITY_FRAMES
        )
    elif category == "greeting":
        prompts = tuple(
            f"{salutation} {context}"
            for salutation in _SALUTATIONS
            for context in _GREETING_CONTEXTS
        )
    elif category in _TOPICS:
        prompts = tuple(
            frame.format(topic=topic)
            for topic in _TOPICS[category]
            for frame in _TOPIC_FRAMES
        )
    else:
        raise ValueError(f"unknown category {category!r}")
    if len(prompts) != PROMPTS_PER_CATEGORY:
        raise ValueError(
            f"category {category!r} has {len(prompts)} prompts, expected "
            f"{PROMPTS_PER_CATEGORY}")
    if len(set(prompts)) != len(prompts):
        raise ValueError(f"category {category!r} repeats a prompt")
    for prompt in prompts:
        if not validate_user(prompt):
            raise ValueError(
                f"category {category!r} has an invalid prompt: {prompt!r}")
    return prompts


@lru_cache(maxsize=None)
def prompt_pool(category: str) -> tuple[str, ...]:
    """Return the validated, deterministic authored prompt pool."""

    return _category_prompts(category)


def available_prompts() -> int:
    """Return the total number of unique authored prompts."""

    return sum(len(prompt_pool(category)) for category in CATEGORY_ORDER)


def _quotas(rows: int) -> dict[str, int]:
    base, remainder = divmod(rows, len(CATEGORY_ORDER))
    return {
        category: base + (1 if index < remainder else 0)
        for index, category in enumerate(CATEGORY_ORDER)
    }


def _empty_counts(rows: int = 0) -> dict[str, Any]:
    quotas = _quotas(rows) if rows else {
        category: 0 for category in CATEGORY_ORDER
    }
    return {
        "requested": int(rows),
        "accepted": 0,
        "rejected": 0,
        "duplicates": 0,
        "attempted": 0,
        "shortfall": int(rows),
        "by_category": {category: 0 for category in CATEGORY_ORDER},
        "requested_by_category": quotas,
        "rejected_by_category": {category: 0 for category in CATEGORY_ORDER},
    }


def build_rows(generator: Teacher | None, *,
               rows: int = DEFAULT_ROWS,
               model: str = DEFAULT_MODEL,
               seed: int = DEFAULT_SEED,
               batch_size: int = DEFAULT_BATCH_SIZE,
               progress: Callable[[Mapping[str, Any]], None] | None = None
               ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate, validate and return ``(rows, counts)``.

    ``generator`` is called with a list of user prompts and must return one
    answer per prompt. It is injectable so tests never load a model. The target
    is ``rows`` accepted rows; validation failures are retried with the next
    unused prompt until the pool is exhausted.
    """

    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 0:
        raise ValueError("rows must be a non-negative int")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an int")
    if (isinstance(batch_size, bool) or not isinstance(batch_size, int)
            or batch_size < 1):
        raise ValueError("batch_size must be a positive int")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("model must be a non-empty string")
    if rows > available_prompts():
        raise ValueError(
            f"rows {rows} exceeds the {available_prompts()} unique authored "
            "prompts; add prompts instead of reusing one")
    if rows == 0:
        return [], _empty_counts(0)
    if not callable(generator):
        raise ValueError("generator must be callable")

    quotas = _quotas(rows)
    counts = _empty_counts(rows)
    safety = SafetyFilter()
    rows_out: list[dict[str, Any]] = []
    seen_users: set[str] = set()
    seen_pairs: set[tuple[str, str]] = set()

    for category_index, category in enumerate(CATEGORY_ORDER):
        quota = quotas[category]
        if quota == 0:
            continue
        prompts = list(prompt_pool(category))
        random.Random(seed + category_index * 100003).shuffle(prompts)
        accepted_here = 0
        cursor = 0
        while accepted_here < quota and cursor < len(prompts):
            batch = prompts[cursor:cursor + batch_size]
            cursor += len(batch)
            answers = coerce_answers(generator(list(batch)), len(batch))
            for prompt, raw_answer in zip(batch, answers):
                counts["attempted"] += 1
                answer = clean_answer(raw_answer)
                reason = rejection_reason(
                    prompt, answer, safety=safety, seen_users=seen_users,
                    seen_pairs=seen_pairs)
                if reason is not None:
                    counts["rejected"] += 1
                    counts["rejected_by_category"][category] += 1
                    if reason in ("duplicate_prompt", "duplicate_pair"):
                        counts["duplicates"] += 1
                    continue
                row = {
                    "user": prompt,
                    "assistant": answer,
                    "family": category,
                    "source": SOURCE,
                    "teacher_model": model,
                    "style": STYLE,
                }
                if not validate_row(row):
                    counts["rejected"] += 1
                    counts["rejected_by_category"][category] += 1
                    continue
                seen_users.add(normalize_text(prompt))
                seen_pairs.add(
                    (normalize_text(prompt), normalize_text(answer)))
                rows_out.append(row)
                accepted_here += 1
                counts["accepted"] += 1
                counts["by_category"][category] += 1
                if accepted_here >= quota:
                    break
            if progress is not None:
                progress(dict(counts))
        if accepted_here < quota:
            counts.setdefault("exhausted_categories", []).append(category)

    counts["shortfall"] = int(rows) - counts["accepted"]
    return rows_out, counts


def render_jsonl(rows: Sequence[Mapping[str, Any]]) -> str:
    """Render rows as UTF-8 JSONL text with a trailing newline."""

    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)


def build_manifest(rows: Sequence[Mapping[str, Any]],
                   counts: Mapping[str, Any], *,
                   model: str = DEFAULT_MODEL,
                   seed: int = DEFAULT_SEED,
                   batch_size: int = DEFAULT_BATCH_SIZE,
                   max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS
                   ) -> dict[str, Any]:
    """Build a content-free manifest for the accepted rows.

    Only aggregate counts, configuration, the teacher id and a SHA-256 of the
    exact JSONL bytes are included. No prompt or answer text is copied.
    """

    content = render_jsonl(rows).encode("utf-8")
    accepted_by_category = Counter(row["family"] for row in rows)
    requested = int(counts.get("requested", len(rows)))
    accepted = int(counts.get("accepted", len(rows)))
    rejected = int(counts.get("rejected", 0))
    if accepted != len(rows):
        raise ValueError("counts accepted does not match the row list")
    if accepted + int(counts.get("shortfall", 0)) != requested:
        raise ValueError("counts must satisfy accepted + shortfall == requested")
    return {
        "schema": SCHEMA_VERSION,
        "source": SOURCE,
        "style": STYLE,
        "teacher_model": str(model),
        "seed": int(seed),
        "requested": requested,
        "accepted": accepted,
        "rejected": rejected,
        "duplicates": int(counts.get("duplicates", 0)),
        "attempted": int(counts.get("attempted", accepted + rejected)),
        "shortfall": int(counts.get("shortfall", 0)),
        "categories": list(CATEGORY_ORDER),
        "by_category": {
            category: accepted_by_category.get(category, 0)
            for category in CATEGORY_ORDER
        },
        "requested_by_category": {
            category: int(counts.get("requested_by_category", {}).get(
                category, 0))
            for category in CATEGORY_ORDER
        },
        "rejected_by_category": {
            category: int(counts.get("rejected_by_category", {}).get(
                category, 0))
            for category in CATEGORY_ORDER
        },
        "batch_size": int(batch_size),
        "max_new_tokens": int(max_new_tokens),
        "sha256": hashlib.sha256(content).hexdigest(),
        "bytes": len(content),
        "content_free": True,
    }


def write_dataset(rows: Sequence[Mapping[str, Any]],
                  out: str | Path = DEFAULT_OUT, *,
                  manifest: str | Path = DEFAULT_MANIFEST,
                  counts: Mapping[str, Any],
                  model: str = DEFAULT_MODEL,
                  seed: int = DEFAULT_SEED,
                  batch_size: int = DEFAULT_BATCH_SIZE,
                  max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                  force: bool = False) -> dict[str, Any]:
    """Write the JSONL and content-free manifest.

    Existing files are never overwritten unless ``force`` is true. Nothing is
    written when either path already exists.
    """

    out_path = Path(out).expanduser()
    manifest_path = Path(manifest).expanduser()
    if out_path.resolve() == manifest_path.resolve():
        raise ValueError("--out and --manifest must be different files")
    for path, label in ((out_path, "output"), (manifest_path, "manifest")):
        if path.exists() and not force:
            raise FileExistsError(
                f"{label} already exists: {path}; use --force to overwrite")
    summary = build_manifest(
        rows, counts, model=model, seed=seed, batch_size=batch_size,
        max_new_tokens=max_new_tokens)
    content = render_jsonl(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(content.encode("utf-8"))
    manifest_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    return summary


def _looks_like_local_path(model_name: str) -> bool:
    path = Path(model_name).expanduser()
    return bool(
        path.is_absolute()
        or model_name.startswith((".", "~"))
        or "\\" in model_name
        or path.is_dir()
    )


def _ensure_local_model(model_name: str, local_files_only: bool) -> None:
    if not local_files_only:
        return
    if _looks_like_local_path(model_name) and not Path(
            model_name).expanduser().exists():
        raise FileNotFoundError(
            f"local teacher path does not exist: {model_name}; pass a cached "
            "model id or omit --local-files-only to allow a download")


def load_teacher(model_name: str = DEFAULT_MODEL, *,
                 device: str = DEFAULT_DEVICE,
                 max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
                 local_files_only: bool = True) -> Teacher:
    """Load a greedy chat teacher and return ``generator(prompts)``.

    Torch and transformers are imported lazily so tests and the zero-row path
    never require either package. A missing local path fails before import.
    """

    _ensure_local_model(model_name, local_files_only)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "teacher loading requires torch and transformers; install them "
            "or inject a generator in tests") from exc
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested but torch.cuda.is_available() is false; "
            "pass --device cpu for a slow CPU smoke test")
    dtype = torch.bfloat16 if device.startswith("cuda") else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
        local_files_only=local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, local_files_only=local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    model.eval()

    def generator(prompts: Sequence[str]) -> list[str]:
        texts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": TEACHER_SYSTEM_PROMPT},
                    {"role": "user", "content": str(prompt)},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
            for prompt in prompts
        ]
        inputs = tokenizer(
            texts, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id)
        input_length = inputs["input_ids"].shape[1]
        return [
            tokenizer.decode(
                output[index][input_length:], skip_special_tokens=True)
            for index in range(len(texts))
        ]

    return generator


def _progress_printer(interval: int = 8) -> Callable[[Mapping[str, Any]], None]:
    batches = 0

    def report(counts: Mapping[str, Any]) -> None:
        nonlocal batches
        batches += 1
        if batches % interval:
            return
        print(json.dumps({
            "accepted": int(counts.get("accepted", 0)),
            "requested": int(counts.get("requested", 0)),
            "rejected": int(counts.get("rejected", 0)),
            "duplicates": int(counts.get("duplicates", 0)),
        }), file=sys.stderr, flush=True)

    return report


def main(argv: Sequence[str] | None = None, *,
         teacher_factory: Callable[..., Teacher] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"teacher model id or local path (default: {DEFAULT_MODEL})")
    parser.add_argument(
        "--device", default=DEFAULT_DEVICE,
        help=f"torch device for the teacher (default: {DEFAULT_DEVICE})")
    parser.add_argument(
        "--rows", type=int, default=DEFAULT_ROWS,
        help=f"accepted rows to target (default: {DEFAULT_ROWS})")
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"prompts per teacher call (default: {DEFAULT_BATCH_SIZE})")
    parser.add_argument(
        "--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS,
        help=f"teacher generation budget (default: {DEFAULT_MAX_NEW_TOKENS})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"deterministic prompt selection and ordering (default: "
             f"{DEFAULT_SEED})")
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"training JSONL path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--manifest", default=DEFAULT_MANIFEST,
        help=f"content-free manifest path (default: {DEFAULT_MANIFEST})")
    parser.add_argument(
        "--local-files-only", action=argparse.BooleanOptionalAction,
        default=True,
        help="do not download the teacher (default: true)")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite existing output and manifest files")
    args = parser.parse_args(argv)

    try:
        if args.rows < 0:
            raise ValueError("--rows must be non-negative")
        if args.batch_size < 1:
            raise ValueError("--batch-size must be positive")
        if args.max_new_tokens < 1:
            raise ValueError("--max-new-tokens must be positive")
    except ValueError as exc:
        parser.error(str(exc))

    if args.rows == 0:
        generator: Teacher = lambda prompts: []  # noqa: E731
    elif teacher_factory is not None:
        generator = teacher_factory(
            args.model, device=args.device,
            max_new_tokens=args.max_new_tokens,
            local_files_only=args.local_files_only)
    else:
        try:
            generator = load_teacher(
                args.model, device=args.device,
                max_new_tokens=args.max_new_tokens,
                local_files_only=args.local_files_only)
        except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    try:
        rows, counts = build_rows(
            generator, rows=args.rows, model=args.model, seed=args.seed,
            batch_size=args.batch_size, progress=_progress_printer())
        summary = write_dataset(
            rows, args.out, manifest=args.manifest, counts=counts,
            model=args.model, seed=args.seed, batch_size=args.batch_size,
            max_new_tokens=args.max_new_tokens, force=args.force)
    except (FileExistsError, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
