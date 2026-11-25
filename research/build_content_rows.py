"""Build deterministic content replay rows from a local facts JSONL corpus.

Mixed copy + content full fine-tuning needs a content rehearsal corpus. The
repository has no large pretraining text, but a local facts file (for example
``world_facts.jsonl``) provides rows shaped::

    {"id": "...", "title": "...", "lang": "en", "text": "..."}

This builder turns those facts into deterministic prompt/answer rows so a
candidate can rehearse factual, source-grounded text while it learns copy
skills. There is no model call and no download: every answer is a trimmed
extract from the fact's own ``text`` field.

Three styles are supported:

``qa``
    "What does the source say about <title>?" -> a short extract.
``summary``
    "Summarize the source about <title>." -> the same extract with a lead-in.
``citation``
    A source-cited request -> the answer names the title and quotes an extract.

Rows use the repository schema plus provenance::

    {"user": "...", "assistant": "...", "family": "content_qa",
     "source_id": "...", "source_title": "...", "style": "qa"}

Facts are skipped when malformed, too short, non-English (unless ``--lang``
selects another language), or a duplicate title. Eligible facts are shuffled
deterministically with ``--seed`` before ``--max-rows`` is applied. The
manifest is content-free: counts, skip reasons, style, seed, output hash and
the source file basename only. It never contains prompts, answers or titles.

Content rows are rehearsal data. They are not a factuality claim, a verified
knowledge base, or evidence that the model has learned the facts in its
weights.

Command::

    python research/build_content_rows.py `
      --facts "C:\\path\\to\\world_facts.jsonl" `
      --out data/content_replay_v1/train.jsonl `
      --manifest data/content_replay_v1/manifest.json `
      --max-rows 5000 --min-chars 200 --max-answer-chars 400 `
      --style qa --seed 13

See ``docs/CONTENT_ROWS.md`` for the copy-mix workflow and caveats.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "atomzero.content_rows.v1"
DEFAULT_OUT = "data/content_replay_v1/train.jsonl"
DEFAULT_STYLE = "qa"
DEFAULT_MAX_ROWS = 5000
DEFAULT_MIN_CHARS = 200
DEFAULT_MAX_ANSWER_CHARS = 400
DEFAULT_SEED = 13
DEFAULT_LANG = "en"
STYLES = ("qa", "summary", "citation")
SKIP_REASONS = (
    "malformed",
    "non_english",
    "short",
    "duplicate_title",
    "answer_budget_too_small",
)
MANIFEST_NOTE = (
    "Content replay rows are rehearsal data derived from a local source "
    "corpus. This manifest is content-free and is not a factuality claim, a "
    "verified dataset, or evidence of model capability.")

_TERMINAL_RE = re.compile(r"[.!?][\"')\]]*")
_ABBREVIATIONS = frozenset({
    "al", "approx", "ca", "cf", "dept", "dr", "e.g", "est", "etc", "fig",
    "i.e", "jr", "mr", "mrs", "ms", "mt", "no", "p.m", "prof", "sr", "st",
    "u.k", "u.s", "vs", "vol",
})

__all__ = [
    "ContentRowsError",
    "DEFAULT_LANG",
    "DEFAULT_MAX_ANSWER_CHARS",
    "DEFAULT_MAX_ROWS",
    "DEFAULT_MIN_CHARS",
    "DEFAULT_OUT",
    "DEFAULT_SEED",
    "DEFAULT_STYLE",
    "MANIFEST_NOTE",
    "SCHEMA",
    "SKIP_REASONS",
    "STYLES",
    "build_rows",
    "clean_extract",
    "main",
    "read_facts",
    "run_build",
]


class ContentRowsError(ValueError):
    """Invalid build request; the CLI prints the message and exits non-zero."""


def _resolve(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = ROOT / value
    return value.resolve()


def _basename(path: str | Path) -> str:
    return _resolve(path).name


def _clean_text(value: str) -> str:
    """Collapse whitespace without adding or rewriting words."""

    return " ".join(value.split())


def _clean_title(value: str) -> str:
    return _clean_text(value)


def _normalized_title(value: str) -> str:
    return unicodedata.normalize("NFKC", _clean_text(value)).casefold()


def _lang_matches(value: str, wanted: str) -> bool:
    actual = value.strip().lower().replace("_", "-")
    target = wanted.strip().lower().replace("_", "-")
    if not actual or not target:
        return False
    return (actual == target
            or actual.split("-", 1)[0] == target.split("-", 1)[0])


def _sentence_ends(text: str) -> list[int]:
    """Return end offsets of likely sentence boundaries in ``text``."""

    ends: list[int] = []
    for match in _TERMINAL_RE.finditer(text):
        end = match.end()
        if end < len(text) and not text[end].isspace():
            continue
        following = text[end:].lstrip()[:1]
        if following and following.islower():
            continue
        if match.group(0)[0] == ".":
            token_match = re.search(r"([A-Za-z][A-Za-z.]*)$", text[:match.start()])
            if token_match is not None:
                token = token_match.group(1).lower().rstrip(".")
                if len(token) == 1 or token in _ABBREVIATIONS:
                    continue
        ends.append(end)
    return ends


def clean_extract(text: str, limit: int) -> str:
    """Return at most ``limit`` characters of ``text``, preferring sentences.

    ``text`` is expected to be whitespace-normalized. The result is always a
    contiguous prefix of ``text``; no words are invented. If a sentence
    boundary fits within the limit, the last such boundary is used. Otherwise
    the cut falls back to a word boundary.
    """

    if limit <= 0 or not text:
        return ""
    if len(text) <= limit:
        return text
    boundaries = [end for end in _sentence_ends(text) if end <= limit]
    if boundaries:
        return text[:max(boundaries)].rstrip()
    cut = text.rfind(" ", 0, limit + 1)
    if cut <= 0:
        cut = limit
    return text[:cut].rstrip()


def _validate_int(value: Any, name: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContentRowsError(f"{name} must be an int")
    if value < minimum:
        raise ContentRowsError(f"{name} must be >= {minimum}")
    return value


def _validate_options(*, style: str, max_rows: Any, min_chars: Any,
                      max_answer_chars: Any, seed: Any, lang: str) -> None:
    if style not in STYLES:
        raise ContentRowsError(
            f"unknown style {style!r}; expected one of {STYLES}")
    _validate_int(max_rows, "max_rows", minimum=0)
    _validate_int(min_chars, "min_chars", minimum=0)
    _validate_int(max_answer_chars, "max_answer_chars", minimum=1)
    _validate_int(seed, "seed", minimum=0)
    if not isinstance(lang, str) or not _clean_text(lang):
        raise ContentRowsError("lang must be a non-empty string")


def _prompt_for(style: str, title: str) -> str:
    if style == "qa":
        return f"What does the source say about {title}?"
    if style == "summary":
        return f"Summarize the source about {title}."
    return f"Give a source-cited answer about {title}."


def _prefix_for(style: str, title: str) -> str:
    if style == "qa":
        return ""
    if style == "summary":
        return "In summary: "
    return f'According to the source "{title}", '


def _row_from_fact(fact_id: str, title: str, text: str, style: str,
                   max_answer_chars: int) -> dict[str, str] | None:
    prefix = _prefix_for(style, title)
    budget = max_answer_chars - len(prefix)
    if budget <= 0:
        return None
    extract = clean_extract(text, budget)
    if not extract:
        return None
    return {
        "user": _prompt_for(style, title),
        "assistant": prefix + extract,
        "family": f"content_{style}",
        "source_id": fact_id,
        "source_title": title,
        "style": style,
    }


def build_rows(facts: Iterable[Any], *, style: str = DEFAULT_STYLE,
               max_rows: int = DEFAULT_MAX_ROWS,
               min_chars: int = DEFAULT_MIN_CHARS,
               max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
               seed: int = DEFAULT_SEED,
               lang: str = DEFAULT_LANG,
               ) -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Filter facts, build style rows and sample them deterministically.

    ``facts`` may contain parsed JSON values. Non-objects and objects missing
    a non-empty ``id``, ``title``, ``lang`` or ``text`` are counted as
    malformed. Blank input lines are handled by :func:`read_facts`.
    """

    _validate_options(
        style=style, max_rows=max_rows, min_chars=min_chars,
        max_answer_chars=max_answer_chars, seed=seed, lang=lang)

    skipped: Counter[str] = Counter({reason: 0 for reason in SKIP_REASONS})
    seen_titles: set[str] = set()
    built: list[dict[str, str]] = []
    rows_read = 0

    for raw in facts:
        rows_read += 1
        if not isinstance(raw, Mapping):
            skipped["malformed"] += 1
            continue
        raw_id = raw.get("id")
        raw_title = raw.get("title")
        raw_lang = raw.get("lang")
        raw_text = raw.get("text")
        if not all(isinstance(value, str)
                   for value in (raw_id, raw_title, raw_lang, raw_text)):
            skipped["malformed"] += 1
            continue
        fact_id = _clean_text(raw_id)
        title = _clean_title(raw_title)
        fact_lang = _clean_text(raw_lang)
        text = _clean_text(raw_text)
        if not fact_id or not title or not fact_lang or not text:
            skipped["malformed"] += 1
            continue
        if not _lang_matches(fact_lang, lang):
            skipped["non_english"] += 1
            continue
        if len(text) < min_chars:
            skipped["short"] += 1
            continue
        key = _normalized_title(title)
        if key in seen_titles:
            skipped["duplicate_title"] += 1
            continue
        row = _row_from_fact(
            fact_id, title, text, style, max_answer_chars)
        if row is None:
            skipped["answer_budget_too_small"] += 1
            continue
        seen_titles.add(key)
        built.append(row)

    random.Random(seed).shuffle(built)
    rows = built[:max_rows]
    counts: dict[str, Any] = {
        "style": style,
        "seed": seed,
        "max_rows": max_rows,
        "min_chars": min_chars,
        "max_answer_chars": max_answer_chars,
        "lang": lang,
        "rows_read": rows_read,
        "eligible": len(built),
        "selected": len(rows),
        "skipped": sum(skipped.values()),
        "skip_reasons": {
            reason: skipped[reason] for reason in SKIP_REASONS
        },
    }
    return rows, counts


def read_facts(path: str | Path) -> tuple[list[Any], dict[str, int]]:
    """Read a facts JSONL file, counting blank lines and invalid JSON.

    Invalid JSON lines are represented as ``None`` so :func:`build_rows` can
    report them as malformed. The file is read-only input.
    """

    resolved = _resolve(path)
    if not resolved.is_file():
        raise FileNotFoundError(f"facts file not found: {resolved}")
    text = resolved.read_text(encoding="utf-8-sig")
    values: list[Any] = []
    counts = {"rows_read": 0, "blank_lines": 0}
    for line in text.splitlines():
        if not line.strip():
            counts["blank_lines"] += 1
            continue
        counts["rows_read"] += 1
        try:
            values.append(json.loads(line))
        except json.JSONDecodeError:
            values.append(None)
    return values, counts


def _encode_rows(rows: Sequence[Mapping[str, Any]]) -> bytes:
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    return payload.encode("utf-8")


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = _resolve(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)
        + "\n",
        encoding="utf-8")


def _manifest(counts: Mapping[str, Any], *, facts_path: str | Path,
              out_path: str | Path, blank_lines: int,
              sha256: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "style": counts["style"],
        "seed": counts["seed"],
        "source": _basename(facts_path),
        "output": _basename(out_path),
        "rows_read": counts["rows_read"],
        "blank_lines": blank_lines,
        "eligible": counts["eligible"],
        "selected": counts["selected"],
        "skipped": counts["skipped"],
        "skip_reasons": dict(counts["skip_reasons"]),
        "max_rows": counts["max_rows"],
        "min_chars": counts["min_chars"],
        "max_answer_chars": counts["max_answer_chars"],
        "lang": counts["lang"],
        "sha256": sha256,
        "note": MANIFEST_NOTE,
    }


def run_build(facts_path: str | Path,
              out: str | Path = DEFAULT_OUT, *,
              style: str = DEFAULT_STYLE,
              max_rows: int = DEFAULT_MAX_ROWS,
              min_chars: int = DEFAULT_MIN_CHARS,
              max_answer_chars: int = DEFAULT_MAX_ANSWER_CHARS,
              seed: int = DEFAULT_SEED,
              lang: str = DEFAULT_LANG,
              force: bool = False,
              manifest_path: str | Path | None = None
              ) -> dict[str, Any]:
    """Build rows, write ``out`` and optionally a content-free manifest."""

    _validate_options(
        style=style, max_rows=max_rows, min_chars=min_chars,
        max_answer_chars=max_answer_chars, seed=seed, lang=lang)

    facts = _resolve(facts_path)
    out_path = _resolve(out)
    if not facts.is_file():
        raise FileNotFoundError(f"facts file not found: {facts}")
    if out_path == facts:
        raise ContentRowsError("--out must not overwrite the --facts input")
    if out_path.exists() and not force:
        raise FileExistsError(
            f"refusing to overwrite {out_path}; pass --force to replace it")
    manifest_target = (
        _resolve(manifest_path) if manifest_path is not None else None)
    if manifest_target is not None:
        if manifest_target in {facts, out_path}:
            raise ContentRowsError(
                "--manifest must differ from --facts and --out")

    facts_values, read_counts = read_facts(facts)
    rows, counts = build_rows(
        facts_values, style=style, max_rows=max_rows, min_chars=min_chars,
        max_answer_chars=max_answer_chars, seed=seed, lang=lang)

    data = _encode_rows(rows)
    sha256 = hashlib.sha256(data).hexdigest()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)

    manifest = _manifest(
        counts, facts_path=facts, out_path=out_path,
        blank_lines=read_counts["blank_lines"], sha256=sha256)
    if manifest_target is not None:
        _write_json(manifest_target, manifest)
    return manifest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--facts", required=True,
        help="facts JSONL file with id/title/lang/text rows")
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help=f"output JSONL path (default: {DEFAULT_OUT})")
    parser.add_argument(
        "--max-rows", type=int, default=DEFAULT_MAX_ROWS,
        help=f"maximum rows to write (default: {DEFAULT_MAX_ROWS})")
    parser.add_argument(
        "--min-chars", type=int, default=DEFAULT_MIN_CHARS,
        help=f"minimum normalized fact text length (default: "
             f"{DEFAULT_MIN_CHARS})")
    parser.add_argument(
        "--max-answer-chars", type=int, default=DEFAULT_MAX_ANSWER_CHARS,
        help=f"maximum answer length (default: {DEFAULT_MAX_ANSWER_CHARS})")
    parser.add_argument(
        "--style", choices=STYLES, default=DEFAULT_STYLE,
        help=f"row style (default: {DEFAULT_STYLE})")
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"deterministic selection seed (default: {DEFAULT_SEED})")
    parser.add_argument(
        "--lang", default=DEFAULT_LANG,
        help=f"language to keep, matched by primary subtag "
             f"(default: {DEFAULT_LANG})")
    parser.add_argument(
        "--manifest", default=None,
        help="write the content-free JSON manifest here")
    parser.add_argument(
        "--force", action="store_true",
        help="overwrite --out if it already exists")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        manifest = run_build(
            args.facts, args.out, style=args.style,
            max_rows=args.max_rows, min_chars=args.min_chars,
            max_answer_chars=args.max_answer_chars, seed=args.seed,
            lang=args.lang, force=args.force,
            manifest_path=args.manifest)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    summary = {
        "output": manifest["output"],
        "style": manifest["style"],
        "seed": manifest["seed"],
        "rows_read": manifest["rows_read"],
        "eligible": manifest["eligible"],
        "selected": manifest["selected"],
        "skipped": manifest["skipped"],
        "skip_reasons": manifest["skip_reasons"],
        "sha256": manifest["sha256"],
        "manifest": (_basename(args.manifest)
                     if args.manifest is not None else None),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
