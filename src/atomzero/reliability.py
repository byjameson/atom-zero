"""Deterministic, content-free source reliability ledger.

The ledger records how an upstream claim audit scored observations of a
source: how many claims were supported, weak or unsupported, and how many
conflicts were found. It deliberately stores only a truncated SHA-256 hash of
the trimmed source label. Source text, queries, answers, chunks and session
identifiers never enter an entry.

The score is intentionally simple and local::

    support_ratio = (supported + 0.5 * weak) / claims

It is a summary of audit outcomes, not a statement that a source is true.

    python -m atomzero.reliability --file ledger.jsonl --summary
    python -m atomzero.reliability --file ledger.jsonl --verify
    python -m atomzero.reliability --file ledger.jsonl \\
        --add-source example.org --supported 3 --weak 1 --unsupported 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

__all__ = [
    "add_observation",
    "load_jsonl",
    "main",
    "source_id",
    "summarize",
    "verify",
    "write_jsonl",
]

VERSION = 1
SOURCE_ID_LENGTH = 16
REQUIRED_KEYS = frozenset({
    "version", "at", "source_id", "supported", "weak", "unsupported",
    "claims", "conflicts",
})
FORBIDDEN_KEYS = frozenset({
    "text", "query", "answer", "chunk", "content", "session", "session_id",
    "source", "source_text",
})
COUNT_KEYS = ("supported", "weak", "unsupported", "conflicts")
STATUS_RELIABLE = 0.75
STATUS_MIXED = 0.40
_SOURCE_ID_RE = re.compile(r"\A[0-9a-f]{16}\Z")


def source_id(source: str) -> str:
    """Return the stable 16-hex SHA-256 prefix of a trimmed source label."""
    if not isinstance(source, str):
        raise ValueError("source must be a string")
    label = source.strip()
    if not label:
        raise ValueError("source must not be empty")
    digest = hashlib.sha256(label.encode("utf-8")).hexdigest()
    return digest[:SOURCE_ID_LENGTH]


def _parse_datetime(value, where: str = "timestamp") -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{where} must be an ISO-8601 string")
    text = value.strip()
    if text.endswith(("Z", "z")):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"{where} is not valid ISO-8601: {value!r}") from exc


def _parse_timestamp(value, where: str = "timestamp") -> datetime:
    stamp = _parse_datetime(value, where)
    if stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
        raise ValueError(f"{where} must carry a zero UTC offset")
    return stamp.astimezone(timezone.utc)


def _format_timestamp(value) -> str:
    if isinstance(value, datetime):
        stamp = value
    else:
        stamp = _parse_datetime(value, "now")
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    stamp = stamp.astimezone(timezone.utc).replace(microsecond=0)
    return stamp.isoformat().replace("+00:00", "Z")


def _count(value, name: str, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where}: '{name}' must be a non-negative integer")
    return value


def _content_path(value, path: str = "") -> str:
    if isinstance(value, dict):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in FORBIDDEN_KEYS:
                return f"{path}.{key}" if path else str(key)
            found = _content_path(
                item, f"{path}.{key}" if path else str(key))
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            found = _content_path(item, f"{path}[{index}]")
            if found:
                return found
    return ""


def _entry_errors(entry, index: int) -> list[str]:
    where = f"entry {index}"
    if not isinstance(entry, dict):
        return [f"{where}: must be a JSON object"]

    errors: list[str] = []
    content = _content_path(entry)
    if content:
        errors.append(f"{where}: forbidden content key: {content}")

    missing = sorted(REQUIRED_KEYS - set(entry))
    if missing:
        errors.append(f"{where}: missing key(s): {', '.join(missing)}")
    extra = sorted(
        (key for key in entry if key not in REQUIRED_KEYS), key=str)
    if extra:
        errors.append(
            f"{where}: unexpected key(s): {', '.join(map(str, extra))}")

    if "version" in entry:
        version = entry["version"]
        if (isinstance(version, bool) or not isinstance(version, int)
                or version != VERSION):
            errors.append(f"{where}: version must be {VERSION}")

    if "at" in entry:
        try:
            _parse_timestamp(entry["at"], f"{where}: at")
        except ValueError as exc:
            errors.append(str(exc))

    if "source_id" in entry:
        value = entry["source_id"]
        if not isinstance(value, str) or not _SOURCE_ID_RE.match(value):
            errors.append(
                f"{where}: source_id must be {SOURCE_ID_LENGTH} lowercase "
                "hex characters")

    valid_counts = True
    counts: dict[str, int] = {}
    for key in COUNT_KEYS:
        if key not in entry:
            valid_counts = False
            continue
        try:
            counts[key] = _count(entry[key], key, where)
        except ValueError as exc:
            errors.append(str(exc))
            valid_counts = False

    if valid_counts:
        total = counts["supported"] + counts["weak"] + counts["unsupported"]
        claims = None
        if "claims" in entry:
            try:
                claims = _count(entry["claims"], "claims", where)
            except ValueError as exc:
                errors.append(str(exc))
        if claims is not None:
            if claims < 1:
                errors.append(f"{where}: 'claims' must be at least 1")
            elif claims != total:
                errors.append(
                    f"{where}: 'claims' does not equal supported + weak "
                    f"+ unsupported ({claims} != {total})")
            if counts["conflicts"] > claims:
                errors.append(
                    f"{where}: 'conflicts' cannot exceed claims")
    return errors


def add_observation(
    ledger: list[dict],
    source: str,
    *,
    supported: int,
    weak: int,
    unsupported: int,
    conflicts: int = 0,
    now=None,
) -> dict:
    """Append one content-free observation and return it."""
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    counts = {
        "supported": _count(supported, "supported", "observation"),
        "weak": _count(weak, "weak", "observation"),
        "unsupported": _count(unsupported, "unsupported", "observation"),
        "conflicts": _count(conflicts, "conflicts", "observation"),
    }
    claims = counts["supported"] + counts["weak"] + counts["unsupported"]
    if claims < 1:
        raise ValueError("an observation must contain at least one claim")
    if counts["conflicts"] > claims:
        raise ValueError("conflicts cannot exceed claims")

    identifier = source_id(source)
    stamp = datetime.now(timezone.utc) if now is None else now
    entry = {
        "version": VERSION,
        "at": _format_timestamp(stamp),
        "source_id": identifier,
        "supported": counts["supported"],
        "weak": counts["weak"],
        "unsupported": counts["unsupported"],
        "claims": claims,
        "conflicts": counts["conflicts"],
    }
    ledger.append(entry)
    return entry


def _support_ratio(supported: int, weak: int, claims: int) -> float:
    if claims <= 0:
        return 0.0
    return (supported + 0.5 * weak) / claims


def _status(ratio: float) -> str:
    if ratio >= STATUS_RELIABLE:
        return "reliable"
    if ratio >= STATUS_MIXED:
        return "mixed"
    return "weak"


def summarize(ledger: list[dict], min_observations: int = 1) -> dict:
    """Group a ledger by source and return deterministic totals."""
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    if (isinstance(min_observations, bool)
            or not isinstance(min_observations, int)
            or min_observations < 0):
        raise ValueError("min_observations must be a non-negative integer")

    errors: list[str] = []
    for index, entry in enumerate(ledger):
        errors.extend(_entry_errors(entry, index))
    if errors:
        raise ValueError("invalid ledger: " + "; ".join(errors))

    grouped: dict[str, dict] = {}
    for entry in ledger:
        identifier = entry["source_id"]
        row = grouped.setdefault(identifier, {
            "source_id": identifier,
            "observations": 0,
            "supported": 0,
            "weak": 0,
            "unsupported": 0,
            "claims": 0,
            "conflicts": 0,
        })
        row["observations"] += 1
        for key in ("supported", "weak", "unsupported", "claims", "conflicts"):
            row[key] += entry[key]

    sources = []
    for identifier in sorted(grouped):
        row = grouped[identifier]
        if row["observations"] < min_observations:
            continue
        ratio = _support_ratio(row["supported"], row["weak"], row["claims"])
        row["support_ratio"] = round(ratio, 4)
        row["status"] = _status(ratio)
        sources.append(row)

    totals = {
        "sources": len(sources),
        "observations": sum(row["observations"] for row in sources),
        "supported": sum(row["supported"] for row in sources),
        "weak": sum(row["weak"] for row in sources),
        "unsupported": sum(row["unsupported"] for row in sources),
        "claims": sum(row["claims"] for row in sources),
        "conflicts": sum(row["conflicts"] for row in sources),
    }
    total_ratio = _support_ratio(
        totals["supported"], totals["weak"], totals["claims"])
    totals["support_ratio"] = round(total_ratio, 4)
    totals["status"] = _status(total_ratio)

    return {
        "sources": sources,
        "totals": totals,
        "observations": totals["observations"],
        "claims": totals["claims"],
        "conflicts": totals["conflicts"],
        "support_ratio": totals["support_ratio"],
        "status": totals["status"],
        "min_observations": min_observations,
    }


def verify(ledger: list[dict]) -> dict:
    """Validate every entry without mutating the ledger."""
    if not isinstance(ledger, list):
        return {
            "ok": False,
            "entries": 0,
            "errors": ["ledger must be a list"],
        }
    errors: list[str] = []
    for index, entry in enumerate(ledger):
        errors.extend(_entry_errors(entry, index))
    return {
        "ok": not errors,
        "entries": len(ledger),
        "errors": errors,
    }


def _reject_duplicate_pairs(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"duplicate key: {key}")
        obj[key] = value
    return obj


def load_jsonl(path) -> list[dict]:
    """Read a UTF-8 JSONL ledger; a missing file is an empty ledger."""
    target = Path(path).expanduser()
    if not target.exists():
        return []
    ledger = []
    for line_no, line in enumerate(
            target.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            entry = json.loads(line, object_pairs_hook=_reject_duplicate_pairs)
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(f"{target}:{line_no}: invalid JSON: {exc}") from exc
        if not isinstance(entry, dict):
            raise ValueError(f"{target}:{line_no}: entry must be a JSON object")
        ledger.append(entry)
    return ledger


def write_jsonl(path, ledger: list[dict]) -> None:
    """Atomically write a UTF-8 JSONL ledger, one object per line."""
    if not isinstance(ledger, list):
        raise ValueError("ledger must be a list")
    lines = []
    for index, entry in enumerate(ledger):
        if not isinstance(entry, dict):
            raise ValueError(f"entry {index}: must be a JSON object")
        lines.append(json.dumps(
            entry, ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    text = "\n".join(lines) + ("\n" if lines else "")

    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent))
    temp = Path(temp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
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
            if temp.exists():
                temp.unlink()
        except OSError:
            pass


def _render_summary(summary: dict) -> str:
    totals = summary["totals"]
    lines = [
        f"Source reliability — {totals['sources']} source(s), "
        f"{totals['observations']} observation(s), "
        f"{totals['claims']} claim(s), {totals['conflicts']} conflict(s), "
        f"support_ratio={totals['support_ratio']:.4f} "
        f"status={totals['status']}",
    ]
    for row in summary["sources"]:
        lines.append(
            f"- {row['source_id']} observations={row['observations']} "
            f"claims={row['claims']} conflicts={row['conflicts']} "
            f"support_ratio={row['support_ratio']:.4f} "
            f"status={row['status']}")
    return "\n".join(lines)


def _render_verify(result: dict) -> str:
    if result["ok"]:
        return f"ok: {result['entries']} entr" \
               f"{'y' if result['entries'] == 1 else 'ies'} verified"
    lines = [f"invalid: {len(result['errors'])} error(s)"]
    lines.extend(f"- {error}" for error in result["errors"])
    return "\n".join(lines)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--file", default="data/source_reliability.jsonl",
        help="JSONL ledger path")
    parser.add_argument(
        "--add-source", default=None,
        help="append one observation for this source label")
    parser.add_argument("--supported", type=int, default=0)
    parser.add_argument("--weak", type=int, default=0)
    parser.add_argument("--unsupported", type=int, default=0)
    parser.add_argument("--conflicts", type=int, default=0)
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args(argv)

    do_add = args.add_source is not None
    do_verify = args.verify
    do_summary = args.summary or not (do_add or do_verify)

    payload: dict = {}
    try:
        ledger = load_jsonl(args.file)
        if do_add:
            entry = add_observation(
                ledger, args.add_source,
                supported=args.supported,
                weak=args.weak,
                unsupported=args.unsupported,
                conflicts=args.conflicts)
            write_jsonl(args.file, ledger)
            payload["added"] = entry
        if do_verify:
            result = verify(ledger)
            payload["verify"] = result
        if do_summary:
            payload["summary"] = summarize(ledger)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        if "added" in payload:
            print(f"added {payload['added']['source_id']}")
        if "verify" in payload:
            print(_render_verify(payload["verify"]))
        if "summary" in payload:
            print(_render_summary(payload["summary"]))
    if do_verify and not payload["verify"]["ok"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
