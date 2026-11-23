"""Proactive, AI-initiated conversation from local state.

The assistant can start a short, useful conversation instead of only
answering.  Everything in this module is deterministic, local and
provider-agnostic: it reads tasks, pending knowledge gaps and recently
indexed sources, then decides whether a local message is worth showing.

This module never sends anything outside the local application.  Callers
should only render a message when ``should_initiate``/``next_tick`` reports
``due=True``; the returned text never claims that a message was delivered
elsewhere.

Typical use::

    config = {"enabled": True, "quiet_start": "22:00", "quiet_end": "08:00"}
    configure(session, **config)
    tick = next_tick(session, ack=True)
    if tick["due"]:
        show_locally(tick["message"], tick["actions"])
"""
from __future__ import annotations

import copy
import hashlib
import re
from datetime import date, datetime, time, timedelta

__all__ = [
    "DEFAULT_CONFIG",
    "ack",
    "build_candidate",
    "configure",
    "next_tick",
    "proactive_state",
    "should_initiate",
]

DEFAULT_CONFIG = {
    "enabled": False,
    "min_interval_minutes": 180,
    "max_per_day": 3,
    "quiet_start": "22:00",
    "quiet_end": "08:00",
}

# Machine-readable reasons, highest priority first.
REASON_OVERDUE = "overdue_tasks"
REASON_DUE_TODAY = "tasks_due_today"
REASON_DUE_SOON = "tasks_due_soon"
REASON_GAPS = "pending_gaps"
REASON_SOURCES = "recent_sources"
REASON_CHECK_IN = "check_in"

MAX_MESSAGE_CHARS = 240
HISTORY_LIMIT = 20
RECENT_SOURCE_DAYS = 7
RECENT_BRIEF_HOURS = 24
DUE_SOON_DAYS = 3

_DONE_STATUSES = {
    "done", "completed", "complete", "closed", "cancelled", "canceled",
    "archived",
}
_GAP_OPEN_STATUSES = {"pending", "open", "new"}
_SOURCE_TIME_KEYS = ("created", "added", "ts", "timestamp", "created_at")
_BRIEF_TIME_KEYS = ("last_brief_at", "brief_at", "briefed_at")
_TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")


class ProactiveError(ValueError):
    """Raised when proactive configuration is invalid."""


def _as_local(value: datetime) -> datetime:
    if value.tzinfo is not None:
        value = value.astimezone().replace(tzinfo=None)
    return value.replace(microsecond=0)


def _parse_datetime(value) -> datetime | None:
    """Parse an ISO string, date, epoch number or datetime as local time."""
    if isinstance(value, datetime):
        return _as_local(value)
    if isinstance(value, date):
        return datetime.combine(value, time.min)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    candidate = text[:-1] + "+00:00" if text.endswith(("Z", "z")) else text
    try:
        return _as_local(datetime.fromisoformat(candidate))
    except ValueError:
        pass
    for pattern in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, pattern)
        except ValueError:
            continue
    # Large numeric strings are epoch seconds, not years.
    if re.fullmatch(r"-?\d+(?:\.\d+)?", text) and len(
            text.split(".", 1)[0].lstrip("-")) >= 9:
        try:
            return datetime.fromtimestamp(float(text))
        except (OverflowError, OSError, ValueError):
            return None
    return None


def _parse_due_date(value) -> date | None:
    if isinstance(value, datetime):
        return _as_local(value).date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        parsed = _parse_datetime(text)
        return parsed.date() if parsed is not None else None


def _now_local(now=None) -> datetime:
    parsed = _parse_datetime(now)
    if parsed is not None:
        return parsed
    return datetime.now().replace(microsecond=0)


def _stamp(moment: datetime) -> str:
    return moment.strftime("%Y-%m-%dT%H:%M:%S")


def _human_data(session) -> dict:
    if isinstance(session, dict):
        return session
    human = getattr(session, "human", None)
    data = getattr(human, "data", None) if human is not None else None
    if isinstance(data, dict):
        return data
    data = getattr(session, "data", None)
    return data if isinstance(data, dict) else {}


def _lookup(session, key):
    if isinstance(session, dict):
        return session.get(key)
    return getattr(session, key, None)


def proactive_state(session) -> dict:
    """Return (and create) the per-session proactive state dict.

    A plain-dict session stores the state at ``session["proactive"]``.  An
    object session stores it at ``session.human.data["proactive"]``.  A
    malformed container is repaired to an empty dict.  If no writable data
    store exists, a fresh detached dict is returned so callers never crash.
    """
    if isinstance(session, dict):
        state = session.get("proactive")
        if not isinstance(state, dict):
            state = {}
            session["proactive"] = state
        return state

    human = getattr(session, "human", None)
    data = getattr(human, "data", None) if human is not None else None
    if not isinstance(data, dict):
        data = getattr(session, "data", None)
    if not isinstance(data, dict) and human is not None:
        try:
            human.data = {}
            data = human.data
        except (AttributeError, TypeError):
            data = None
    if not isinstance(data, dict):
        return {}
    state = data.get("proactive")
    if not isinstance(state, dict):
        state = {}
        data["proactive"] = state
    return state


def _save(session) -> None:
    if isinstance(session, dict):
        return
    human = getattr(session, "human", None)
    save = getattr(human, "save", None)
    if callable(save):
        try:
            save()
        except (OSError, TypeError, ValueError):
            pass


def _coerce_bool(value, key: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    raise ProactiveError(f"{key} must be a boolean")


def _coerce_int(value, key: str, minimum: int, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str, float)):
        raise ProactiveError(f"{key} must be an integer")
    if isinstance(value, float) and not value.is_integer():
        raise ProactiveError(f"{key} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise ProactiveError(f"{key} must be an integer") from exc
    if number < minimum or number > maximum:
        raise ProactiveError(
            f"{key} must be between {minimum} and {maximum}")
    return number


def _coerce_hhmm(value, key: str) -> str:
    if not isinstance(value, str):
        raise ProactiveError(f"{key} must be a time string like '22:00'")
    match = _TIME_RE.fullmatch(value.strip())
    if not match:
        raise ProactiveError(f"{key} must be a time string like '22:00'")
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        raise ProactiveError(f"{key} must be a valid local time")
    return f"{hour:02d}:{minute:02d}"


def _coerce_config_value(key: str, value):
    if key == "enabled":
        return _coerce_bool(value, key)
    if key == "min_interval_minutes":
        return _coerce_int(value, key, 0, 525600)
    if key == "max_per_day":
        return _coerce_int(value, key, 0, 1000)
    if key in ("quiet_start", "quiet_end"):
        return _coerce_hhmm(value, key)
    raise ProactiveError(f"unknown proactive config key: {key}")


def _resolve_config(state: dict, config=None) -> dict:
    """Merge defaults, persisted state and an explicit config override."""
    resolved = dict(DEFAULT_CONFIG)
    sources = []
    nested = state.get("config")
    if isinstance(nested, dict):
        sources.append(nested)
    sources.append(state)
    for source in sources:
        for key in DEFAULT_CONFIG:
            if key not in source:
                continue
            try:
                resolved[key] = _coerce_config_value(key, source[key])
            except (ProactiveError, TypeError, ValueError):
                # A malformed persisted value falls back to the default.
                continue
    if config is not None:
        if not isinstance(config, dict):
            raise ProactiveError("config must be a dict")
        for key, value in config.items():
            if key not in DEFAULT_CONFIG:
                raise ProactiveError(f"unknown proactive config key: {key}")
            resolved[key] = _coerce_config_value(key, value)
    return resolved


def configure(session, **updates) -> dict:
    """Validate and persist proactive settings; returns the state copy."""
    state = proactive_state(session)
    cleaned = {}
    for key, value in updates.items():
        if key not in DEFAULT_CONFIG:
            raise ProactiveError(f"unknown proactive config key: {key}")
        cleaned[key] = _coerce_config_value(key, value)
    if cleaned:
        state.update(cleaned)
        _save(session)
    return copy.deepcopy(state)


def _safe_int(value, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _history(state: dict) -> list[dict]:
    rows = state.get("history")
    if not isinstance(rows, list):
        return []
    return [dict(row) for row in rows if isinstance(row, dict)]


def _open_tasks(data: dict) -> list[dict]:
    rows = data.get("tasks")
    if not isinstance(rows, list):
        return []
    tasks = []
    for task in rows:
        if not isinstance(task, dict):
            continue
        status = str(task.get("status", "open") or "open").strip().lower()
        if status in _DONE_STATUSES:
            continue
        tasks.append(task)
    return tasks


def _task_name(task: dict) -> str:
    for key in ("text", "title", "name", "summary"):
        value = task.get(key)
        if isinstance(value, str) and value.strip():
            return value
    task_id = task.get("id")
    return f"task {task_id}" if task_id not in (None, "") else "untitled task"


def _short(value, limit: int = 60) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = text.replace('"', "'")
    if len(text) > limit:
        text = text[:max(1, limit - 1)].rstrip() + "…"
    return text


def _bounded_message(text: str) -> str:
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if len(text) > MAX_MESSAGE_CHARS:
        text = text[:MAX_MESSAGE_CHARS - 1].rstrip() + "…"
    return text


def _plural(count: int, singular: str, plural: str | None = None) -> str:
    return singular if count == 1 else (plural or singular + "s")


def _task_sort_key(entry: tuple[date, dict]):
    due, task = entry
    return (due, str(task.get("id", "")), _task_name(task))


def _gap_rows(session) -> list[dict]:
    store = _lookup(session, "gap_store")
    if store is None:
        store = _lookup(session, "gaps")
    if store is None:
        data = _human_data(session)
        store = data.get("gap_store")
        if store is None:
            store = data.get("gaps")
    if store is None:
        return []

    rows = None
    if isinstance(store, dict):
        if isinstance(store.get("items"), list):
            rows = store["items"]
        elif isinstance(store.get("gaps"), list):
            rows = store["gaps"]
        elif any(key in store for key in ("id", "query", "question")):
            rows = [store]
        elif all(isinstance(value, dict) for value in store.values()):
            rows = list(store.values())
    elif isinstance(store, (list, tuple)):
        rows = list(store)
    else:
        for name in ("list", "pending"):
            method = getattr(store, name, None)
            if callable(method):
                try:
                    rows = method()
                except (OSError, TypeError, ValueError):
                    rows = None
                if isinstance(rows, (list, tuple)):
                    break
        if not isinstance(rows, (list, tuple)):
            rows = getattr(store, "items", None)
    if not isinstance(rows, (list, tuple)):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _pending_gaps(session) -> list[dict]:
    pending = []
    for gap in _gap_rows(session):
        status = str(gap.get("status", "pending") or "pending").strip().lower()
        if status in _GAP_OPEN_STATUSES:
            pending.append(gap)
    return pending


def _gap_name(gap: dict) -> str:
    for key in ("query", "question", "text", "title"):
        value = gap.get(key)
        if isinstance(value, str) and value.strip():
            return value
    return "an untitled gap"


def _source_rows(value) -> list[dict]:
    if isinstance(value, list):
        return [dict(row) for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        if any(key in value for key in ("source", "title", "created")):
            return [dict(value)]
        rows = []
        for key, row in value.items():
            if isinstance(row, dict):
                item = dict(row)
                item.setdefault("source", str(key))
                rows.append(item)
        return rows
    return []


def _all_sources(session) -> list[dict]:
    rows = []
    for key in ("sources", "documents"):
        value = _lookup(session, key)
        if value is None:
            value = _human_data(session).get(key)
        rows.extend(_source_rows(value))
    docs = _lookup(session, "docs")
    source_method = getattr(docs, "sources", None) if docs is not None else None
    if callable(source_method):
        try:
            rows.extend(_source_rows(source_method()))
        except (OSError, TypeError, ValueError):
            pass
    # De-duplicate by source/title while preserving order.
    seen = set()
    unique = []
    for row in rows:
        key = (str(row.get("source", "")), str(row.get("title", "")))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    return unique


def _source_stamp(item: dict) -> datetime | None:
    for key in _SOURCE_TIME_KEYS:
        if key in item:
            parsed = _parse_datetime(item.get(key))
            if parsed is not None:
                return parsed
    return None


def _recent_sources(session, moment: datetime) -> list[dict]:
    cutoff = moment - timedelta(days=RECENT_SOURCE_DAYS)
    recent = []
    for item in _all_sources(session):
        raw = next((item.get(key) for key in _SOURCE_TIME_KEYS
                    if item.get(key)), None)
        stamp = _source_stamp(item)
        if stamp is not None:
            if stamp >= cutoff:
                recent.append(item)
            continue
        # Sources without a timestamp are treated as recent, matching the
        # conservative behaviour used by the local weekly review.
        if not raw:
            recent.append(item)
    return recent


def _brief_stamps(session) -> list[datetime]:
    state = proactive_state(session)
    values = [state.get("last_brief_at")]
    data = _human_data(session)
    for key in _BRIEF_TIME_KEYS:
        values.append(data.get(key))
        if not isinstance(session, dict):
            values.append(getattr(session, key, None))
    for key in ("briefs", "brief_history"):
        rows = data.get(key)
        if not isinstance(rows, list):
            continue
        for row in rows:
            if not isinstance(row, dict):
                continue
            for field in ("at", "created", "ts", "timestamp", "shown_at"):
                values.append(row.get(field))
    parsed = []
    for value in values:
        stamp = _parse_datetime(value)
        if stamp is not None:
            parsed.append(stamp)
    return parsed


def _recent_brief(session, moment: datetime) -> bool:
    cutoff = moment - timedelta(hours=RECENT_BRIEF_HOURS)
    return any(stamp >= cutoff for stamp in _brief_stamps(session))


def _make_candidate(reason: str, message: str, actions: list[str],
                    priority: int, next_check_seconds: int,
                    key: str, moment: datetime) -> dict:
    raw = f"{reason}|{key}|{moment.date().isoformat()}"
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return {
        "due": True,
        "reason": reason,
        "message": _bounded_message(message),
        "actions": list(actions),
        "priority": int(priority),
        "next_check_seconds": max(60, int(next_check_seconds)),
        "proactive_id": f"p_{digest}",
    }


def _empty_candidate(next_check_seconds: int = 21600) -> dict:
    return {
        "due": False,
        "reason": "",
        "message": "",
        "actions": [],
        "priority": 0,
        "next_check_seconds": max(60, int(next_check_seconds)),
        "proactive_id": "",
    }


def build_candidate(session, now=None, config=None) -> dict:
    """Build the highest-priority local conversation candidate.

    The candidate is independent of the enabled flag for real signals; the
    fallback check-in is only produced when proactive mode is enabled.
    """
    moment = _now_local(now)
    state = proactive_state(session)
    cfg = _resolve_config(state, config)
    data = _human_data(session)
    today = moment.date()
    soon_end = today + timedelta(days=DUE_SOON_DAYS)

    overdue = []
    due_today = []
    due_soon = []
    for task in _open_tasks(data):
        due = _parse_due_date(task.get("due"))
        if due is None:
            continue
        if due < today:
            overdue.append((due, task))
        elif due == today:
            due_today.append((due, task))
        elif due <= soon_end:
            due_soon.append((due, task))
    overdue.sort(key=_task_sort_key)
    due_today.sort(key=_task_sort_key)
    due_soon.sort(key=_task_sort_key)

    if overdue:
        task = overdue[0][1]
        name = _short(_task_name(task))
        count = len(overdue)
        if count == 1:
            message = (
                f'Your local task "{name}" is overdue. '
                "Want to review it or reschedule it?")
        else:
            message = (
                f'You have {count} overdue tasks in your local list, '
                f'including "{name}". Want to review them?')
        return _make_candidate(
            REASON_OVERDUE, message, ["list tasks", "brief me"],
            priority=100, next_check_seconds=900,
            key=f"{count}:{_task_name(task)}:{overdue[0][0].isoformat()}",
            moment=moment)

    if due_today:
        task = due_today[0][1]
        name = _short(_task_name(task))
        count = len(due_today)
        if count == 1:
            message = (
                f'Your local task "{name}" is due today. '
                "Want to plan it?")
        else:
            message = (
                f'You have {count} tasks due today, including "{name}". '
                "Want a short plan?")
        return _make_candidate(
            REASON_DUE_TODAY, message, ["list tasks", "brief me"],
            priority=90, next_check_seconds=1800,
            key=f"{count}:{_task_name(task)}:{today.isoformat()}",
            moment=moment)

    if due_soon:
        task = due_soon[0][1]
        name = _short(_task_name(task))
        count = len(due_soon)
        message = (
            f'You have {count} {_plural(count, "task")} due within '
            f'{DUE_SOON_DAYS} days, including "{name}". Want to plan ahead?')
        return _make_candidate(
            REASON_DUE_SOON, message, ["list tasks", "brief me"],
            priority=70, next_check_seconds=21600,
            key=f"{count}:{_task_name(task)}:{due_soon[0][0].isoformat()}",
            moment=moment)

    gaps = _pending_gaps(session)
    if gaps:
        gap = gaps[0]
        name = _short(_gap_name(gap), limit=80)
        count = len(gaps)
        if count == 1:
            message = (
                f'You have a pending knowledge gap: "{name}". '
                "Want to review it?")
        else:
            message = (
                f'You have {count} pending knowledge gaps, including '
                f'"{name}". Want to review them?')
        return _make_candidate(
            REASON_GAPS, message, ["list gaps", "brief me"],
            priority=60, next_check_seconds=21600,
            key=f"{count}:{_gap_name(gap)}",
            moment=moment)

    recent_sources = _recent_sources(session, moment)
    if recent_sources and not _recent_brief(session, moment):
        count = len(recent_sources)
        first = recent_sources[0]
        title = _short(first.get("title") or first.get("source")
                       or "a local source", limit=60)
        message = (
            f'You added {count} recent local {_plural(count, "source")}, '
            f'including "{title}". Want a short brief that cites them?')
        return _make_candidate(
            REASON_SOURCES, message, ["brief me"],
            priority=50, next_check_seconds=21600,
            key=f"{count}:{first.get('source', '')}",
            moment=moment)

    if cfg["enabled"]:
        message = (
            "Just checking in. Nothing urgent is due; want a quick local "
            "recap of your tasks and notes?")
        return _make_candidate(
            REASON_CHECK_IN, message, ["brief me", "list tasks"],
            priority=10, next_check_seconds=21600,
            key="fallback", moment=moment)

    return _empty_candidate()


def _parse_clock(value: str) -> time | None:
    match = _TIME_RE.fullmatch(str(value or "").strip())
    if not match:
        return None
    hour = int(match.group(1))
    minute = int(match.group(2))
    if hour > 23 or minute > 59:
        return None
    return time(hour, minute)


def _in_quiet_hours(moment: datetime, quiet_start: str,
                    quiet_end: str) -> bool:
    start = _parse_clock(quiet_start)
    end = _parse_clock(quiet_end)
    if start is None or end is None or start == end:
        return False
    current = moment.time().replace(second=0, microsecond=0)
    if start < end:
        return start <= current < end
    # The quiet window wraps around midnight.
    return current >= start or current < end


def _quiet_end_seconds(moment: datetime, quiet_start: str,
                       quiet_end: str) -> int:
    start = _parse_clock(quiet_start)
    end = _parse_clock(quiet_end)
    if start is None or end is None:
        return 3600
    today = moment.date()
    if start < end:
        target = datetime.combine(today, end)
        if target <= moment:
            target += timedelta(days=1)
    else:
        target = datetime.combine(today, end)
        if moment.time() >= start:
            target += timedelta(days=1)
        elif target <= moment:
            target += timedelta(days=1)
    return max(60, int((target - moment).total_seconds()))


def _midnight_seconds(moment: datetime) -> int:
    target = datetime.combine(moment.date() + timedelta(days=1), time.min)
    return max(60, int((target - moment).total_seconds()))


def _count_today(state: dict, moment: datetime) -> int:
    today = moment.date().isoformat()
    day = state.get("day") or state.get("date")
    if isinstance(day, str) and day.strip()[:10] == today:
        return max(0, _safe_int(state.get("count"), 0))
    last = _parse_datetime(state.get("last_at"))
    if last is not None and last.date() == moment.date():
        return max(0, _safe_int(state.get("count"), 0))
    return 0


def _last_at(state: dict) -> datetime | None:
    return _parse_datetime(state.get("last_at"))


def _same_day_as_now(state: dict, moment: datetime) -> bool:
    today = moment.date().isoformat()
    for key in ("day", "date"):
        value = state.get(key)
        if isinstance(value, str) and value.strip()[:10] == today:
            return True
    last = _last_at(state)
    if last is not None:
        return last.date() == moment.date()
    for row in reversed(_history(state)):
        stamp = _parse_datetime(row.get("at"))
        if stamp is not None:
            return stamp.date() == moment.date()
    return False


def should_initiate(session, now=None, config=None) -> dict:
    """Decide whether a local proactive message may be shown now."""
    moment = _now_local(now)
    state = proactive_state(session)
    cfg = _resolve_config(state, config)
    candidate = build_candidate(session, now=moment, config=cfg)
    reason = candidate["reason"] if candidate["due"] else ""
    result = {
        "due": False,
        "reason": reason,
        "next_check_seconds": candidate["next_check_seconds"],
        "blocked_by": "",
    }

    if not cfg["enabled"]:
        result["blocked_by"] = "disabled"
        return result
    if not candidate["due"]:
        return result

    if _in_quiet_hours(moment, cfg["quiet_start"], cfg["quiet_end"]):
        result["blocked_by"] = "quiet_hours"
        result["next_check_seconds"] = _quiet_end_seconds(
            moment, cfg["quiet_start"], cfg["quiet_end"])
        return result

    last = _last_at(state)
    if last is not None and cfg["min_interval_minutes"] > 0:
        interval = cfg["min_interval_minutes"] * 60
        elapsed = (moment - last).total_seconds()
        if elapsed < interval:
            result["blocked_by"] = "cooldown"
            result["next_check_seconds"] = max(
                60, int(interval - elapsed))
            return result

    used = _count_today(state, moment)
    if used >= cfg["max_per_day"]:
        result["blocked_by"] = "daily_limit"
        result["next_check_seconds"] = _midnight_seconds(moment)
        return result

    last_reason = str(state.get("last_reason") or "")
    if (reason and last_reason == reason
            and _same_day_as_now(state, moment)):
        result["blocked_by"] = "duplicate"
        result["next_check_seconds"] = _midnight_seconds(moment)
        return result

    result["due"] = True
    result["next_check_seconds"] = candidate["next_check_seconds"]
    return result


def _reason_for_ack(session, state: dict, proactive_id: str,
                    candidate: dict) -> str:
    if not proactive_id or proactive_id == candidate.get("proactive_id"):
        return str(candidate.get("reason") or "")
    for row in reversed(_history(state)):
        if str(row.get("proactive_id", "")) == proactive_id:
            return str(row.get("reason", ""))
    return str(candidate.get("reason") or state.get("last_reason") or "")


def _record_ack(session, proactive_id="", action="shown", now=None) -> dict:
    moment = _now_local(now)
    state = proactive_state(session)
    candidate = build_candidate(session, now=moment)
    pid = str(proactive_id or candidate.get("proactive_id") or "")[:80]
    reason = _reason_for_ack(session, state, pid, candidate)[:80]
    stamp = _stamp(moment)
    today = moment.date().isoformat()

    used = _count_today(state, moment)
    state["count"] = used + 1
    state["day"] = today
    state["date"] = today
    state["last_at"] = stamp
    state["last_reason"] = reason
    state["last_proactive_id"] = pid

    history = _history(state)
    history.append({
        "proactive_id": pid,
        "reason": reason,
        "action": str(action or "shown")[:40],
        "at": stamp,
    })
    state["history"] = history[-HISTORY_LIMIT:]
    if str(action or "").strip().lower() == "brief":
        state["last_brief_at"] = stamp

    _save(session)
    return copy.deepcopy(state)


def ack(session, proactive_id="", action="shown", now=None) -> dict:
    """Record that a proactive message was shown/acted on locally."""
    return _record_ack(session, proactive_id=proactive_id,
                       action=action, now=now)


def next_tick(session, now=None, config=None, ack=False) -> dict:
    """Run one scheduler tick and optionally record the shown message."""
    moment = _now_local(now)
    state = proactive_state(session)
    cfg = _resolve_config(state, config)
    candidate = build_candidate(session, now=moment, config=cfg)
    decision = should_initiate(session, now=moment, config=cfg)
    result = {
        "due": decision["due"],
        "reason": decision["reason"],
        "message": candidate["message"],
        "actions": list(candidate["actions"]),
        "priority": candidate["priority"],
        "next_check_seconds": decision["next_check_seconds"],
        "proactive_id": candidate["proactive_id"],
        "blocked_by": decision["blocked_by"],
        "acked": False,
    }
    if ack and decision["due"]:
        _record_ack(session, proactive_id=candidate["proactive_id"],
                    action="shown", now=moment)
        result["acked"] = True
    return result
