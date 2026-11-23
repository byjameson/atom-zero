"""Tests for local, deterministic proactive conversation."""
from datetime import datetime, timedelta
from types import SimpleNamespace

import pytest

from atomzero import proactive

NOW = datetime(2026, 11, 23, 12, 0, 0)


class FakeGaps:
    def __init__(self, items):
        self.items = [dict(item) for item in items]

    def list(self):
        return [dict(item) for item in self.items]


def make_session(tasks=None, gaps=None, data=None):
    memory = dict(data or {})
    memory.setdefault("tasks", list(tasks or []))
    session = SimpleNamespace(human=SimpleNamespace(data=memory))
    if gaps is not None:
        session.gap_store = gaps
    return session


def enable(session, **overrides):
    settings = {
        "enabled": True,
        "min_interval_minutes": 0,
        "max_per_day": 5,
        "quiet_start": "00:00",
        "quiet_end": "00:00",
    }
    settings.update(overrides)
    proactive.configure(session, **settings)


def overdue_task(text="Send the report", task_id=1, due="2026-11-20"):
    return {"id": task_id, "text": text, "status": "open", "due": due}


def test_default_config_and_disabled_gate():
    assert proactive.DEFAULT_CONFIG == {
        "enabled": False,
        "min_interval_minutes": 180,
        "max_per_day": 3,
        "quiet_start": "22:00",
        "quiet_end": "08:00",
    }
    session = make_session(tasks=[overdue_task()])
    decision = proactive.should_initiate(session, now=NOW)
    assert decision["due"] is False
    assert decision["blocked_by"] == "disabled"
    assert isinstance(decision["next_check_seconds"], int)
    assert decision["next_check_seconds"] > 0


def test_overdue_task_is_highest_priority():
    session = make_session(tasks=[
        overdue_task(),
        {"id": 2, "text": "Due today", "status": "open",
         "due": "2026-11-23"},
    ])
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["due"] is True
    assert candidate["reason"] == proactive.REASON_OVERDUE
    assert "overdue" in candidate["message"]
    assert "Send the report" in candidate["message"]
    assert candidate["actions"] == ["list tasks", "brief me"]
    assert candidate["priority"] >= 90
    assert candidate["proactive_id"].startswith("p_")
    assert len(candidate["message"]) <= proactive.MAX_MESSAGE_CHARS


def test_task_due_today():
    session = make_session(tasks=[
        {"id": 4, "text": "Call the dentist", "status": "open",
         "due": "2026-11-23T09:30:00"},
    ])
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["due"] is True
    assert candidate["reason"] == proactive.REASON_DUE_TODAY
    assert "today" in candidate["message"].lower()
    assert "Call the dentist" in candidate["message"]


def test_task_due_within_three_days():
    session = make_session(tasks=[
        {"id": 5, "text": "Prepare slides", "status": "open",
         "due": "2026-11-26"},
    ])
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["reason"] == proactive.REASON_DUE_SOON
    assert "within 3 days" in candidate["message"]
    assert "Prepare slides" in candidate["message"]


def test_pending_knowledge_gap():
    gaps = FakeGaps([
        {"id": "g0001", "query": "What is a vector database?",
         "status": "pending"},
        {"id": "g0002", "query": "Already proposed?", "status": "proposed"},
    ])
    session = make_session(gaps=gaps)
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["due"] is True
    assert candidate["reason"] == proactive.REASON_GAPS
    assert "knowledge gap" in candidate["message"]
    assert "vector database" in candidate["message"]
    assert candidate["actions"] == ["list gaps", "brief me"]


def test_fallback_check_in_only_when_enabled():
    session = make_session()
    assert proactive.build_candidate(session, now=NOW)["due"] is False

    enable(session)
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["due"] is True
    assert candidate["reason"] == proactive.REASON_CHECK_IN
    assert candidate["priority"] < 50

    decision = proactive.should_initiate(session, now=NOW)
    assert decision["due"] is True
    assert decision["blocked_by"] == ""


def test_recent_sources_without_recent_brief():
    session = make_session()
    session.human.data["sources"] = [{
        "source": "notes.txt",
        "title": "Launch notes",
        "created": (NOW - timedelta(days=1)).isoformat(),
    }]
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["reason"] == proactive.REASON_SOURCES
    assert "Launch notes" in candidate["message"]

    session.human.data["last_brief_at"] = (
        NOW - timedelta(hours=2)).isoformat()
    assert proactive.build_candidate(session, now=NOW)["due"] is False


def test_cooldown_after_ack():
    session = make_session(tasks=[overdue_task()])
    enable(session, min_interval_minutes=180)
    candidate = proactive.build_candidate(session, now=NOW)
    proactive.ack(session, candidate["proactive_id"], now=NOW)

    decision = proactive.should_initiate(
        session, now=NOW + timedelta(minutes=10))
    assert decision["due"] is False
    assert decision["blocked_by"] == "cooldown"
    assert decision["next_check_seconds"] > 0

    later = proactive.should_initiate(
        session, now=NOW + timedelta(minutes=181))
    assert later["due"] is False
    assert later["blocked_by"] == "duplicate"


def test_quiet_hours_wrap_around_midnight():
    session = make_session(tasks=[overdue_task()])
    enable(session, quiet_start="22:00", quiet_end="08:00")

    night = proactive.should_initiate(
        session, now=datetime(2026, 11, 23, 23, 30))
    assert night["due"] is False
    assert night["blocked_by"] == "quiet_hours"
    assert night["next_check_seconds"] == 8 * 3600 + 1800

    morning = proactive.should_initiate(
        session, now=datetime(2026, 11, 23, 7, 0))
    assert morning["blocked_by"] == "quiet_hours"
    assert morning["next_check_seconds"] == 3600

    daytime = proactive.should_initiate(
        session, now=datetime(2026, 11, 23, 12, 0))
    assert daytime["due"] is True
    assert daytime["blocked_by"] == ""


def test_daily_limit():
    session = make_session(tasks=[overdue_task()])
    enable(session, max_per_day=1)
    candidate = proactive.build_candidate(session, now=NOW)
    proactive.ack(session, candidate["proactive_id"], now=NOW)

    decision = proactive.should_initiate(
        session, now=NOW + timedelta(minutes=5))
    assert decision["due"] is False
    assert decision["blocked_by"] == "daily_limit"
    assert decision["next_check_seconds"] > 0


def test_duplicate_reason_same_day():
    session = make_session(tasks=[overdue_task()])
    enable(session, max_per_day=3)
    candidate = proactive.build_candidate(session, now=NOW)
    state = proactive.ack(session, candidate["proactive_id"], now=NOW)
    assert state["last_reason"] == proactive.REASON_OVERDUE

    decision = proactive.should_initiate(
        session, now=NOW + timedelta(minutes=5))
    assert decision["due"] is False
    assert decision["blocked_by"] == "duplicate"

    # A different reason is not treated as a duplicate.
    session.human.data["tasks"] = []
    session.gap_store = FakeGaps([
        {"id": "g1", "query": "What changed?", "status": "pending"}])
    other = proactive.should_initiate(
        session, now=NOW + timedelta(minutes=6))
    assert other["due"] is True
    assert other["reason"] == proactive.REASON_GAPS


def test_ack_history_is_bounded():
    session = make_session(tasks=[overdue_task()])
    enable(session)
    for index in range(25):
        proactive.ack(
            session,
            proactive_id=f"p_manual_{index}",
            action="shown",
            now=NOW + timedelta(minutes=index),
        )
    state = proactive.proactive_state(session)
    assert state["count"] == 25
    assert state["day"] == "2026-11-23"
    assert state["date"] == "2026-11-23"
    assert state["last_at"] == "2026-11-23T12:24:00"
    assert state["last_reason"] == proactive.REASON_OVERDUE
    assert len(state["history"]) == proactive.HISTORY_LIMIT
    assert state["history"][0]["proactive_id"] == "p_manual_5"
    assert state["history"][-1]["proactive_id"] == "p_manual_24"
    assert all(row["action"] == "shown" for row in state["history"])
    assert all(row["at"] for row in state["history"])


def test_next_tick_records_ack_and_dedupes():
    session = make_session(tasks=[overdue_task()])
    enable(session, max_per_day=2)
    tick = proactive.next_tick(session, now=NOW, ack=True)
    assert tick["due"] is True
    assert tick["acked"] is True
    assert tick["blocked_by"] == ""
    assert tick["message"]
    assert tick["actions"] == ["list tasks", "brief me"]
    state = proactive.proactive_state(session)
    assert state["count"] == 1
    assert len(state["history"]) == 1

    second = proactive.next_tick(
        session, now=NOW + timedelta(minutes=1))
    assert second["due"] is False
    assert second["acked"] is False
    assert second["blocked_by"] == "duplicate"


def test_malformed_state_is_defensive():
    session = SimpleNamespace(human=SimpleNamespace(data={
        "proactive": "not-a-dict",
        "tasks": "not-a-list",
    }))
    state = proactive.proactive_state(session)
    assert state == {}
    assert session.human.data["proactive"] is state

    state.update({
        "last_at": "not-a-time",
        "count": "many",
        "day": 17,
        "history": "not-a-list",
        "last_reason": 123,
    })
    session.human.data["tasks"] = [
        None,
        "bad task",
        {"id": 1, "text": "Bad due", "status": "open", "due": "soon"},
        {"id": 2, "text": "Future?", "status": "open", "due": "9999"},
    ]
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["due"] is False
    assert proactive.should_initiate(session, now=NOW)["blocked_by"] == "disabled"

    result = proactive.ack(session, proactive_id="", now=NOW)
    assert isinstance(result["history"], list)
    assert len(result["history"]) == 1
    assert result["count"] == 1
    assert result["day"] == "2026-11-23"


def test_plain_dict_session():
    session = {"tasks": [overdue_task()]}
    state = proactive.proactive_state(session)
    assert session["proactive"] is state
    proactive.configure(
        session, enabled=True, min_interval_minutes=0, max_per_day=2,
        quiet_start="00:00", quiet_end="00:00")
    tick = proactive.next_tick(session, now=NOW, ack=True)
    assert tick["due"] is True
    assert tick["acked"] is True
    assert session["proactive"]["count"] == 1
    assert session["proactive"]["last_reason"] == proactive.REASON_OVERDUE


def test_configure_validates_and_normalizes():
    session = make_session()
    with pytest.raises(ValueError):
        proactive.configure(session, enabled="yes")
    with pytest.raises(ValueError):
        proactive.configure(session, min_interval_minutes=-1)
    with pytest.raises(ValueError):
        proactive.configure(session, quiet_start="25:00")
    with pytest.raises(ValueError):
        proactive.configure(session, typo=True)
    with pytest.raises(ValueError):
        proactive.configure(session, enabled=True, max_per_day=-5)
    assert proactive.proactive_state(session).get("enabled") is None

    proactive.configure(
        session, quiet_start="8:00", quiet_end="7:30",
        min_interval_minutes="15", max_per_day=2)
    state = proactive.proactive_state(session)
    assert state["quiet_start"] == "08:00"
    assert state["quiet_end"] == "07:30"
    assert state["min_interval_minutes"] == 15
    assert state["max_per_day"] == 2


def test_long_message_is_bounded():
    session = make_session(tasks=[overdue_task(text="x" * 500)])
    candidate = proactive.build_candidate(session, now=NOW)
    assert candidate["due"] is True
    assert len(candidate["message"]) <= proactive.MAX_MESSAGE_CHARS
    assert candidate["message"].endswith("?")
    assert "sent" not in candidate["message"].lower()
