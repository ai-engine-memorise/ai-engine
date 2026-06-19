from ai_engine.recsys.adapters.rudderstack import (
    normalize_content_id, normalize_event, normalize_events,
)
from ai_engine.recsys.contracts.enums import EndReason


def test_normalize_content_id_strips_prefix():
    assert normalize_content_id("content_1234") == "1234"
    assert normalize_content_id("841") == "841"
    assert normalize_content_id(None) is None


def test_content_and_candidates_as_bare_strings():
    """RudderStack/clients may send `content` (and context.candidates) as the bare
    content_id string instead of a {content_id:...} dict — must not 500."""
    raw = {
        "event": "CONTENT_VIEW_ENDED", "userId": "u1", "timestamp": "2026-06-19T10:00:00Z",
        "properties": {"content": "content_2669",
                       "details": {"reason": "next_button", "dwell_seconds": 12},
                       "context": {"candidates": ["101", "content_102"]}},
    }
    ev = normalize_event(raw)
    assert ev.content_id == "2669"
    assert ev.impressions == ["101", "102"]


def test_normalize_content_view_ended():
    raw = {
        "event": "CONTENT_VIEW_ENDED",
        "userId": "u1",
        "timestamp": "2026-06-10T10:00:00Z",
        "properties": {
            "content": {"content_id": "content_841", "content_type": "text_item"},
            "details": {"reason": "next_button"},
        },
    }
    ev = normalize_event(raw)
    assert ev is not None
    assert ev.user_id == "u1"
    assert ev.content_id == "841"
    assert ev.end_reason == EndReason.next_button


def test_normalize_pulls_impressions_from_candidates():
    raw = {
        "event": "CONTENT_VIEW_STARTED",
        "userId": "u1",
        "timestamp": "2026-06-10T10:00:00Z",
        "properties": {
            "content": {"content_id": "content_100"},
            "context": {"candidates": [
                {"content_id": "content_200"},
                {"content_id": "content_300"},
            ]},
        },
    }
    ev = normalize_event(raw)
    assert ev.content_id == "100"
    assert ev.impressions == ["200", "300"]


def test_normalize_event_rejects_missing_user_or_event():
    assert normalize_event({"event": "X"}) is None
    assert normalize_event({"userId": "u1"}) is None


def test_normalize_events_sorts_by_ts():
    raws = [
        {"event": "E", "userId": "u", "timestamp": "2026-06-10T10:00:02Z", "properties": {}},
        {"event": "E", "userId": "u", "timestamp": "2026-06-10T10:00:01Z", "properties": {}},
    ]
    evs = normalize_events(raws)
    assert [e.ts for e in evs] == sorted(e.ts for e in evs)
