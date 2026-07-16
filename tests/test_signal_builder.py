from datetime import datetime, timedelta

from ai_engine.recsys.contracts import RecConfig
from ai_engine.recsys.signals.signal_builder import build_user_signals, aggregate_views
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events

NOW = datetime(2026, 6, 10, 12, 0, 0)
CFG = RecConfig()
CONTENTS, VECTORS = make_contents_and_vectors()


def _build(events, now=NOW, demographics=None):
    return build_user_signals(
        user_id="u1", events=events, contents=CONTENTS, vectors=VECTORS,
        now=now, cfg=CFG, demographics=demographics,
    )


def test_deep_engagement_becomes_positive_with_tags_and_taste():
    events = (
        view_events("u1", "101", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
        + view_events("u1", "102", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    )
    sig = _build(events)
    assert "101" in sig.positives and "102" in sig.positives
    assert sig.tag_affinity.get("theme_what:forced labor", 0) > 0  # keys canonicalized lowercase
    # taste vector points along the Forced-Labor axis (index 0)
    assert sig.taste_vector is not None
    assert sig.taste_vector[0] == max(sig.taste_vector)


def test_quick_abandon_becomes_negative():
    events = view_events("u1", "201", dwell=1, reason="abandon", base_ts=NOW - timedelta(hours=1))
    sig = _build(events)
    assert "201" in sig.negatives
    assert "201" not in sig.positives
    # the abandoned content's THEME becomes an aversion (B1 = Family) -> downranks later
    assert sig.tag_aversion.get("theme_what:family", 0) > 0


def test_recency_decays_weight():
    recent = _build(view_events("u1", "101", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1)))
    old = _build(view_events("u1", "101", dwell=120, reason="next_button", base_ts=NOW - timedelta(days=60)))
    assert recent.positives["101"] > old.positives["101"]


def test_impression_not_viewed_is_soft_negative():
    from ai_engine.recsys.contracts.models import InteractionEvent
    ev = InteractionEvent(
        user_id="u1", event="CONTENT_VIEW_STARTED", content_id="101",
        ts=NOW - timedelta(hours=1), impressions=["301"],
    )
    seen = view_events("u1", "101", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    sig = _build(seen + [ev])
    assert "301" in sig.negatives  # shown alongside A1 but never opened


def test_demographics_seed_person_who_affinity():
    sig = _build([], demographics={"age": 20, "gender": "female", "nationality": "dutch"})
    keys = sig.tag_affinity.keys()
    assert any(k.startswith("person_who.age_group") for k in keys)
    assert any(k.startswith("person_who.gender_and_age") for k in keys)


def test_recent_views_ordered_and_recency_vector():
    events = (
        view_events("u1", "101", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=2))
        + view_events("u1", "103", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    )
    sig = _build(events)
    assert sig.recent_views[0] == "103"          # most-recent first (sequence)
    assert sig.recency_vector == VECTORS["103"]
    assert set(sig.viewed) == {"101", "103"}      # full view history


def test_aggregate_views_pairs_dwell():
    events = view_events("u1", "101", dwell=42, reason="next_button", base_ts=NOW)
    aggs = aggregate_views(events)
    assert aggs["101"].dwell_seconds == 42
    assert aggs["101"].end_reason is not None
