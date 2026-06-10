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
        view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
        + view_events("u1", "A2", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    )
    sig = _build(events)
    assert "A1" in sig.positives and "A2" in sig.positives
    assert sig.tag_affinity.get("theme_what:Forced Labor", 0) > 0
    # taste vector points along the Forced-Labor axis (index 0)
    assert sig.taste_vector is not None
    assert sig.taste_vector[0] == max(sig.taste_vector)


def test_quick_abandon_becomes_negative():
    events = view_events("u1", "B1", dwell=1, reason="abandon", base_ts=NOW - timedelta(hours=1))
    sig = _build(events)
    assert "B1" in sig.negatives
    assert "B1" not in sig.positives


def test_recency_decays_weight():
    recent = _build(view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1)))
    old = _build(view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(days=60)))
    assert recent.positives["A1"] > old.positives["A1"]


def test_impression_not_viewed_is_soft_negative():
    from ai_engine.recsys.contracts.models import InteractionEvent
    ev = InteractionEvent(
        user_id="u1", event="CONTENT_VIEW_STARTED", content_id="A1",
        ts=NOW - timedelta(hours=1), impressions=["C1"],
    )
    seen = view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    sig = _build(seen + [ev])
    assert "C1" in sig.negatives  # shown alongside A1 but never opened


def test_demographics_seed_person_who_affinity():
    sig = _build([], demographics={"age": 20, "gender": "female", "nationality": "dutch"})
    keys = sig.tag_affinity.keys()
    assert any(k.startswith("person_who.age_group") for k in keys)
    assert any(k.startswith("person_who.gender_and_age") for k in keys)


def test_aggregate_views_pairs_dwell():
    events = view_events("u1", "A1", dwell=42, reason="next_button", base_ts=NOW)
    aggs = aggregate_views(events)
    assert aggs["A1"].dwell_seconds == 42
    assert aggs["A1"].end_reason is not None
