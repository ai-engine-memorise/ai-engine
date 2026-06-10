from datetime import datetime, timedelta

from ai_engine.recsys.contracts import RecConfig
from ai_engine.recsys.recommender import Recommender
from ai_engine.recsys.updater import UserModelUpdater
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.testing.fakes import (
    FakeContentStore, FakeEventSource, InMemoryUserModelStore,
)
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events

NOW = datetime(2026, 6, 10, 12, 0, 0)
CFG = RecConfig()
CONTENTS, VECTORS = make_contents_and_vectors()


def _store():
    return FakeContentStore(CONTENTS, VECTORS)


def _signals(events, demographics=None):
    return build_user_signals(
        user_id="u1", events=events, contents=CONTENTS, vectors=VECTORS,
        now=NOW, cfg=CFG, demographics=demographics,
    )


def _ids(rec):
    return [it.content_id for it in rec.items]


# --------------------------------------------------------------------------- #
# Golden scenarios (known answer)
# --------------------------------------------------------------------------- #

def test_clear_taste_surfaces_unseen_sibling():
    events = (
        view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
        + view_events("u1", "A2", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    )
    rec = Recommender(_store(), InMemoryUserModelStore(), CFG).recommend_for_signals(_signals(events))
    ids = _ids(rec)
    assert "A3" in ids                 # the unseen Forced-Labor story is recommended
    assert "A1" not in ids and "A2" not in ids  # already-seen items excluded
    assert rec.items[0].content_id == "A3"


def test_disliked_item_never_recommended():
    events = (
        view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
        + view_events("u1", "C1", dwell=1, reason="abandon", base_ts=NOW - timedelta(hours=1))
    )
    rec = Recommender(_store(), InMemoryUserModelStore(), CFG).recommend_for_signals(_signals(events))
    assert "C1" not in _ids(rec)


def test_no_user_model_is_cold_and_empty():
    rec = Recommender(_store(), InMemoryUserModelStore(), CFG).recommend("ghost")
    assert rec.strategy == "cold"
    assert rec.items == []


# --------------------------------------------------------------------------- #
# Path B: ingest -> store -> serve
# --------------------------------------------------------------------------- #

def test_online_flow_updater_then_recommender():
    source = FakeEventSource()
    for ev in view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1)):
        source.append(ev)
    for ev in view_events("u1", "A2", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1)):
        source.append(ev)

    store = InMemoryUserModelStore()
    content = _store()
    UserModelUpdater(content, store, CFG).refresh("u1", source, now=NOW)

    rec = Recommender(content, store, CFG).recommend("u1")
    assert rec.strategy == "warm"
    assert "A3" in _ids(rec)


# --------------------------------------------------------------------------- #
# Invariants (hold for any input)
# --------------------------------------------------------------------------- #

def test_invariants_scores_seen_dedup_limit():
    cfg = RecConfig(final_limit=3)
    events = (
        view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
        + view_events("u1", "B1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    )
    rec = Recommender(_store(), InMemoryUserModelStore(), cfg).recommend_for_signals(_signals(events))

    ids = _ids(rec)
    assert len(ids) <= cfg.final_limit
    assert len(ids) == len(set(ids))                 # no duplicates
    assert "A1" not in ids and "B1" not in ids       # no seen items
    for it in rec.items:
        assert 0.0 <= it.final_score <= 1.0
        assert abs(sum(it.breakdown.values()) - it.final_score) < 1e-9
