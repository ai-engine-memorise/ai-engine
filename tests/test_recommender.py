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


def test_tag_recall_works_with_semantic_off():
    # semantic disabled -> candidates must come purely from tag recall (case-insensitive)
    cfg = RecConfig()
    cfg.fusion.semantic = 0.0
    cfg.fusion.tag = 1.0
    events = view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    rec = Recommender(_store(), InMemoryUserModelStore(), cfg).recommend_for_signals(_signals(events))
    ids = _ids(rec)
    assert {"A2", "A3"} <= set(ids)                    # Forced-Labor siblings recalled by tag
    assert rec.items[0].content_id in {"A2", "A3"}     # and ranked top (tag-driven)
    assert rec.items[0].breakdown.get("tag", 0) > 0


def test_neutral_view_excluded_via_viewed_history():
    events = (
        view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
        + view_events("u1", "B1", dwell=8, reason="close_button", base_ts=NOW - timedelta(hours=1))
    )
    sig = _signals(events)
    assert "B1" in sig.viewed and "B1" not in sig.positives and "B1" not in sig.negatives  # neutral
    rec = Recommender(_store(), InMemoryUserModelStore(), RecConfig()).recommend_for_signals(sig)
    assert "B1" not in _ids(rec)              # viewed -> de-duplicated out


def test_recency_contributes_to_score():
    events = view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    rec = Recommender(_store(), InMemoryUserModelStore(), RecConfig()).recommend_for_signals(_signals(events))
    recs = [it for it in rec.items if it.kind == "recommendation"]
    assert recs and all("recency" in it.breakdown for it in recs)


def test_filter_restricts_candidates_to_tagged_content():
    from ai_engine.recsys.contracts.models import Content, Tag, UserSignals
    contents = {
        "L1": Content(id="L1", title="a", tags=[Tag(facet="location", label="AiARLocationBarrack3"),
                                                Tag(facet="theme_what", label="Forced Labor")]),
        "L2": Content(id="L2", title="b", tags=[Tag(facet="location", label="AiARLocationBarrack3")]),
        "X1": Content(id="X1", title="c", tags=[Tag(facet="theme_what", label="Family")]),
    }
    vectors = {k: [1.0, 0.0] for k in contents}
    store = FakeContentStore(contents, vectors)
    sig = UserSignals(user_id="u", tag_affinity={"theme_what:Forced Labor": 1.0})
    rec = Recommender(store, InMemoryUserModelStore(), RecConfig(distractor_enabled=False)) \
        .recommend_for_signals(sig, filter="AiARLocationBarrack3")
    ids = [i.content_id for i in rec.items]
    assert set(ids) <= {"L1", "L2"} and "X1" not in ids       # only the location's content
    assert rec.diagnostics["filter"] == "AiARLocationBarrack3"


def test_no_user_model_cold_start_returns_content():
    rec = Recommender(_store(), InMemoryUserModelStore(), CFG).recommend("ghost")
    assert rec.strategy == "cold"
    assert rec.items                                   # never empty-handed
    assert rec.diagnostics.get("cold_start_fallback") is True


def test_cold_start_fallback_when_signal_matches_nothing():
    from ai_engine.recsys.contracts.models import UserSignals
    sig = UserSignals(user_id="u", tag_affinity={"theme_what:NoSuchTheme": 1.0})  # no content match
    rec = Recommender(_store(), InMemoryUserModelStore(), CFG).recommend_for_signals(sig)
    assert rec.items and rec.diagnostics.get("cold_start_fallback") is True


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
