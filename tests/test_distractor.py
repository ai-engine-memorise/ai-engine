from datetime import datetime, timedelta

from ai_engine.recsys.contracts import RecConfig
from ai_engine.recsys.recommender import Recommender
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.testing.fakes import FakeContentStore, InMemoryUserModelStore
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events

NOW = datetime(2026, 6, 11, 12, 0, 0)
CONTENTS, VECTORS = make_contents_and_vectors()


def _signals(events, cfg):
    return build_user_signals(user_id="u1", events=events, contents=CONTENTS,
                              vectors=VECTORS, now=NOW, cfg=cfg)


def _rec(cfg):
    events = view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    store = FakeContentStore(CONTENTS, VECTORS)
    return Recommender(store, InMemoryUserModelStore(), cfg).recommend_for_signals(_signals(events, cfg))


def test_one_distractor_labelled_at_fixed_slot():
    cfg = RecConfig(distractor_slot=2, final_limit=4)  # final_limit < catalogue -> headroom
    rec = _rec(cfg)
    kinds = [it.kind for it in rec.items]
    assert kinds.count("distractor") == 1
    d = next(it for it in rec.items if it.kind == "distractor")
    assert rec.items.index(d) == min(cfg.distractor_slot, len(rec.items) - 1)
    assert d.content_id not in {"A1"}                    # unseen
    assert rec.diagnostics["distractor"]["content_id"] == d.content_id


def test_distractor_strategies_all_produce_one():
    for strat in ("max_dissimilar", "unexplored_theme", "random"):
        rec = _rec(RecConfig(distractor_strategy=strat, final_limit=4))
        assert sum(1 for it in rec.items if it.kind == "distractor") == 1, strat


def test_distractor_can_be_disabled():
    rec = _rec(RecConfig(distractor_enabled=False, final_limit=4))
    assert all(it.kind == "recommendation" for it in rec.items)


def test_distractor_not_a_seen_item():
    rec = _rec(RecConfig(final_limit=4))
    d = next(it for it in rec.items if it.kind == "distractor")
    assert d.content_id != "A1"
