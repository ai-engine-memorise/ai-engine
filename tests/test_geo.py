"""Geo: proximity scoring + the geo radius filter, INDEPENDENT of the tag filter."""
from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.contracts.models import Content, Tag, UserSignals
from ai_engine.recsys.ranking.scorers import haversine_m, score_geo
from ai_engine.recsys.recommender import Recommender
from ai_engine.recsys.testing.fakes import FakeContentStore, InMemoryUserModelStore

# Camp Westerbork-ish anchor + points at increasing distance
ANCHOR = (52.9200, 6.6000)


def _geo_world():
    # G1 ~0m, G2 ~150m, G3 ~5km; F1 has a tag but is far; all share a vector
    contents = {
        "G1": Content(id="G1", title="at anchor", lat=52.9200, lon=6.6000,
                      tags=[Tag(facet="location", label="Camp")]),
        "G2": Content(id="G2", title="nearby", lat=52.9213, lon=6.6000,
                      tags=[Tag(facet="location", label="Camp")]),
        "G3": Content(id="G3", title="far", lat=52.9650, lon=6.6000,
                      tags=[Tag(facet="location", label="Camp")]),
        "F1": Content(id="F1", title="far, other tag", lat=52.9650, lon=6.6000,
                      tags=[Tag(facet="theme_what", label="Family")]),
    }
    vectors = {k: [1.0, 0.0] for k in contents}
    return FakeContentStore(contents, vectors)


def test_haversine_and_score_geo_bounds():
    assert haversine_m(*ANCHOR, *ANCHOR) == 0.0
    near = score_geo(Content(id="x", lat=ANCHOR[0], lon=ANCHOR[1]), ANCHOR, 300.0)
    far = score_geo(Content(id="y", lat=53.5, lon=6.6), ANCHOR, 300.0)
    assert near == 1.0 and 0.0 <= far < 0.01
    assert score_geo(Content(id="z"), ANCHOR, 300.0) == 0.0          # no coords -> 0
    assert score_geo(Content(id="z", lat=1, lon=1), None, 300.0) == 0.0  # no ref -> 0


def test_geo_scoring_orders_by_proximity():
    rec = Recommender(_geo_world(), InMemoryUserModelStore(), RecConfig(distractor_enabled=False)) \
        .recommend_for_signals(UserSignals(user_id="u"), near=ANCHOR)
    score = {it.content_id: it.breakdown.get("geo", 0.0) for it in rec.items}
    assert score["G1"] > score["G2"] > score["G3"]      # nearer -> higher geo contribution
    assert all("geo" in it.breakdown for it in rec.items)


def test_geo_radius_filter_restricts_candidates():
    rec = Recommender(_geo_world(), InMemoryUserModelStore(), RecConfig(distractor_enabled=False)) \
        .recommend_for_signals(UserSignals(user_id="u"), near=ANCHOR, geo_radius_m=500.0)
    ids = {it.content_id for it in rec.items}
    assert ids == {"G1", "G2"}                          # G3/F1 (~5km) excluded by radius


def test_geo_and_tag_filters_are_independent_and_compose():
    store = _geo_world()
    # tag filter alone: all Camp-tagged (near AND far)
    tag_only = Recommender(store, InMemoryUserModelStore(), RecConfig(distractor_enabled=False)) \
        .recommend_for_signals(UserSignals(user_id="u"), filter="Camp")
    assert {it.content_id for it in tag_only.items} == {"G1", "G2", "G3"}
    # geo filter alone: everything within radius regardless of tag
    geo_only = Recommender(store, InMemoryUserModelStore(), RecConfig(distractor_enabled=False)) \
        .recommend_for_signals(UserSignals(user_id="u"), near=ANCHOR, geo_radius_m=500.0)
    assert {it.content_id for it in geo_only.items} == {"G1", "G2"}
    # BOTH -> intersection (AND): Camp-tagged AND within radius
    both = Recommender(store, InMemoryUserModelStore(), RecConfig(distractor_enabled=False)) \
        .recommend_for_signals(UserSignals(user_id="u"), filter="Camp", near=ANCHOR, geo_radius_m=500.0)
    assert {it.content_id for it in both.items} == {"G1", "G2"}
    assert set(both.diagnostics["generators"]) == {"filter", "geo"}
