"""Bandit bootstrap from interaction sequences: feature-replay + weighted updates."""
from datetime import datetime, timedelta

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.ranking.bandit import LinearBandit, FEATURE_ORDER, feature_vector
from ai_engine.recsys.recommender import score_features
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.signals.engagement import estimate_reading_time, engagement_strength
from ai_engine.recsys.contracts.enums import EndReason
from ai_engine.recsys.testing.fakes import FakeContentStore
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events

NOW = datetime(2026, 6, 12, 12, 0, 0)
CFG = RecConfig()
C, V = make_contents_and_vectors()


def test_score_features_is_the_shared_context_vector():
    sig = build_user_signals(
        user_id="u",
        events=view_events("u", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1)),
        contents=C, vectors=V, now=NOW, cfg=CFG)
    per = score_features(sig, C["A2"], V["A2"], [], CFG)
    assert set(per) == {"semantic", "affinity", "tag", "recency", "aversion"}   # geo only when near given
    assert len(feature_vector(per)) == len(FEATURE_ORDER)
    assert per["tag"] > 0                                   # A2 shares Forced-Labor theme


def test_update_weight_scales_contribution():
    w = {n: 0.2 for n in FEATURE_ORDER}
    x = [0.0] * len(FEATURE_ORDER); x[FEATURE_ORDER.index("tag")] = 1.0

    full = LinearBandit.with_prior(w, ridge=1.0)
    full.update(x, 1.0, weight=1.0)
    half_twice = LinearBandit.with_prior(w, ridge=1.0)
    half_twice.update(x, 1.0, weight=0.5)
    half_twice.update(x, 1.0, weight=0.5)
    # two half-weight updates == one full-weight update
    for a, b in zip(full.theta(), half_twice.theta()):
        assert abs(a - b) < 1e-9


def _replay(store, user_views):
    """Temporal replay (the bandit/replay.py core): model BEFORE each view -> (x, reward)."""
    cfg = CFG
    history = []
    out = []
    for v in user_views:
        cid = v["content_id"]
        ids = list({e.content_id for e in history if e.content_id})
        sig = build_user_signals(user_id="u", events=history, contents=store.get(ids),
                                 vectors=store.get_vectors(ids), now=NOW, cfg=cfg)
        liked = []
        if sig.positives:
            lv = store.get_vectors(list(sig.positives)); mx = max(sig.positives.values()) or 1.0
            liked = [(sig.positives[c] / mx, lv[c]) for c in sig.positives if c in lv]
        content, vec = store.get([cid]).get(cid), store.get_vectors([cid]).get(cid)
        per = score_features(sig, content, vec, liked, cfg)
        est = estimate_reading_time(content.word_count, content.has_image, cfg)
        r = engagement_strength(dwell_seconds=v["dwell"], est_reading_time=est,
                                end_reason=EndReason(v["reason"]), visits=1, survey_rating=None, cfg=cfg)
        out.append((feature_vector(per), r))
        history += view_events("u", cid, dwell=v["dwell"], reason=v["reason"], base_ts=v["ts"])
    return out


def test_temporal_replay_yields_samples_and_shifts_theta():
    store = FakeContentStore(C, V)
    # a Forced-Labor reader: deep, completing -> positive rewards on tag/semantic-rich contexts
    views = [
        {"content_id": "A1", "dwell": 140, "reason": "next_button", "ts": NOW - timedelta(hours=3)},
        {"content_id": "A2", "dwell": 150, "reason": "next_button", "ts": NOW - timedelta(hours=2)},
        {"content_id": "A3", "dwell": 130, "reason": "next_button", "ts": NOW - timedelta(hours=1)},
    ]
    samples = _replay(store, views)
    assert len(samples) == 3
    # later views have nonzero context (the model has warmed up) and positive reward
    assert any(any(xi != 0 for xi in x) for x, _ in samples)
    assert any(r > 0 for _, r in samples)

    weights = {n: getattr(CFG.fusion, n, 0.0) for n in FEATURE_ORDER}
    bandit = LinearBandit.with_prior(weights, ridge=1.0)
    before = bandit.theta()
    for x, r in samples:
        bandit.update(x, r, weight=0.5)        # down-weighted foreign bootstrap
    after = bandit.theta()
    assert before != after                     # bootstrap moved the policy off the prior
