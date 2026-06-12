from datetime import datetime, timedelta

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.contracts.models import UserSignals
from ai_engine.recsys.ranking.bandit import (
    LinearBandit, FEATURE_ORDER, feature_vector, _mat_inverse, _matvec,
)
from ai_engine.recsys.recommender import Recommender
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.testing.fakes import FakeContentStore, InMemoryUserModelStore
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events

NOW = datetime(2026, 6, 12, 12, 0, 0)
CONTENTS, VECTORS = make_contents_and_vectors()


# ----- pure core ----------------------------------------------------------- #

def test_mat_inverse_roundtrip():
    A = [[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]]
    inv = _mat_inverse(A)
    prod = [_matvec(A, [inv[r][c] for r in range(3)]) for c in range(3)]  # A·A⁻¹ columns
    for c in range(3):
        for r in range(3):
            assert abs(prod[c][r] - (1.0 if r == c else 0.0)) < 1e-9


def test_prior_theta_equals_static_weights():
    w = {"semantic": 0.30, "affinity": 0.25, "tag": 0.25, "recency": 0.10, "aversion": -0.25, "geo": 0.20}
    b = LinearBandit.with_prior(w, ridge=1.0)
    theta = b.theta()
    for name, t in zip(FEATURE_ORDER, theta):
        assert abs(t - w[name]) < 1e-9          # θ0 == the fusion weights (safe day-one prior)


def test_update_moves_theta_toward_rewarded_feature():
    w = {n: 0.2 for n in FEATURE_ORDER}
    b = LinearBandit.with_prior(w, ridge=1.0)
    tag_i = FEATURE_ORDER.index("tag")
    before = b.theta()[tag_i]
    x = [0.0] * len(FEATURE_ORDER); x[tag_i] = 1.0
    for _ in range(20):
        b.update(x, 1.0)                        # tag-only context, strong positive reward
    after = b.theta()[tag_i]
    assert after > before and after > 0.5       # learns to weight 'tag' up toward the reward


def test_update_penalizes_negatively_rewarded_feature():
    w = {n: 0.3 for n in FEATURE_ORDER}
    b = LinearBandit.with_prior(w, ridge=1.0)
    i = FEATURE_ORDER.index("semantic")
    x = [0.0] * len(FEATURE_ORDER); x[i] = 1.0
    for _ in range(20):
        b.update(x, -1.0)
    assert b.theta()[i] < 0.0                   # consistently bad -> negative weight


def test_exploration_bonus_is_nonnegative_and_optional():
    b = LinearBandit.with_prior({n: 0.2 for n in FEATURE_ORDER}, ridge=1.0, alpha=0.5)
    x = {"c": [0.5] * len(FEATURE_ORDER)}
    greedy = b.rank_scores(x, explore=False)["c"]
    ucb = b.rank_scores(x, explore=True)["c"]
    assert ucb >= greedy                        # UCB only adds an exploration bonus


def test_health_tracks_updates_data_and_uncertainty():
    b = LinearBandit.with_prior({n: 0.2 for n in FEATURE_ORDER}, ridge=1.0)
    h0 = b.health()
    assert h0["n_updates"] == 0 and all(d == 0 for d in h0["data"])
    ti, gi = FEATURE_ORDER.index("tag"), FEATURE_ORDER.index("geo")
    x = [0.0] * len(FEATURE_ORDER); x[ti] = 1.0
    for _ in range(5):
        b.update(x, 1.0)
    h = b.health()
    assert h["n_updates"] == 5
    assert h["data"][ti] > 0                 # tag fired -> has data
    assert h["data"][gi] == 0                # geo never fired -> still at prior, no data
    assert h["std"][ti] < h0["std"][ti]      # posterior uncertainty shrank where data arrived
    assert h["std"][gi] == h0["std"][gi]     # unchanged where none did


def test_to_from_dict_roundtrip():
    b = LinearBandit.with_prior({n: 0.1 for n in FEATURE_ORDER}, ridge=2.0, alpha=0.4)
    b.update([1.0] * len(FEATURE_ORDER), 0.7)
    b2 = LinearBandit.from_dict(b.to_dict())
    assert b2.theta() == b.theta() and b2.alpha == 0.4


# ----- recommender integration --------------------------------------------- #

def _signals():
    events = view_events("u1", "A1", dwell=120, reason="next_button", base_ts=NOW - timedelta(hours=1))
    return build_user_signals(user_id="u1", events=events, contents=CONTENTS, vectors=VECTORS,
                              now=NOW, cfg=RecConfig())


def test_features_always_logged_even_in_static_mode():
    rec = Recommender(FakeContentStore(CONTENTS, VECTORS), InMemoryUserModelStore(), RecConfig()) \
        .recommend_for_signals(_signals())
    targets = [it for it in rec.items if it.kind == "recommendation"]
    assert targets and all(len(it.features) == len(FEATURE_ORDER) for it in targets)
    assert rec.diagnostics["ranking"] == "static"


def test_bandit_prior_greedy_reproduces_static_scores():
    # θ0 == weights, so the bandit at its prior (greedy) must score IDENTICALLY to fusion
    sig = _signals()
    cfg_static = RecConfig(distractor_enabled=False)
    static = Recommender(FakeContentStore(CONTENTS, VECTORS), InMemoryUserModelStore(), cfg_static) \
        .recommend_for_signals(sig)

    cfg_bandit = RecConfig(distractor_enabled=False, ranking_mode="bandit", bandit_explore=False)
    weights = {n: getattr(cfg_bandit.fusion, n, 0.0) for n in FEATURE_ORDER}
    policy = LinearBandit.with_prior(weights, ridge=cfg_bandit.bandit_ridge)
    bandit = Recommender(FakeContentStore(CONTENTS, VECTORS), InMemoryUserModelStore(), cfg_bandit,
                         policy=policy).recommend_for_signals(sig)

    assert bandit.diagnostics["ranking"] == "bandit"
    s_static = {it.content_id: it.final_score for it in static.items}
    s_bandit = {it.content_id: it.final_score for it in bandit.items}
    assert s_static.keys() == s_bandit.keys()
    for cid in s_static:
        assert abs(s_static[cid] - s_bandit[cid]) < 1e-9
