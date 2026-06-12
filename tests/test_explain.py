from datetime import datetime, timedelta

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.contracts.models import UserSignals
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events
from ai_engine.recsys.explain.persona import explain_user
from ai_engine.recsys.explain.verbalize import verbalize
from ai_engine.recsys.explain.clusters import cluster_users, assign

NOW = datetime(2026, 6, 12, 12, 0, 0)
CFG = RecConfig()
C, V = make_contents_and_vectors()


def _model(uid, events):
    return build_user_signals(user_id=uid, events=events, contents=C, vectors=V, now=NOW, cfg=CFG)


def _deep_narrow(uid="hobby"):   # 3 Forced-Labor stories, fully read
    return _model(uid,
        view_events(uid, "A1", dwell=140, reason="next_button", base_ts=NOW - timedelta(hours=3))
        + view_events(uid, "A2", dwell=150, reason="next_button", base_ts=NOW - timedelta(hours=2))
        + view_events(uid, "A3", dwell=130, reason="next_button", base_ts=NOW - timedelta(hours=1)))


def _broad_skim(uid="broad"):    # one story from each theme, all abandoned quickly
    return _model(uid,
        view_events(uid, "A1", dwell=8, reason="close_button", base_ts=NOW - timedelta(hours=4))
        + view_events(uid, "B1", dwell=6, reason="abandon", base_ts=NOW - timedelta(hours=3))
        + view_events(uid, "C1", dwell=7, reason="close_button", base_ts=NOW - timedelta(hours=2)))


# ----- behavior summary on the user model ---------------------------------- #

def test_behavior_summary_populated():
    b = _deep_narrow().behavior
    assert b["n_views"] == 3 and b["completion_rate"] == 1.0
    assert b["avg_dwell_ratio"] >= 0.9 and b["depth"] == 1.0


# ----- Falk visitor types -------------------------------------------------- #

def test_deep_narrow_reads_as_hobbyist():
    exp = explain_user(_deep_narrow(), C)
    assert exp.visitor_type.type == "Hobbyist"
    assert exp.engagement_style == "deep_reader"
    assert exp.experience_preference == "cognitive"
    # interests carry their EVIDENCE (Tintarev scrutability)
    forced = next(i for i in exp.interests if i.label.lower() == "forced labor")
    assert set(forced.evidence) == {"A1", "A2", "A3"}


def test_broad_skim_reads_as_experience_seeker():
    exp = explain_user(_broad_skim(), C)
    assert exp.visitor_type.type == "Experience-Seeker"
    assert exp.engagement_style in {"skimmer", "sampler"}
    # browsed-but-not-engaged is NOT collapsed to cold: it has a trajectory
    assert len(exp.trajectory) == 3
    assert "no engagement yet" not in verbalize(exp)


def test_aversion_surfaces_in_explanation():
    # like Forced-Labor, abandon Family -> Family becomes an explained aversion
    sig = _model("u",
        view_events("u", "A1", dwell=140, reason="next_button", base_ts=NOW - timedelta(hours=2))
        + view_events("u", "B1", dwell=1, reason="abandon", base_ts=NOW - timedelta(hours=1)))
    exp = explain_user(sig, C)
    assert any(a.label.lower() == "family" for a in exp.aversions)


def test_cold_user_explanation_is_graceful():
    exp = explain_user(UserSignals(user_id="ghost"))
    assert exp.is_cold and not exp.interests
    assert "New visitor" in verbalize(exp)


def test_verbalize_is_evidence_grounded_prose():
    s = verbalize(explain_user(_deep_narrow(), C))
    assert "Hobbyist" in s and "forced labor" in s.lower()


# ----- explainable clusters ------------------------------------------------ #

def test_clusters_separate_narrow_from_broad_with_readable_profiles():
    corpus = [_deep_narrow("h1"), _deep_narrow("h2"), _broad_skim("b1"), _broad_skim("b2")]
    model = cluster_users(corpus, k=2, seed=1)
    assert len(model["profiles"]) == 2
    # every cluster is described by its top taxonomy tags (the cluster IS the explanation)
    assert all(p["top_tags"] for p in model["profiles"])
    # a deep-narrow user lands in a cluster and shares its driving tags
    a = assign(_deep_narrow("h3"), model)
    assert a["cluster"] is not None
    assert any(t["label"].lower() == "forced labor" for t in a["shared_tags"])


def test_assign_handles_empty_model():
    assert assign(UserSignals(user_id="x"), {"keys": [], "centroids": []})["cluster"] is None
