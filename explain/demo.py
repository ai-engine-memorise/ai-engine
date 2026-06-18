"""Demo / smoke for the explainability + ranking additions — no infra, no network.

Synthesizes a few museum-visitor ARCHETYPES as event streams, builds their user models,
and prints: the persona (Falk type + Pekarik preference + interests w/ evidence + prose),
the learned visitor clusters, and the top recommendations with the score breakdown.

    python explain/demo.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")     # the trajectory arrow prints on Windows
except Exception:
    pass

from datetime import datetime, timedelta

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.recommender import Recommender
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.testing.fakes import FakeContentStore, InMemoryUserModelStore
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events
from ai_engine.recsys.explain.persona import explain_user
from ai_engine.recsys.explain.verbalize import verbalize
from ai_engine.recsys.explain.clusters import cluster_users, assign

NOW = datetime(2026, 6, 12, 12, 0, 0)
CFG = RecConfig()
CONTENTS, VECTORS = make_contents_and_vectors()


def _model(uid, events):
    return build_user_signals(user_id=uid, events=events, contents=CONTENTS,
                              vectors=VECTORS, now=NOW, cfg=CFG)


def _deep(uid, cid, h):
    return view_events(uid, cid, dwell=140, reason="next_button", base_ts=NOW - timedelta(hours=h))


def _skim(uid, cid, h):
    return view_events(uid, cid, dwell=7, reason="close_button", base_ts=NOW - timedelta(hours=h))


ARCHETYPES = {
    # deep + narrow (all Forced Labor)
    "hobbyist":          _deep("hobbyist", "A1", 3) + _deep("hobbyist", "A2", 2) + _deep("hobbyist", "A3", 1),
    # broad + engaged (one per theme, read fully)
    "explorer":          _deep("explorer", "A1", 3) + _deep("explorer", "B1", 2) + _deep("explorer", "C1", 1),
    # broad + light (one per theme, skimmed)
    "experience_seeker": _skim("experience_seeker", "A1", 3) + _skim("experience_seeker", "B1", 2) + _skim("experience_seeker", "C1", 1),
    # few + slow + narrow
    "recharger":         view_events("recharger", "B2", dwell=200, reason="close_button", base_ts=NOW - timedelta(hours=1)),
}


def main() -> None:
    store = FakeContentStore(CONTENTS, VECTORS)
    models = {uid: _model(uid, evs) for uid, evs in ARCHETYPES.items()}

    sep = "=" * 78
    for uid, sig in models.items():
        contents = store.get(list(dict.fromkeys(list(sig.positives) + list(sig.negatives) + sig.recent_views)))
        exp = explain_user(sig, contents)
        print(f"\n{sep}\n{uid.upper()}   behavior={sig.behavior}")
        vt = exp.visitor_type
        print(f"  Falk: {vt.type} (conf {vt.confidence:.2f})  |  Pekarik: {exp.experience_preference}"
              f"  |  style: {exp.engagement_style}")
        print(f"  interests: {[(i.label, round(i.weight, 2), i.evidence) for i in exp.interests[:3]]}")
        print(f"  trajectory: {exp.trajectory}")
        print(f"  PROSE: {verbalize(exp)}")

        recs = Recommender(store, InMemoryUserModelStore(), CFG).recommend_for_signals(sig)
        top = [(it.content_id, round(it.final_score, 3), it.kind) for it in recs.items[:4]]
        print(f"  top recs: {top}")

    # explainable clusters over the archetypes
    print(f"\n{sep}\nCLUSTERS")
    model = cluster_users(list(models.values()), k=2, seed=1)
    for p in model["profiles"]:
        tags = ", ".join(t["label"] for t in p["top_tags"][:4])
        print(f"  cluster {p['cluster']}  n={p['size']}  {p['falk_hint']:<24} [{tags}]")
    for uid, sig in models.items():
        a = assign(sig, model)
        print(f"  {uid:<18} -> cluster {a['cluster']}  shares {[t['label'] for t in a['shared_tags'][:3]]}")


if __name__ == "__main__":
    main()
