"""Compare k-means (hard) vs fuzzy-c-means (soft) visitor clustering on the same corpus.

Includes a deliberately BLENDED visitor (deep in two themes) to show where soft
membership helps: k-means must pick one bucket, fuzzy reports the split.

    python explain/compare_clustering.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from datetime import datetime, timedelta

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.testing.fixtures import make_contents_and_vectors, view_events
from ai_engine.recsys.explain.clusters import cluster_users, cluster_users_fuzzy, assign, assign_fuzzy

NOW = datetime(2026, 6, 12, 12, 0, 0)
CFG = RecConfig()
C, V = make_contents_and_vectors()


def _m(uid, evs):
    return build_user_signals(user_id=uid, events=evs, contents=C, vectors=V, now=NOW, cfg=CFG)


def _deep(uid, cid, h):
    return view_events(uid, cid, dwell=140, reason="next_button", base_ts=NOW - timedelta(hours=h))


CORPUS = [
    _m("forced_1", _deep("forced_1", "A1", 3) + _deep("forced_1", "A2", 2) + _deep("forced_1", "A3", 1)),
    _m("forced_2", _deep("forced_2", "A1", 3) + _deep("forced_2", "A3", 1)),
    _m("family_1", _deep("family_1", "B1", 3) + _deep("family_1", "B2", 2) + _deep("family_1", "B3", 1)),
    _m("family_2", _deep("family_2", "B2", 3) + _deep("family_2", "B3", 1)),
    # BLENDED: deep in BOTH Forced Labor and Family -> spans two clusters
    _m("blended", _deep("blended", "A1", 4) + _deep("blended", "A2", 3)
                  + _deep("blended", "B1", 2) + _deep("blended", "B2", 1)),
]


def main() -> None:
    k = 2
    km = cluster_users(CORPUS, k=k, seed=1)
    fc = cluster_users_fuzzy(CORPUS, c=k, m=2.0, seed=1)

    print("CLUSTER PROFILES (top theme per cluster)")
    for tag, model in (("kmeans", km), ("fcm", fc)):
        labels = ", ".join(f"c{p['cluster']}=[{', '.join(t['label'] for t in p['top_tags'][:2])}]"
                           for p in model["profiles"])
        print(f"  {tag:<7} {labels}")

    print("\nPER-VISITOR ASSIGNMENT  (kmeans = hard bucket | fcm = soft membership)")
    print(f"  {'visitor':<10} {'kmeans':<8} fcm membership")
    for s in CORPUS:
        hard = assign(s, km)["cluster"]
        soft = assign_fuzzy(s, fc)
        mem = "  ".join(f"c{j}:{p:.2f}" for j, p in soft["membership"].items())
        flag = "  <- split" if max(soft["membership"].values()) < 0.65 else ""
        print(f"  {s.user_id:<10} c{hard:<7} {mem}{flag}")


if __name__ == "__main__":
    main()
