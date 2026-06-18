"""Offline trainer for explainable visitor clusters.

Gathers the live user models (Redis `umodel:*`) into a corpus, clusters them by
tag-affinity profile, and writes a ClusterModel JSON. Each cluster is described by
its top taxonomy tags (the cluster IS the explanation) + a Falk breadth hint.

    REDIS_URL=redis://localhost:6379 python explain/cluster_train.py --k 4 --out ./data/clusters.json
    # then serve with:  CLUSTER_MODEL_PATH=./data/clusters.json   (GET /api/clusters, /usermodel/explain)
"""
from __future__ import annotations
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_engine.recsys.contracts.models import UserSignals
from ai_engine.recsys.explain.clusters import cluster_users, cluster_users_fuzzy


def _gather_from_redis(url: str, prefix: str = "umodel") -> list[UserSignals]:
    import redis
    client = redis.from_url(url, decode_responses=True)
    out: list[UserSignals] = []
    for key in client.scan_iter(match=f"{prefix}:*"):
        raw = client.get(key)
        if raw:
            try:
                out.append(UserSignals.model_validate_json(raw))
            except Exception:
                pass
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--redis", default=os.getenv("REDIS_URL"))
    ap.add_argument("--method", choices=("kmeans", "fcm"), default="kmeans",
                    help="kmeans = hard buckets; fcm = fuzzy soft membership")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--m", type=float, default=2.0, help="fcm fuzziness")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.getenv("CLUSTER_MODEL_PATH", "./data/clusters.json"))
    args = ap.parse_args()

    if not args.redis:
        sys.exit("set REDIS_URL (or --redis) — the corpus is the live user models")
    corpus = _gather_from_redis(args.redis)
    if not corpus:
        sys.exit("no user models found in Redis (umodel:*) — serve some traffic first")

    if args.method == "fcm":
        model = cluster_users_fuzzy(corpus, c=args.k, m=args.m, seed=args.seed)
    else:
        model = cluster_users(corpus, k=args.k, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    # keep what the API needs: profiles + the geometry to assign new visitors + method/m
    persisted = {"keys": model["keys"], "centroids": model["centroids"],
                 "profiles": model["profiles"], "method": model.get("method", "kmeans"),
                 "m": model.get("m", 2.0)}
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(persisted, fh, indent=2, ensure_ascii=False)

    print(f"users={len(corpus)}  clusters={len(model['profiles'])}")
    for p in model["profiles"]:
        tags = ", ".join(t["label"] for t in p["top_tags"][:4])
        print(f"  cluster {p['cluster']}  n={p['size']:<4} {p['falk_hint']:<24} [{tags}]")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
