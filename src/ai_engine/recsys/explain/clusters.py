"""Explainable visitor clusters — k-means over tag-affinity profiles.

The cluster IS the explanation: each centroid is literally a tag-weight profile in the
expert taxonomy, so a cluster reads as "resistance + liberation, broad" rather than an
opaque embedding id. Breadth (entropy) of the centroid maps it loosely to a Falk type.

Pure-Python (no numpy/sklearn): corpus of UserSignals -> ClusterModel; assign(signals).
The corpus is gathered offline (explain/cluster_train.py reads the live Redis models).
"""
from __future__ import annotations
from typing import Optional

from ..contracts.models import UserSignals
from .persona import _norm_entropy, _split


def _rng(seed: int):
    s = seed or 1
    def nxt() -> float:
        nonlocal s
        s = (1103515245 * s + 12345) & 0x7FFFFFFF
        return s / 0x7FFFFFFF
    return nxt


def vectorize(corpus: list[UserSignals], *, min_freq: int = 1,
              max_features: Optional[int] = None) -> tuple[list[str], list[list[float]]]:
    """Affinity dicts -> a shared dense feature space (keys kept if they appear in
    >= min_freq users; optionally capped to the most frequent max_features)."""
    freq: dict[str, int] = {}
    for s in corpus:
        for k in s.tag_affinity:
            freq[k] = freq.get(k, 0) + 1
    keys = [k for k, c in freq.items() if c >= min_freq]
    keys.sort(key=lambda k: (-freq[k], k))
    if max_features:
        keys = keys[:max_features]
    keys.sort()
    matrix = [[float(s.tag_affinity.get(k, 0.0)) for k in keys] for s in corpus]
    return keys, matrix


def _dist2(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def kmeans(matrix: list[list[float]], k: int, *, iters: int = 25, seed: int = 0):
    """Lloyd's algorithm with seeded k-means++ init. Returns (labels, centroids)."""
    n = len(matrix)
    k = max(1, min(k, n))
    rnd = _rng(seed)
    # k-means++ init
    centroids = [matrix[int(rnd() * n) % n]]
    while len(centroids) < k:
        d2 = [min(_dist2(x, c) for c in centroids) for x in matrix]
        tot = sum(d2) or 1.0
        target, acc, pick = rnd() * tot, 0.0, 0
        for i, d in enumerate(d2):
            acc += d
            if acc >= target:
                pick = i
                break
        centroids.append(matrix[pick])
    centroids = [list(c) for c in centroids]

    labels = [0] * n
    for _ in range(iters):
        changed = False
        for i, x in enumerate(matrix):
            best = min(range(k), key=lambda c: _dist2(x, centroids[c]))
            if best != labels[i]:
                labels[i], changed = best, True
        dim = len(matrix[0]) if matrix else 0
        sums = [[0.0] * dim for _ in range(k)]
        counts = [0] * k
        for x, lb in zip(matrix, labels):
            counts[lb] += 1
            row = sums[lb]
            for j, v in enumerate(x):
                row[j] += v
        for c in range(k):
            if counts[c]:
                centroids[c] = [v / counts[c] for v in sums[c]]
        if not changed:
            break
    return labels, centroids


def fuzzy_cmeans(matrix: list[list[float]], c: int, *, m: float = 2.0,
                 iters: int = 50, seed: int = 0, tol: float = 1e-4):
    """Fuzzy c-means: SOFT membership (a visitor can be 70% one cluster, 30% another —
    fits Falk's overlapping identities). Returns (U, centroids) where U[i][j] in [0,1],
    rows sum to 1. `m` = fuzziness (2.0 typical; ->1 approaches hard k-means)."""
    n = len(matrix)
    dim = len(matrix[0]) if matrix else 0
    c = max(1, min(c, n))
    rnd = _rng(seed)
    # random membership init, each row normalized to sum 1
    U = []
    for _ in range(n):
        row = [rnd() + 1e-6 for _ in range(c)]
        s = sum(row)
        U.append([v / s for v in row])
    centroids = [[0.0] * dim for _ in range(c)]
    exp = 2.0 / (m - 1.0) if m > 1.0 else 2.0

    for _ in range(iters):
        for j in range(c):                                    # weighted centroids
            num = [0.0] * dim
            den = 0.0
            for i in range(n):
                w = U[i][j] ** m
                den += w
                xi = matrix[i]
                for d in range(dim):
                    num[d] += w * xi[d]
            if den > 0:
                centroids[j] = [v / den for v in num]
        max_delta = 0.0
        for i in range(n):                                    # update memberships
            dists = [_dist2(matrix[i], centroids[j]) ** 0.5 for j in range(c)]
            zero = [j for j, d in enumerate(dists) if d < 1e-12]
            if zero:
                row = [(1.0 / len(zero)) if j in zero else 0.0 for j in range(c)]
            else:
                row = [1.0 / sum((dists[j] / dists[k]) ** exp for k in range(c)) for j in range(c)]
            for j in range(c):
                max_delta = max(max_delta, abs(row[j] - U[i][j]))
            U[i] = row
        if max_delta < tol:
            break
    return U, centroids


def _memberships_for(x: list[float], centroids: list[list[float]], m: float) -> list[float]:
    exp = 2.0 / (m - 1.0) if m > 1.0 else 2.0
    dists = [_dist2(x, cen) ** 0.5 for cen in centroids]
    zero = [j for j, d in enumerate(dists) if d < 1e-12]
    if zero:
        return [(1.0 / len(zero)) if j in zero else 0.0 for j in range(len(centroids))]
    return [1.0 / sum((dists[j] / dists[k]) ** exp for k in range(len(centroids)))
            for j in range(len(centroids))]


def _falk_from_breadth(breadth: float) -> str:
    if breadth >= 0.6:
        return "Explorer-like (broad)"
    if breadth <= 0.3:
        return "Hobbyist-like (narrow)"
    return "Mixed"


def cluster_users(corpus: list[UserSignals], *, k: int = 4, seed: int = 0,
                  min_freq: int = 1, top: int = 5) -> dict:
    """Cluster the visitor corpus by affinity profile. Returns an explainable model:
    feature keys, centroids, and a readable profile per cluster (top tags + Falk hint)."""
    warm = [s for s in corpus if s.tag_affinity]
    if not warm:
        return {"keys": [], "centroids": [], "profiles": [], "user_ids": [], "method": "kmeans"}
    keys, matrix = vectorize(warm, min_freq=min_freq)
    labels, centroids = kmeans(matrix, k, seed=seed)
    members = [[warm[i].user_id for i, lb in enumerate(labels) if lb == c] for c in range(len(centroids))]
    return {"keys": keys, "centroids": centroids, "method": "kmeans",
            "profiles": _cluster_profiles(keys, centroids, members, top),
            "user_ids": [s.user_id for s in warm], "labels": labels}


def _shared_tags(x: list[float], centroid: list[float], keys: list[str], top: int = 5) -> list[dict]:
    shared = sorted(((k, min(xi, ci)) for k, xi, ci in zip(keys, x, centroid) if xi > 0 and ci > 0),
                    key=lambda kv: kv[1], reverse=True)
    return [{"facet": _split(k)[0], "label": _split(k)[1]} for k, _ in shared[:top]]


def assign(signals: UserSignals, model: dict) -> dict:
    """Place a visitor in the nearest cluster, explained by the tags they share with it."""
    keys, centroids = model.get("keys", []), model.get("centroids", [])
    if not keys or not centroids:
        return {"cluster": None, "shared_tags": []}
    x = [float(signals.tag_affinity.get(k, 0.0)) for k in keys]
    c = min(range(len(centroids)), key=lambda i: _dist2(x, centroids[i]))
    return {"cluster": c, "shared_tags": _shared_tags(x, centroids[c], keys)}


def _cluster_profiles(keys: list[str], centroids: list[list[float]],
                      members: list[list[str]], top: int) -> list[dict]:
    profiles = []
    for c, centroid in enumerate(centroids):
        ranked = sorted(zip(keys, centroid), key=lambda kv: kv[1], reverse=True)
        top_tags = [{"facet": _split(k)[0], "label": _split(k)[1], "weight": round(w, 4)}
                    for k, w in ranked[:top] if w > 0]
        theme_w = [w for kk, w in zip(keys, centroid) if kk.startswith("theme_what")]
        breadth = _norm_entropy(theme_w if theme_w else centroid)
        profiles.append({"cluster": c, "size": len(members[c]), "top_tags": top_tags,
                         "breadth": round(breadth, 4), "falk_hint": _falk_from_breadth(breadth),
                         "members": members[c][:50]})
    return profiles


def cluster_users_fuzzy(corpus: list[UserSignals], *, c: int = 4, m: float = 2.0, seed: int = 0,
                        min_freq: int = 1, top: int = 5) -> dict:
    """Fuzzy-c-means version: each visitor gets a membership distribution across clusters
    (soft), not one bucket. Profiles use the argmax (dominant) members; `memberships`
    carries the full soft assignment per user. `method`='fcm'."""
    warm = [s for s in corpus if s.tag_affinity]
    if not warm:
        return {"keys": [], "centroids": [], "profiles": [], "memberships": [], "method": "fcm", "m": m}
    keys, matrix = vectorize(warm, min_freq=min_freq)
    U, centroids = fuzzy_cmeans(matrix, c, m=m, seed=seed)
    k = len(centroids)
    members: list[list[str]] = [[] for _ in range(k)]
    memberships = []
    for i, s in enumerate(warm):
        dom = max(range(k), key=lambda j: U[i][j])
        members[dom].append(s.user_id)
        memberships.append({"user_id": s.user_id, "dominant": dom,
                            "membership": {j: round(U[i][j], 4) for j in range(k)}})
    return {"keys": keys, "centroids": centroids, "m": m, "method": "fcm",
            "profiles": _cluster_profiles(keys, centroids, members, top),
            "memberships": memberships, "user_ids": [s.user_id for s in warm]}


def assign_fuzzy(signals: UserSignals, model: dict) -> dict:
    """Soft assignment: the visitor's membership across ALL clusters (sums to 1), plus
    the dominant cluster and the tags shared with it."""
    keys, centroids = model.get("keys", []), model.get("centroids", [])
    if not keys or not centroids:
        return {"dominant": None, "membership": {}, "shared_tags": []}
    x = [float(signals.tag_affinity.get(k, 0.0)) for k in keys]
    u = _memberships_for(x, centroids, model.get("m", 2.0))
    dom = max(range(len(centroids)), key=lambda j: u[j])
    return {"dominant": dom,
            "membership": {j: round(u[j], 4) for j in range(len(centroids))},
            "shared_tags": _shared_tags(x, centroids[dom], keys)}
