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
        return {"keys": [], "centroids": [], "profiles": [], "user_ids": []}
    keys, matrix = vectorize(warm, min_freq=min_freq)
    labels, centroids = kmeans(matrix, k, seed=seed)

    profiles = []
    for c, centroid in enumerate(centroids):
        members = [warm[i].user_id for i, lb in enumerate(labels) if lb == c]
        ranked = sorted(zip(keys, centroid), key=lambda kv: kv[1], reverse=True)
        top_tags = [{"facet": _split(k)[0], "label": _split(k)[1], "weight": round(w, 4)}
                    for k, w in ranked[:top] if w > 0]
        theme_w = [w for kk, w in zip(keys, centroid) if kk.startswith("theme_what")]
        breadth = _norm_entropy(theme_w if theme_w else centroid)
        profiles.append({
            "cluster": c, "size": len(members), "top_tags": top_tags,
            "breadth": round(breadth, 4), "falk_hint": _falk_from_breadth(breadth),
            "members": members[:50],
        })
    return {"keys": keys, "centroids": centroids, "profiles": profiles,
            "user_ids": [s.user_id for s in warm], "labels": labels}


def assign(signals: UserSignals, model: dict) -> dict:
    """Place a visitor in the nearest cluster, explained by the tags they share with it."""
    keys, centroids = model.get("keys", []), model.get("centroids", [])
    if not keys or not centroids:
        return {"cluster": None, "shared_tags": []}
    x = [float(signals.tag_affinity.get(k, 0.0)) for k in keys]
    c = min(range(len(centroids)), key=lambda i: _dist2(x, centroids[i]))
    centroid = centroids[c]
    shared = sorted(
        ((k, min(xi, ci)) for k, xi, ci in zip(keys, x, centroid) if xi > 0 and ci > 0),
        key=lambda kv: kv[1], reverse=True)
    return {"cluster": c,
            "shared_tags": [{"facet": _split(k)[0], "label": _split(k)[1]} for k, _ in shared[:5]]}
