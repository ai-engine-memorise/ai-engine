"""In-memory implementations of the three ports + the user-model store.

These let the whole pipeline run offline, deterministic, no Qdrant / Redis / network.
Each satisfies the corresponding Protocol in contracts.ports.
"""
from __future__ import annotations
import hashlib
from typing import Optional, Sequence

from ..contracts.models import Content, InteractionEvent, UserSignals, Candidate, Vector
from ..ranking.scorers import cosine, haversine_m


class FakeContentStore:
    """ContentStore backed by dicts. search_vector = brute-force cosine;
    search_tags = tag-key overlap."""

    def __init__(self, contents: dict[str, Content], vectors: dict[str, Vector]):
        self._contents = contents
        self._vectors = vectors

    def get(self, ids: Sequence[str]) -> dict[str, Content]:
        return {i: self._contents[i] for i in ids if i in self._contents}

    def get_vectors(self, ids: Sequence[str]) -> dict[str, Vector]:
        return {i: self._vectors[i] for i in ids if i in self._vectors}

    def search_vector(self, vector: Vector, *, limit: int) -> list[Candidate]:
        scored = [
            (cid, cosine(vector, v)) for cid, v in self._vectors.items()
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [
            Candidate(content_id=cid, generated_by="semantic", base_score=s)
            for cid, s in scored[:limit]
        ]

    def search_tags(self, tag_keys: Sequence[str], *, limit: int) -> list[Candidate]:
        # case-insensitive, mirroring Qdrant's lowercased tag_labels index
        wanted = {k.lower() for k in tag_keys}
        hits = []
        for cid, content in self._contents.items():
            overlap = sum(1 for t in content.tags if t.key.lower() in wanted)
            if overlap > 0:
                hits.append((cid, float(overlap)))
        hits.sort(key=lambda t: t[1], reverse=True)
        return [
            Candidate(content_id=cid, generated_by="tag", base_score=s)
            for cid, s in hits[:limit]
        ]

    def sample(self, *, limit: int, exclude=()) -> list[Candidate]:
        # deterministic (sorted) for test stability
        ex = {str(e) for e in exclude}
        ids = [cid for cid in sorted(self._contents) if cid not in ex]
        return [Candidate(content_id=cid, generated_by="distractor") for cid in ids[:limit]]

    def search_filter(self, value: str, *, limit: int, exclude=()) -> list[Candidate]:
        ex = {str(e) for e in exclude}
        v = value.lower()
        hits = [cid for cid, c in self._contents.items()
                if cid not in ex and any(t.label.lower() == v for t in c.tags)]
        return [Candidate(content_id=cid, generated_by="filter") for cid in sorted(hits)[:limit]]

    def search_geo(self, lat: float, lon: float, radius_m: float, *, limit: int, exclude=()) -> list[Candidate]:
        ex = {str(e) for e in exclude}
        hits = []
        for cid, c in self._contents.items():
            if cid in ex or c.lat is None or c.lon is None:
                continue
            d = haversine_m(lat, lon, c.lat, c.lon)
            if d <= radius_m:
                hits.append((cid, d))
        hits.sort(key=lambda t: t[1])   # nearest first
        return [Candidate(content_id=cid, generated_by="geo", base_score=-d) for cid, d in hits[:limit]]


class FakeEventSource:
    """EventSource backed by an in-memory per-user buffer (mimics the Redis hot
    buffer fed by the ingestion webhook)."""

    def __init__(self, events_by_user: Optional[dict[str, list[InteractionEvent]]] = None):
        self._events: dict[str, list[InteractionEvent]] = events_by_user or {}

    def append(self, event: InteractionEvent) -> None:
        self._events.setdefault(event.user_id, []).append(event)

    def fetch_events(self, user_id: str) -> list[InteractionEvent]:
        return list(self._events.get(user_id, []))


class InMemoryImpressionStore:
    """ImpressionStore backed by a dict (no TTL needed for tests)."""
    def __init__(self) -> None:
        self._d: dict[str, dict] = {}

    def put(self, request_id: str, features: dict) -> None:
        if request_id and features:
            self._d[request_id] = dict(features)

    def get(self, request_id: str) -> dict:
        return dict(self._d.get(request_id, {}))

    def consume(self, request_id: str, content_id: str) -> None:
        d = self._d.get(request_id)
        if d and content_id in d:
            del d[content_id]
            if not d:
                self._d.pop(request_id, None)


class InMemoryUserModelStore:
    """UserModelStore backed by a dict (recompute-backed equivalent of Redis)."""

    def __init__(self) -> None:
        self._signals: dict[str, UserSignals] = {}

    def get_signals(self, user_id: str) -> Optional[UserSignals]:
        return self._signals.get(user_id)

    def save_signals(self, signals: UserSignals) -> None:
        self._signals[signals.user_id] = signals

    def iter_signals(self) -> list:
        return list(self._signals.values())


class InMemoryEmbeddingModel:
    """Deterministic text -> vector (hash buckets). For cold-start / profile paths
    in tests; no model download."""

    def __init__(self, dim: int = 8):
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def encode(self, text: str) -> Vector:
        vec = [0.0] * self._dim
        for tok in (text or "").lower().split():
            h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = sum(x * x for x in vec) ** 0.5
        return [x / norm for x in vec] if norm else vec
