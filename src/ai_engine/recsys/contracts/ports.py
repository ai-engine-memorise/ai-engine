from __future__ import annotations
from typing import Optional, Protocol, Sequence, runtime_checkable

from .models import Content, InteractionEvent, UserSignals, Candidate, Vector


@runtime_checkable
class EmbeddingModel(Protocol):
    """Text -> vector. Real impl = fastembed; test fake = deterministic."""
    @property
    def dim(self) -> int: ...
    def encode(self, text: str) -> Vector: ...


@runtime_checkable
class EventSource(Protocol):
    """Raw user data (RudderStack/PostHog/Postgres) -> canonical events.

    The adapter is responsible for normalization, so downstream logic never sees
    source-specific shapes. In the online (path B) setup this is a Redis-backed
    hot buffer fed by the ingestion webhook; in batch it is a warehouse query.
    """
    def fetch_events(self, user_id: str) -> list[InteractionEvent]: ...


@runtime_checkable
class DemographicsProvider(Protocol):
    """Supplies a user's survey demographics (age/gender/nationality) for the
    cold-start tag bridge. Source is pluggable: Postgres visitor table, survey
    events, or a static map. Returns {} when unknown.
    """
    def get_demographics(self, user_id: str) -> dict: ...


@runtime_checkable
class UserModelStore(Protocol):
    """Materialized user model (UserSignals) for online serving.

    Path B: the ingestion webhook updates this on each event so a rec request
    is a fast read, not a rebuild. The in-memory fake / recompute-backed impl
    make this a drop-in: swap to Redis without touching the recommender.
    """
    def get_signals(self, user_id: str) -> Optional[UserSignals]: ...
    def save_signals(self, signals: UserSignals) -> None: ...


@runtime_checkable
class ImpressionStore(Protocol):
    """Short-lived store of the FEATURE VECTORS we served, keyed by request_id, so a
    later reward event (CONTENT_VIEW echoing that request_id) can be joined back to the
    exact context for an ONLINE bandit update. TTL'd; not durable (the Parquet log is)."""
    def put(self, request_id: str, features: dict) -> None: ...      # {content_id: vector}
    def get(self, request_id: str) -> dict: ...


@runtime_checkable
class ContentStore(Protocol):
    """Content structure + vectors (Qdrant). Test fake = in-memory."""
    def get(self, ids: Sequence[str]) -> dict[str, Content]: ...
    def get_vectors(self, ids: Sequence[str]) -> dict[str, Vector]: ...
    def search_vector(self, vector: Vector, *, limit: int) -> list[Candidate]: ...
    def search_tags(self, tag_keys: Sequence[str], *, limit: int) -> list[Candidate]: ...
    def sample(self, *, limit: int, exclude: Sequence[str] = ()) -> list[Candidate]: ...
    def search_filter(self, value: str, *, limit: int, exclude: Sequence[str] = ()) -> list[Candidate]: ...
    # geo radius filter — INDEPENDENT of the tag filter (composes via AND when both given)
    def search_geo(self, lat: float, lon: float, radius_m: float, *,
                   limit: int, exclude: Sequence[str] = ()) -> list[Candidate]: ...
