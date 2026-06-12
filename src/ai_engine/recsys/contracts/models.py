from __future__ import annotations
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field

from .enums import ContentType, EndReason, Outcome

Vector = list[float]


# --------------------------------------------------------------------------- #
# Content structure (tags + embeddings live here)
# --------------------------------------------------------------------------- #

class Tag(BaseModel):
    """One expert tag on a piece of content. `facet` is a taxonomy dimension
    (e.g. 'theme_what', 'person_who.age_group'); `label` the value."""
    facet: str
    label: str
    weight: float = 1.0  # expert confidence

    @property
    def key(self) -> str:
        return f"{self.facet}:{self.label}"


class Content(BaseModel):
    """Normalized item. Supersedes the loose Qdrant payload dicts."""
    id: str
    content_type: ContentType = ContentType.text_item
    title: str = ""
    text: str = ""
    tags: list[Tag] = Field(default_factory=list)
    word_count: int = 0
    has_image: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None


# --------------------------------------------------------------------------- #
# User signals (events -> the "user model")
# --------------------------------------------------------------------------- #

class InteractionEvent(BaseModel):
    """Canonical event. EVERY source (RudderStack/PostHog/Postgres) normalizes to this."""
    user_id: str
    event: str                                   # CONTENT_VIEW_ENDED, CONTENT_LOOKUP, ...
    ts: datetime
    session_id: Optional[str] = None
    request_id: Optional[str] = None             # echoes the rec response -> joins reward to impression
    content_id: Optional[str] = None
    dwell_seconds: Optional[float] = None
    end_reason: Optional[EndReason] = None
    query_text: Optional[str] = None
    clicked_id: Optional[str] = None
    impressions: list[str] = Field(default_factory=list)  # shown-but-context ids (soft-neg)
    survey_answers: dict = Field(default_factory=dict)
    raw: dict = Field(default_factory=dict)


class EngagementScore(BaseModel):
    content_id: str
    outcome: Outcome
    strength: float          # continuous [-1, 1], NOT binary
    ts: datetime             # latest interaction ts (for recency)


class UserSignals(BaseModel):
    """THE USER MODEL. Everything the recommender needs about a user, derived from events."""
    user_id: str
    positives: dict[str, float] = Field(default_factory=dict)   # content_id -> decayed weight
    negatives: dict[str, float] = Field(default_factory=dict)
    viewed: list[str] = Field(default_factory=list)             # every content_id viewed (any outcome) -> full dedup
    recent_views: list[str] = Field(default_factory=list)       # ordered most-recent-first (sequence awareness)
    tag_affinity: dict[str, float] = Field(default_factory=dict)  # "facet:label" -> score
    tag_aversion: dict[str, float] = Field(default_factory=dict)  # themes of negatively-engaged content -> penalty
    taste_vector: Optional[Vector] = None                         # semantic centroid (whole history)
    recency_vector: Optional[Vector] = None                       # vector of the most-recent view (sequence signal)
    demographics: dict = Field(default_factory=dict)

    @property
    def is_cold(self) -> bool:
        return not self.positives


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #

class Candidate(BaseModel):
    content_id: str
    generated_by: str        # "semantic" | "tag" | "geo"
    base_score: float = 0.0  # generator's own score (recall signal)


class ScoredCandidate(BaseModel):
    content_id: str
    final_score: float
    breakdown: dict[str, float] = Field(default_factory=dict)  # per-scorer contribution
    features: list[float] = Field(default_factory=list)        # ordered context vector (bandit training)
    content: Optional[Content] = None
    kind: str = "recommendation"   # "recommendation" | "distractor" (novelty/exploration)


class Recommendation(BaseModel):
    user_id: str
    items: list[ScoredCandidate]
    strategy: str = "warm"
    diagnostics: dict = Field(default_factory=dict)
