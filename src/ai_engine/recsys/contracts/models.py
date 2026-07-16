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
    """Normalized item. Supersedes the loose Qdrant payload dicts.

    This is the ONE place raw payload shape is interpreted (see
    `qdrant_store._payload_to_content`): consumers — endpoints and the dashboard —
    read these fields and never parse payloads themselves (docs/debt-payload-scatter.md D1)."""
    id: str
    content_type: ContentType = ContentType.text_item
    title: str = ""
    text: str = ""
    tags: list[Tag] = Field(default_factory=list)
    word_count: int = 0
    has_image: bool = False
    lat: Optional[float] = None
    lon: Optional[float] = None
    image_url: Optional[str] = None     # thumbnail / hero image, if the source has one
    public_url: Optional[str] = None    # canonical source record (Omeka page)
    years: list[int] = Field(default_factory=list)   # creation years (time_metadata)


# --------------------------------------------------------------------------- #
# User signals (events -> the "user model")
# --------------------------------------------------------------------------- #

class InteractionEvent(BaseModel):
    """One thing a visitor did, in a source-agnostic shape.

    Every event source (RudderStack, PostHog, Postgres) is normalized into this single
    model (see `ai_engine.recsys.adapters.rudderstack.normalize_events`), so the rest
    of the engine never sees a raw payload. The `event` string says what happened, and
    different events fill different fields (the rest stay `None`/empty):

    - **A view** (`CONTENT_VIEW_ENDED`) fills `content_id`, `dwell_seconds`, `end_reason`,
      `request_id`, and `impressions`.
    - **A survey / identify** (`SURVEY_SUBMITTED`, `IDENTIFY`) fills `survey_answers`.
    - **A search / lookup** (`CONTENT_LOOKUP`) fills `query_text` and maybe `clicked_id`.

    `build_user_signals` folds a visitor's list of these into their user model.
    """
    user_id: str = Field(description="Stable visitor id (RudderStack userId or anonymousId).")
    event: str = Field(description="What happened, e.g. CONTENT_VIEW_ENDED, SURVEY_SUBMITTED, IDENTIFY, CONTENT_LOOKUP.")
    ts: datetime = Field(description="UTC timestamp of the interaction (used for recency decay + ordering).")
    session_id: Optional[str] = Field(default=None, description="Browser/app session, for sequence grouping.")
    request_id: Optional[str] = Field(default=None, description="The rec-response id echoed back on a resulting view, joins a reward to the exact impression (its served feature vector) for bandit training.")
    content_id: Optional[str] = Field(default=None, description="The item this event is about (source prefix like 'content_1234' stripped to '1234').")
    dwell_seconds: Optional[float] = Field(default=None, description="Seconds spent on the content (explicit, or computed from a start/end pair).")
    end_reason: Optional[EndReason] = Field(default=None, description="How a view ended (next_button/link/close_button/abandon) → a completion score.")
    query_text: Optional[str] = Field(default=None, description="Search or lookup text, for CONTENT_LOOKUP events.")
    clicked_id: Optional[str] = Field(default=None, description="Item clicked from a result/rec list.")
    impressions: list[str] = Field(default_factory=list, description="Other item ids shown alongside but not engaged → treated as soft negatives.")
    survey_answers: dict = Field(default_factory=dict, description="question_id → answer (str / list for multi-select / float rating). Presurvey + personalization.")
    raw: dict = Field(default_factory=dict, description="The untransformed source payload, kept for debugging/audit.")


class EngagementScore(BaseModel):
    content_id: str
    outcome: Outcome
    strength: float          # continuous [-1, 1], NOT binary
    ts: datetime             # latest interaction ts (for recency)


class UserSignals(BaseModel):
    """The user model: everything the recommender needs about one visitor.

    Built by `ai_engine.recsys.signals.signal_builder.build_user_signals` from the
    visitor's events (plus survey demographics), and read at serve time by the
    `Recommender`. Two shapes of signal live here: **tag** signals (`tag_affinity` /
    `tag_aversion`, matched by `score_tag`) and **vector** signals (`taste_vector` /
    `recency_vector`, matched by the embedding scorers). A visitor with no positive
    engagement yet is "cold" (see `is_cold`), and recommendations lean on their survey
    tags until browsing warms the model up.
    """
    user_id: str = Field(description="The visitor this model describes.")
    positives: dict[str, float] = Field(default_factory=dict, description="content_id -> recency-decayed positive strength, for items the visitor engaged with well. Seeds the taste vector and tag affinity.")
    negatives: dict[str, float] = Field(default_factory=dict, description="content_id -> recency-decayed penalty, from disliked views and shown-but-ignored impressions (soft negatives).")
    viewed: list[str] = Field(default_factory=list, description="Every content_id the visitor has seen (any outcome). Used to exclude already-seen items from recommendations.")
    recent_views: list[str] = Field(default_factory=list, description="content_ids ordered most-recent-first, giving the model sequence awareness (the recency signal).")
    tag_affinity: dict[str, float] = Field(default_factory=dict, description="'facet:label' -> [0,1] interest weight, blended from survey answers and engaged content. The main signal score_tag matches against content tags.")
    tag_aversion: dict[str, float] = Field(default_factory=dict, description="'facet:label' -> [0,1] penalty weight, from the themes of content the visitor disliked. Applied as a negative in fusion.")
    taste_vector: Optional[Vector] = Field(default=None, description="L2-normalized centroid of liked items' embeddings (the whole-history semantic taste). None until the visitor has a positive.")
    recency_vector: Optional[Vector] = Field(default=None, description="Embedding of the most-recent viewed item, powering the 'more like what you just read' signal.")
    behavior: dict = Field(default_factory=dict, description="Engagement summary stats (n_views, completion_rate, depth, ...). Used for persona explanations, not for scoring.")
    demographics: dict = Field(default_factory=dict, description='Raw survey demographics, stored for inspection. The affinity they seed lives in tag_affinity. E.g. {"age": 60, "gender": "female", "nationality": "france", "province": "Drenthe", "personal_connection": "descendant"}.')

    @property
    def is_cold(self) -> bool:
        """True until the visitor has at least one positively-engaged item. Cold visitors
        get survey-led / diverse recommendations; warm ones get taste-vector-led ones."""
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
    """The served result: the ranked `items`, the `strategy` that produced them
    (`warm` / `cold`), and a `diagnostics` dict (pool size, generators used, filter,
    ranking mode, distractor placement) for the inspector and explanations."""
    user_id: str
    items: list[ScoredCandidate]
    strategy: str = "warm"
    diagnostics: dict = Field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Explainability: a glass-box reading of the user model, grounded in museum-
# visitor theory (Falk identity types + Pekarik experience preferences).
# --------------------------------------------------------------------------- #

class Interest(BaseModel):
    facet: str
    label: str
    weight: float                                    # affinity (or aversion) strength
    evidence: list[str] = Field(default_factory=list)  # content ids that drove it


class VisitorType(BaseModel):
    """Falk (2009) visit-identity classification of the visitor."""
    type: str                                        # Explorer | Hobbyist | Recharger | Experience-Seeker | Facilitator
    confidence: float                                # margin between top two candidate scores
    rationale: str
    scores: dict[str, float] = Field(default_factory=dict)


class PersonaExplanation(BaseModel):
    """Structured, evidence-backed explanation derived purely from UserSignals +
    content taxonomy. Deterministic; `summary` is the optional verbalized prose."""
    user_id: str
    is_cold: bool
    interests: list[Interest] = Field(default_factory=list)
    aversions: list[Interest] = Field(default_factory=list)
    engagement_style: str = "unknown"                # deep_reader|completionist|skimmer|sampler|contemplative
    experience_preference: str = "unknown"           # Pekarik: object|cognitive|introspective|social
    visitor_type: Optional[VisitorType] = None       # Falk
    trajectory: list[str] = Field(default_factory=list)  # recent thematic arc (labels, most-recent first)
    demographics: dict = Field(default_factory=dict)
    behavior: dict = Field(default_factory=dict)         # engagement stats (counts/dwell) — for data-driven prose
    summary: Optional[str] = None
