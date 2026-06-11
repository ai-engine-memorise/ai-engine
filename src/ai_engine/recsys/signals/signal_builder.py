"""Pure construction of the USER MODEL (UserSignals) from events + content structure.

events (+ content tags/vectors) -> UserSignals. No IO: the caller fetches content
and vectors and passes them in. `now` is passed in too, so the function is fully
deterministic and testable.
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, Sequence

from ..contracts.enums import EndReason, Outcome
from ..contracts.config import RecConfig
from ..contracts.models import Content, InteractionEvent, UserSignals, Vector
from .engagement import estimate_reading_time, engagement_strength, classify_outcome

_VIEW_START = "CONTENT_VIEW_STARTED"
_VIEW_END = "CONTENT_VIEW_ENDED"


@dataclass
class ViewAggregate:
    """All views of one content folded together."""
    content_id: str
    dwell_seconds: Optional[float] = None
    visits: int = 0
    end_reason: Optional[EndReason] = None
    last_ts: Optional[datetime] = None
    survey_rating: Optional[float] = None


def aggregate_views(events: Sequence[InteractionEvent]) -> dict[str, ViewAggregate]:
    """Group events by content_id and pair start/end into dwell.

    Robust to path B (start and end arrive as separate webhook events) and to
    sources that already carry dwell_seconds on the end event.
    """
    by_content: dict[str, list[InteractionEvent]] = {}
    for e in events:
        if e.content_id is None:
            continue
        by_content.setdefault(e.content_id, []).append(e)

    out: dict[str, ViewAggregate] = {}
    for cid, evs in by_content.items():
        agg = ViewAggregate(content_id=cid)
        starts = [e for e in evs if e.event == _VIEW_START]
        ends = [e for e in evs if e.event == _VIEW_END]
        agg.visits = max(len(starts), 1)

        explicit = [e.dwell_seconds for e in evs if e.dwell_seconds is not None]
        if explicit:
            agg.dwell_seconds = max(explicit)
        elif starts and ends:
            span = max(e.ts for e in ends) - min(e.ts for e in starts)
            agg.dwell_seconds = max(span.total_seconds(), 0.0)

        if ends:
            last_end = max(ends, key=lambda e: e.ts)
            agg.end_reason = last_end.end_reason

        agg.last_ts = max(e.ts for e in evs)

        ratings = [
            float(e.survey_answers["rating"])
            for e in evs
            if isinstance(e.survey_answers, dict) and "rating" in e.survey_answers
        ]
        if ratings:
            agg.survey_rating = ratings[-1]

        out[cid] = agg
    return out


def _decay(ts: Optional[datetime], now: datetime, half_life_days: float) -> float:
    if ts is None or half_life_days <= 0:
        return 1.0
    age_days = max((now - ts).total_seconds() / 86400.0, 0.0)
    return 0.5 ** (age_days / half_life_days)


def _normalize_unit(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def build_user_signals(
    *,
    user_id: str,
    events: Sequence[InteractionEvent],
    contents: dict[str, Content],
    vectors: dict[str, Vector],
    now: datetime,
    cfg: RecConfig,
    demographics: Optional[dict] = None,
) -> UserSignals:
    """Fold events + content structure into the user model."""
    aggs = aggregate_views(events)

    positives: dict[str, float] = {}
    negatives: dict[str, float] = {}
    tag_affinity: dict[str, float] = {}

    engaged_ids = set(aggs.keys())

    for cid, agg in aggs.items():
        content = contents.get(cid)
        est = estimate_reading_time(
            content.word_count if content else 0,
            content.has_image if content else False,
            cfg,
        )
        strength = engagement_strength(
            dwell_seconds=agg.dwell_seconds,
            est_reading_time=est,
            end_reason=agg.end_reason,
            visits=agg.visits,
            survey_rating=agg.survey_rating,
            cfg=cfg,
        )
        outcome = classify_outcome(strength, cfg)
        decay = _decay(agg.last_ts, now, cfg.half_life_days)

        if outcome == Outcome.positive:
            positives[cid] = max(strength, 0.0) * decay
            if content:
                for tag in content.tags:
                    tag_affinity[tag.key] = tag_affinity.get(tag.key, 0.0) + positives[cid] * tag.weight
        elif outcome == Outcome.negative:
            negatives[cid] = abs(strength) * decay

    # soft negatives: shown in an impression set but never engaged
    for e in events:
        for imp in e.impressions:
            if imp not in engaged_ids and imp not in positives:
                pen = cfg.soft_negative_weight * _decay(e.ts, now, cfg.half_life_days)
                negatives[imp] = max(negatives.get(imp, 0.0), pen)

    # survey events (event-catalog SURVEY_*) -> demographics + person_who affinity
    from ..survey import SURVEY_EVENTS, survey_affinity, extract_demographics
    survey_demo: dict = {}
    for e in events:
        if e.event in SURVEY_EVENTS and e.survey_answers:
            survey_demo.update(extract_demographics(e.survey_answers))
            for key, w in survey_affinity(e.survey_answers).items():
                tag_affinity[key] = tag_affinity.get(key, 0.0) + w

    # explicit demographic affinity (cold-start seed; person_who facets)
    demographics = {**survey_demo, **(demographics or {})}
    if demographics:
        for key, w in _demographic_affinity(demographics).items():
            tag_affinity[key] = tag_affinity.get(key, 0.0) + w

    # taste vector = weighted centroid of positively-engaged content vectors
    taste_vector: Optional[list[float]] = None
    acc: Optional[list[float]] = None
    for cid, w in positives.items():
        v = vectors.get(cid)
        if not v:
            continue
        if acc is None:
            acc = [0.0] * len(v)
        for i, x in enumerate(v):
            acc[i] += w * x
    if acc is not None and any(acc):
        taste_vector = _normalize_unit(acc)

    # canonicalize keys to lowercase (merge case variants) so content-derived and
    # demographic-derived affinities line up regardless of taxonomy casing.
    folded: dict[str, float] = {}
    for k, v in tag_affinity.items():
        folded[k.lower()] = folded.get(k.lower(), 0.0) + v
    tag_affinity = folded

    # normalize tag affinity to [0, 1] by max
    if tag_affinity:
        mx = max(tag_affinity.values())
        if mx > 0:
            tag_affinity = {k: v / mx for k, v in tag_affinity.items()}

    return UserSignals(
        user_id=user_id,
        positives=positives,
        negatives=negatives,
        tag_affinity=tag_affinity,
        taste_vector=taste_vector,
        demographics=demographics or {},
    )


def _demographic_affinity(demographics: dict) -> dict[str, float]:
    """Map survey demographics straight onto person_who tag facets.

    This is the direct user-data <-> content-tag bridge: the taxonomy's person_who
    dimension (age_group, gender_and_age, country) mirrors the survey fields.
    """
    out: dict[str, float] = {}
    age = demographics.get("age")
    if isinstance(age, (int, float)):
        out[f"person_who.age_group:{_age_bucket(age)}"] = 0.5
    gender = demographics.get("gender")
    if gender:
        out[f"person_who.gender_and_age:{str(gender).capitalize()}"] = 0.3
    nat = demographics.get("nationality")
    if nat:
        out[f"person_who.city_village_country:From: {str(nat).capitalize()}"] = 0.4
    return out


def _age_bucket(age: float) -> str:
    if age < 18:
        return "child"
    if age <= 24:
        return "age 18-24"
    if age <= 34:
        return "age 25-34"
    if age <= 44:
        return "age 35-44"
    if age <= 54:
        return "age 45-54"
    if age <= 64:
        return "age 55-64"
    return "elderly"
