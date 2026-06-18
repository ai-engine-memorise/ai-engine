"""Pure engagement scoring. No IO. Input = plain numbers, output = float/enum.

These functions are the easiest thing to validate: feed known numbers, assert the
behavior the design promises (longer dwell -> higher, abandon -> negative, ...).
"""
from __future__ import annotations
import math
from typing import Optional

from ..contracts.enums import EndReason, Outcome
from ..contracts.config import RecConfig

# How "finishing" a content maps to a completion signal in [-1, 1].
_COMPLETION = {
    EndReason.next_button: 1.0,
    EndReason.link: 0.6,
    EndReason.close_button: 0.0,
    EndReason.abandon: -0.5,
}


def estimate_reading_time(word_count: int, has_image: bool, cfg: RecConfig) -> float:
    """Seconds a typical visitor needs to consume this content."""
    base = word_count / cfg.reading_speed_wps if cfg.reading_speed_wps > 0 else 0.0
    if has_image:
        base += cfg.img_extra_time
    return base


def _dwell_ratio(dwell_seconds: Optional[float], est_reading_time: float, cfg: RecConfig) -> float:
    """How much of the expected reading time the user actually spent, in [0, 1]."""
    if dwell_seconds is None or dwell_seconds <= 0:
        return 0.0
    if est_reading_time <= 0:
        return 1.0  # no estimate but they spent time -> treat as fully read
    raw = dwell_seconds / est_reading_time
    return min(raw, cfg.dwell_cap_ratio) / cfg.dwell_cap_ratio


def _revisit(visits: int) -> float:
    """Saturating bonus for repeated visits, in [0, 1)."""
    visits = max(visits, 0)
    return 1.0 - math.exp(-visits / 2.0)


def _survey(rating: Optional[float]) -> float:
    """1..5 Likert -> [-1, 1]; 0 if no rating."""
    if rating is None:
        return 0.0
    return (rating - 3.0) / 2.0


def engagement_strength(
    *,
    dwell_seconds: Optional[float],
    est_reading_time: float,
    end_reason: Optional[EndReason],
    visits: int,
    survey_rating: Optional[float],
    cfg: RecConfig,
) -> float:
    """Continuous engagement in roughly [-1, 1]. Weighted blend of behavioral signals."""
    w = cfg.engagement
    completion = _COMPLETION.get(end_reason, 0.0)
    strength = (
        w.dwell * _dwell_ratio(dwell_seconds, est_reading_time, cfg)
        + w.completion * completion
        + w.revisit * _revisit(visits)
        + w.survey * _survey(survey_rating)
    )
    return strength


def classify_outcome(strength: float, cfg: RecConfig) -> Outcome:
    if strength >= cfg.positive_threshold:
        return Outcome.positive
    if strength <= cfg.negative_threshold:
        return Outcome.negative
    return Outcome.neutral
