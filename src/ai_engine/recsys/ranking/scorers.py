"""Pure scorers. CONTRACT: every scorer returns a value in [0, 1].

That contract is what makes the weighted sum in fusion valid without rescaling.
"""
from __future__ import annotations
import math
from typing import Optional

from ..contracts.models import Content, UserSignals, Vector


def cosine(a: Optional[Vector], b: Optional[Vector]) -> float:
    """Cosine similarity in [-1, 1]; 0 if either side is missing/zero."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def score_semantic(signals: UserSignals, candidate_vector: Optional[Vector]) -> float:
    """How close the candidate is to the user's taste vector. -> [0, 1]."""
    if signals.taste_vector is None or candidate_vector is None:
        return 0.0
    return (cosine(signals.taste_vector, candidate_vector) + 1.0) / 2.0


def score_affinity(candidate_vector: Optional[Vector], liked: list[tuple[float, Vector]]) -> float:
    """Item-kNN like signal: strength-weighted MAX cosine to ANY individually liked
    item (vs the blurred whole-history centroid in score_semantic). Sharper for
    multi-interest users — a candidate near ONE strong like scores high even if it
    is far from the centroid. `liked` = [(relative_weight in [0,1], vector)]. -> [0,1]."""
    if not candidate_vector or not liked:
        return 0.0
    best = 0.0
    for w, v in liked:
        sim = (cosine(candidate_vector, v) + 1.0) / 2.0
        best = max(best, sim * w)
    return best


def score_recency(signals: UserSignals, candidate_vector: Optional[Vector]) -> float:
    """Sequence awareness: closeness to the user's MOST-RECENT view (vs the whole-history
    taste vector). Boosts 'more like what you just read'. -> [0, 1]."""
    if signals.recency_vector is None or candidate_vector is None:
        return 0.0
    return (cosine(signals.recency_vector, candidate_vector) + 1.0) / 2.0


def score_tag(signals: UserSignals, content: Optional[Content]) -> float:
    """Affinity-weighted overlap between the user's tag affinity and the
    candidate's tags. -> [0, 1].

        score = sum_l  user_affinity[l] * cand_tag_weight[l]  /  sum_l user_affinity[l]
    """
    if content is None or not signals.tag_affinity:
        return 0.0
    # match case-insensitively: taxonomy casing (e.g. "female", "Photograph") must
    # line up with constructed keys (e.g. demographic "...:Female").
    cand_weights = {t.key.lower(): t.weight for t in content.tags}
    total = sum(signals.tag_affinity.values())
    if total <= 0:
        return 0.0
    matched = sum(
        aff * cand_weights.get(key.lower(), 0.0)
        for key, aff in signals.tag_affinity.items()
    )
    return max(0.0, min(matched / total, 1.0))


def score_aversion(signals: UserSignals, content: Optional[Content]) -> float:
    """Overlap between the candidate's tags and themes the user DISLIKED. -> [0, 1].
    Mirrors score_tag over tag_aversion; fused with a NEGATIVE weight so a candidate
    sharing themes with abandoned content is pushed down."""
    if content is None or not signals.tag_aversion:
        return 0.0
    cand_weights = {t.key.lower(): t.weight for t in content.tags}
    total = sum(signals.tag_aversion.values())
    if total <= 0:
        return 0.0
    matched = sum(
        av * cand_weights.get(key.lower(), 0.0)
        for key, av in signals.tag_aversion.items()
    )
    return max(0.0, min(matched / total, 1.0))
