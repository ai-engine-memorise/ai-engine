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


def score_tag(signals: UserSignals, content: Optional[Content]) -> float:
    """Affinity-weighted overlap between the user's tag affinity and the
    candidate's tags. -> [0, 1].

        score = sum_l  user_affinity[l] * cand_tag_weight[l]  /  sum_l user_affinity[l]
    """
    if content is None or not signals.tag_affinity:
        return 0.0
    cand_weights = {t.key: t.weight for t in content.tags}
    total = sum(signals.tag_affinity.values())
    if total <= 0:
        return 0.0
    matched = sum(
        aff * cand_weights.get(key, 0.0)
        for key, aff in signals.tag_affinity.items()
    )
    return max(0.0, min(matched / total, 1.0))
