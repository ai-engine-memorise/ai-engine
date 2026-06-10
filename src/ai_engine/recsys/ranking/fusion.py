"""Pure fusion + diversity. No IO.

weighted_fuse: combine per-scorer [0,1] scores into one fused score + keep the
breakdown (explainability). mmr_rerank: greedy Maximal Marginal Relevance to avoid
returning 10 near-identical stories.
"""
from __future__ import annotations
from typing import Optional

from ..contracts.config import FusionWeights
from ..contracts.models import ScoredCandidate, Vector
from .scorers import cosine


def weighted_fuse(per_scorer: dict[str, float], weights: FusionWeights) -> tuple[float, dict[str, float]]:
    """Return (fused_score, breakdown). breakdown[s] = weight[s] * score[s]."""
    wmap = {
        "semantic": weights.semantic,
        "tag": weights.tag,
        "geo": weights.geo,
        "popularity": weights.popularity,
    }
    breakdown = {name: wmap.get(name, 0.0) * val for name, val in per_scorer.items()}
    return sum(breakdown.values()), breakdown


def mmr_rerank(
    candidates: list[ScoredCandidate],
    vectors: dict[str, Optional[Vector]],
    *,
    lambda_: float,
    limit: int,
) -> list[ScoredCandidate]:
    """Greedy MMR. Relevance = candidate.final_score; diversity = cosine between
    candidate vectors. lambda_=1 pure relevance, lambda_=0 pure diversity.

    Returns up to `limit` items. Stable: the first pick is always the top-relevance
    candidate (no selected set to penalize against yet).
    """
    pool = sorted(candidates, key=lambda c: c.final_score, reverse=True)
    selected: list[ScoredCandidate] = []
    while pool and len(selected) < limit:
        best_idx, best_val = 0, float("-inf")
        for i, cand in enumerate(pool):
            if not selected:
                mmr = cand.final_score
            else:
                max_sim = max(
                    cosine(vectors.get(cand.content_id), vectors.get(s.content_id))
                    for s in selected
                )
                mmr = lambda_ * cand.final_score - (1.0 - lambda_) * max_sim
            if mmr > best_val:
                best_idx, best_val = i, mmr
        selected.append(pool.pop(best_idx))
    return selected
