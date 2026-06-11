"""Serving side: read the user model, match it against content structure, rank.

Reads the materialized UserSignals from the UserModelStore (path B), so a request is
a fast read + candidate scoring, not a rebuild.
"""
from __future__ import annotations
from typing import Optional

from .contracts.config import RecConfig
from .contracts.models import (
    Candidate,
    Recommendation,
    ScoredCandidate,
    UserSignals,
)
from .contracts.ports import ContentStore, UserModelStore
from .ranking.scorers import score_semantic, score_tag
from .ranking.fusion import weighted_fuse, mmr_rerank


class Recommender:
    def __init__(self, content_store: ContentStore, model_store: UserModelStore, cfg: RecConfig):
        self.content_store = content_store
        self.model_store = model_store
        self.cfg = cfg

    def recommend(self, user_id: str) -> Recommendation:
        signals = self.model_store.get_signals(user_id)
        if signals is None:
            return Recommendation(user_id=user_id, items=[], strategy="cold",
                                  diagnostics={"reason": "no_user_model"})
        return self.recommend_for_signals(signals)

    # exposed for tests / batch use without a store
    def recommend_for_signals(self, signals: UserSignals) -> Recommendation:
        cfg = self.cfg
        strategy = "warm" if not signals.is_cold else "cold"

        # 1) candidate generation: semantic (taste vector) + tag (affinity)
        seen = set(signals.positives) | set(signals.negatives)
        pool: dict[str, Candidate] = {}

        if signals.taste_vector is not None:
            for c in self.content_store.search_vector(signals.taste_vector, limit=cfg.pool_per_generator):
                pool.setdefault(c.content_id, c)

        top_tag_keys = [
            k for k, _ in sorted(signals.tag_affinity.items(), key=lambda kv: kv[1], reverse=True)
        ][:20]
        if top_tag_keys:
            for c in self.content_store.search_tags(top_tag_keys, limit=cfg.pool_per_generator):
                pool.setdefault(c.content_id, Candidate(content_id=c.content_id, generated_by="tag",
                                                       base_score=c.base_score))

        candidate_ids = [cid for cid in pool if cid not in seen]
        if not candidate_ids:
            return Recommendation(user_id=signals.user_id, items=[], strategy=strategy,
                                  diagnostics={"reason": "empty_pool", "pool_size": len(pool)})

        # 2) fetch content structure for the pool, score each
        contents = self.content_store.get(candidate_ids)
        vectors = self.content_store.get_vectors(candidate_ids)

        scored: list[ScoredCandidate] = []
        for cid in candidate_ids:
            content = contents.get(cid)
            sem = score_semantic(signals, vectors.get(cid))
            tag = score_tag(signals, content)
            fused, breakdown = weighted_fuse({"semantic": sem, "tag": tag}, cfg.fusion)
            scored.append(ScoredCandidate(
                content_id=cid, final_score=fused, breakdown=breakdown, content=content,
            ))

        # 3) diversity-aware rerank (reserve one slot for the distractor if enabled)
        rel_limit = cfg.final_limit - 1 if cfg.distractor_enabled else cfg.final_limit
        ranked = mmr_rerank(scored, vectors, lambda_=cfg.mmr_lambda, limit=max(rel_limit, 1))

        diagnostics = {"pool_size": len(pool), "scored": len(scored),
                       "generators": ["semantic", "tag"]}

        # 4) inject a labelled distractor (novelty / exploration) at a fixed slot
        if cfg.distractor_enabled:
            exclude = seen | {r.content_id for r in ranked}
            distractor = self._distractor(signals, exclude)
            if distractor is not None:
                slot = min(cfg.distractor_slot, len(ranked))
                ranked.insert(slot, distractor)
                diagnostics["distractor"] = {
                    "content_id": distractor.content_id,
                    "strategy": cfg.distractor_strategy, "slot": slot,
                }

        return Recommendation(user_id=signals.user_id, items=ranked, strategy=strategy,
                              diagnostics=diagnostics)

    # ----- distractor (novelty) ------------------------------------------- #

    def _distractor(self, signals: UserSignals, exclude: set) -> Optional[ScoredCandidate]:
        cid = self._pick_distractor_id(signals, exclude)
        if not cid:
            return None
        content = self.content_store.get([cid]).get(cid)
        vec = self.content_store.get_vectors([cid]).get(cid)
        sem = score_semantic(signals, vec)
        tag = score_tag(signals, content)
        fused, breakdown = weighted_fuse({"semantic": sem, "tag": tag}, self.cfg.fusion)
        return ScoredCandidate(content_id=cid, final_score=fused, breakdown=breakdown,
                               content=content, kind="distractor")

    def _pick_distractor_id(self, signals: UserSignals, exclude: set) -> Optional[str]:
        cfg, store = self.cfg, self.content_store
        strat = cfg.distractor_strategy

        if strat == "max_dissimilar" and signals.taste_vector:
            neg = [-x for x in signals.taste_vector]  # opposite of the taste
            for c in store.search_vector(neg, limit=cfg.pool_per_generator):
                if c.content_id not in exclude:
                    return c.content_id

        if strat == "unexplored_theme":
            cands = [c for c in store.sample(limit=20, exclude=tuple(exclude))
                     if c.content_id not in exclude]
            if cands:
                contents = store.get([c.content_id for c in cands])
                def overlap(cid: str) -> int:
                    ct = contents.get(cid)
                    keys = {t.key.lower() for t in ct.tags} if ct else set()
                    return sum(1 for k in signals.tag_affinity if k in keys)
                return min(cands, key=lambda c: overlap(c.content_id)).content_id

        # random (also the fallback for cold / empty results)
        s = store.sample(limit=1, exclude=tuple(exclude))
        return s[0].content_id if s else None
