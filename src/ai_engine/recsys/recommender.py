"""Serving side: read the user model, match it against content structure, rank.

Reads the materialized UserSignals from the UserModelStore (path B), so a request is
a fast read + candidate scoring, not a rebuild.
"""
from __future__ import annotations
import random
from typing import Optional

from .contracts.config import RecConfig
from .contracts.models import (
    Candidate,
    Recommendation,
    ScoredCandidate,
    UserSignals,
)
from .contracts.ports import ContentStore, UserModelStore
from .ranking.scorers import score_semantic, score_affinity, score_tag, score_recency, score_aversion
from .ranking.fusion import weighted_fuse, mmr_rerank


class Recommender:
    def __init__(self, content_store: ContentStore, model_store: UserModelStore, cfg: RecConfig):
        self.content_store = content_store
        self.model_store = model_store
        self.cfg = cfg

    def recommend(self, user_id: str, *, filter: Optional[str] = None) -> Recommendation:
        # no stored model -> empty persona -> cold-start diverse fallback (never empty-handed)
        signals = self.model_store.get_signals(user_id) or UserSignals(user_id=user_id)
        return self.recommend_for_signals(signals, filter=filter)

    # exposed for tests / batch use without a store
    def recommend_for_signals(self, signals: UserSignals, *, filter: Optional[str] = None) -> Recommendation:
        cfg = self.cfg
        strategy = "warm" if not signals.is_cold else "cold"

        # exclude everything already viewed (any outcome) — full view-history dedup
        seen = set(signals.positives) | set(signals.negatives) | set(signals.viewed)
        pool: dict[str, Candidate] = {}

        # 1) candidate generation
        if filter:
            # filtered to one tag (e.g. a location): the candidate set IS the filtered content
            for c in self.content_store.search_filter(filter, limit=cfg.pool_per_generator, exclude=tuple(seen)):
                pool.setdefault(c.content_id, c)
            generators = ["filter"]
        else:
            generators = ["semantic", "tag"]
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

        # cold-start fallback (no signal) — global diverse sample. NOT for filtered queries
        # (a filter must never leak content from outside it).
        cold_fallback = False
        if not candidate_ids and not filter:
            for c in self.content_store.sample(limit=cfg.pool_per_generator, exclude=tuple(seen)):
                pool.setdefault(c.content_id, c)
            candidate_ids = [cid for cid in pool if cid not in seen]
            cold_fallback = True
            strategy = "cold"

        if not candidate_ids:   # empty filter / catalogue empty / all seen
            return Recommendation(user_id=signals.user_id, items=[], strategy=strategy,
                                  diagnostics={"reason": "empty_filter" if filter else "no_content",
                                               "filter": filter, "pool_size": 0})

        # 2) fetch content structure for the pool, score each
        contents = self.content_store.get(candidate_ids)
        vectors = self.content_store.get_vectors(candidate_ids)

        # item-kNN like signal: fetch the liked items' vectors once (relative weights).
        # kept out of the stored user model so it doesn't grow with #likes — read at serve.
        liked: list[tuple[float, list[float]]] = []
        if signals.positives:
            lv = self.content_store.get_vectors(list(signals.positives))
            mx = max(signals.positives.values()) or 1.0
            liked = [(signals.positives[cid] / mx, lv[cid]) for cid in signals.positives if cid in lv]

        scored: list[ScoredCandidate] = []
        for cid in candidate_ids:
            content = contents.get(cid)
            vec = vectors.get(cid)
            sem = score_semantic(signals, vec)
            aff = score_affinity(vec, liked)     # max-sim to any one liked item (sharp)
            tag = score_tag(signals, content)
            rec = score_recency(signals, vec)   # sequence: closeness to most-recent view
            av = score_aversion(signals, content)   # penalty for disliked themes (negative weight)
            fused, breakdown = weighted_fuse(
                {"semantic": sem, "affinity": aff, "tag": tag, "recency": rec, "aversion": av}, cfg.fusion)
            scored.append(ScoredCandidate(
                content_id=cid, final_score=fused, breakdown=breakdown, content=content,
            ))

        # 3) diversity-aware rerank — decide the distractor first (so we size the list right)
        inject = cfg.distractor_enabled and (
            cfg.distractor_probability >= 1.0 or random.random() < cfg.distractor_probability)
        rel_limit = cfg.final_limit - (1 if inject else 0)
        ranked = mmr_rerank(scored, vectors, lambda_=cfg.mmr_lambda, limit=max(rel_limit, 1))

        diagnostics = {"pool_size": len(pool), "scored": len(scored),
                       "generators": generators, "cold_start_fallback": cold_fallback,
                       "filter": filter}

        # 4) inject a labelled distractor (novelty / exploration) at slot 3 or 4 (random)
        if inject:
            ranked_ids = {r.content_id for r in ranked}
            if filter:
                # within-filter distractor: lowest-relevance item from the SAME filtered set
                leftover = [s for s in scored if s.content_id not in ranked_ids]
                distractor = min(leftover, key=lambda s: s.final_score) if leftover else None
                if distractor is not None:
                    distractor = distractor.model_copy(update={"kind": "distractor"})
            else:
                distractor = self._distractor(signals, seen | ranked_ids)
            if distractor is not None:
                slot = min(random.choice(cfg.distractor_slots), len(ranked))
                ranked.insert(slot, distractor)
                diagnostics["distractor"] = {
                    "content_id": distractor.content_id,
                    "strategy": "within_filter" if filter else cfg.distractor_strategy, "slot": slot,
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
