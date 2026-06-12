"""Composition root: assemble the recsys components from environment.

If REDIS_URL / QDRANT_API_URL are set, use the real adapters; otherwise fall back
to in-memory fakes (with dev fixtures) so the service runs locally with no infra.
This is the ONE place IO backends are chosen — everything else takes ports.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from .contracts.config import RecConfig
from .contracts.ports import ContentStore, EventSource, UserModelStore, DemographicsProvider
from .recommender import Recommender
from .updater import UserModelUpdater


@dataclass
class Components:
    cfg: RecConfig
    content_store: ContentStore
    event_buffer: EventSource          # also supports .append() (Redis buffer / fake)
    model_store: UserModelStore
    updater: UserModelUpdater
    recommender: Recommender
    demographics: DemographicsProvider
    event_log: object   # durable append-only log (Parquet) — .append(events)


def _build_content_store() -> ContentStore:
    url = os.getenv("QDRANT_API_URL")
    if url:
        from qdrant_client import QdrantClient
        from .adapters.qdrant_store import QdrantContentStore
        client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        return QdrantContentStore(client, os.getenv("COLLECTION_NAME", "omeka-items"))
    # dev fallback: the hand-built fixture world
    from .testing.fakes import FakeContentStore
    from .testing.fixtures import make_contents_and_vectors
    contents, vectors = make_contents_and_vectors()
    return FakeContentStore(contents, vectors)


def _build_stores():
    url = os.getenv("REDIS_URL")
    if url:
        import redis
        from .adapters.redis_store import RedisEventBuffer, RedisUserModelStore
        client = redis.from_url(url, decode_responses=True)
        return RedisEventBuffer(client), RedisUserModelStore(client)
    from .testing.fakes import FakeEventSource, InMemoryUserModelStore
    return FakeEventSource(), InMemoryUserModelStore()


def _build_demographics() -> DemographicsProvider:
    if os.getenv("DB_NAME"):
        from .adapters.demographics import PostgresDemographicsProvider
        return PostgresDemographicsProvider()
    from .adapters.demographics import NullDemographicsProvider
    return NullDemographicsProvider()


def _build_event_log():
    d = os.getenv("EVENT_LOG_DIR")
    if d:
        from .adapters.event_log import ParquetEventLog
        return ParquetEventLog(d)
    from .adapters.event_log import NullEventLog
    return NullEventLog()


def _build_config() -> RecConfig:
    """RecConfig from defaults, overridden by env so weights/limits are tunable
    WITHOUT a redeploy (a stop-gap until a learned ranker owns the weights).

        RECSYS_W_{SEMANTIC,AFFINITY,TAG,RECENCY,AVERSION}   fusion weights (float)
        RECSYS_MMR_LAMBDA                                    relevance<->diversity
        RECSYS_FINAL_LIMIT                                   items returned (int)
        RECSYS_DISTRACTOR_PROBABILITY                        0..1
    """
    cfg = RecConfig()

    def _f(name: str, cur: float) -> float:
        v = os.getenv(name)
        try:
            return float(v) if v is not None else cur
        except ValueError:
            return cur

    f = cfg.fusion
    f.semantic = _f("RECSYS_W_SEMANTIC", f.semantic)
    f.affinity = _f("RECSYS_W_AFFINITY", f.affinity)
    f.tag = _f("RECSYS_W_TAG", f.tag)
    f.recency = _f("RECSYS_W_RECENCY", f.recency)
    f.aversion = _f("RECSYS_W_AVERSION", f.aversion)
    cfg.mmr_lambda = _f("RECSYS_MMR_LAMBDA", cfg.mmr_lambda)
    cfg.distractor_probability = _f("RECSYS_DISTRACTOR_PROBABILITY", cfg.distractor_probability)
    fl = os.getenv("RECSYS_FINAL_LIMIT")
    if fl is not None:
        try:
            cfg.final_limit = int(fl)
        except ValueError:
            pass
    return cfg


def build_components(cfg: Optional[RecConfig] = None) -> Components:
    cfg = cfg or _build_config()
    content_store = _build_content_store()
    event_buffer, model_store = _build_stores()
    return Components(
        cfg=cfg,
        content_store=content_store,
        event_buffer=event_buffer,
        model_store=model_store,
        updater=UserModelUpdater(content_store, model_store, cfg),
        recommender=Recommender(content_store, model_store, cfg),
        demographics=_build_demographics(),
        event_log=_build_event_log(),
    )
