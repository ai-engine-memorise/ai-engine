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
from .contracts.ports import ContentStore, EventSource, UserModelStore
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


def build_components(cfg: Optional[RecConfig] = None) -> Components:
    cfg = cfg or RecConfig()
    content_store = _build_content_store()
    event_buffer, model_store = _build_stores()
    return Components(
        cfg=cfg,
        content_store=content_store,
        event_buffer=event_buffer,
        model_store=model_store,
        updater=UserModelUpdater(content_store, model_store, cfg),
        recommender=Recommender(content_store, model_store, cfg),
    )
