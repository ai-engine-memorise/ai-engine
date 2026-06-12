"""Redis-backed online stores (path B). Requires `redis` (not needed for tests).

- RedisEventBuffer: hot per-user event buffer (sorted set by ts, time-windowed).
  The ingestion webhook calls `append`; the updater reads via `fetch_events`.
- RedisUserModelStore: materialized UserSignals cache (one JSON value per user).
"""
from __future__ import annotations
import json
from datetime import datetime, timezone, timedelta
from typing import Optional

import redis  # type: ignore

from ..contracts.models import InteractionEvent, UserSignals


class RedisEventBuffer:
    def __init__(self, client: "redis.Redis", *, window_days: int = 30, key_prefix: str = "evt"):
        self.client = client
        self.window = timedelta(days=window_days)
        self.prefix = key_prefix

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}:{user_id}"

    def append(self, event: InteractionEvent) -> None:
        key = self._key(event.user_id)
        score = event.ts.timestamp()
        self.client.zadd(key, {event.model_dump_json(): score})
        cutoff = (datetime.now(timezone.utc) - self.window).timestamp()
        self.client.zremrangebyscore(key, "-inf", cutoff)

    def fetch_events(self, user_id: str) -> list[InteractionEvent]:
        raw = self.client.zrange(self._key(user_id), 0, -1)
        return [InteractionEvent.model_validate_json(r) for r in raw]


class RedisImpressionStore:
    """request_id -> {content_id: feature_vector} (JSON), TTL'd. Feeds online bandit updates."""
    def __init__(self, client: "redis.Redis", *, ttl_seconds: int = 24 * 3600, key_prefix: str = "imp"):
        self.client = client
        self.ttl = ttl_seconds
        self.prefix = key_prefix

    def put(self, request_id: str, features: dict) -> None:
        if not request_id or not features:
            return
        self.client.set(f"{self.prefix}:{request_id}", json.dumps(features), ex=self.ttl)

    def get(self, request_id: str) -> dict:
        raw = self.client.get(f"{self.prefix}:{request_id}")
        return json.loads(raw) if raw else {}


class RedisUserModelStore:
    def __init__(self, client: "redis.Redis", *, ttl_seconds: int = 7 * 24 * 3600, key_prefix: str = "umodel"):
        self.client = client
        self.ttl = ttl_seconds
        self.prefix = key_prefix

    def _key(self, user_id: str) -> str:
        return f"{self.prefix}:{user_id}"

    def get_signals(self, user_id: str) -> Optional[UserSignals]:
        raw = self.client.get(self._key(user_id))
        return UserSignals.model_validate_json(raw) if raw else None

    def save_signals(self, signals: UserSignals) -> None:
        self.client.set(self._key(signals.user_id), signals.model_dump_json(), ex=self.ttl)
