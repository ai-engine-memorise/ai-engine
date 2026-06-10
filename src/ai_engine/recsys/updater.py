"""Ingestion side (path B): events -> user model -> store.

The webhook appends each RudderStack event to the EventSource buffer, then calls
`refresh` to rebuild the materialized UserSignals and save it. Rebuild-from-buffer
(rather than fragile true-incremental decay math) keeps build_user_signals as the
single source of truth, while staying fast (hot buffer read + pure fold).
"""
from __future__ import annotations
from datetime import datetime
from typing import Optional, Sequence

from .contracts.config import RecConfig
from .contracts.models import InteractionEvent, UserSignals
from .contracts.ports import ContentStore, EventSource, UserModelStore
from .signals.signal_builder import build_user_signals, aggregate_views


class UserModelUpdater:
    def __init__(self, content_store: ContentStore, model_store: UserModelStore, cfg: RecConfig):
        self.content_store = content_store
        self.model_store = model_store
        self.cfg = cfg

    def build(
        self,
        user_id: str,
        events: Sequence[InteractionEvent],
        *,
        now: datetime,
        demographics: Optional[dict] = None,
    ) -> UserSignals:
        """Fold events into the user model (fetching only the content they touched)."""
        engaged_ids = list(aggregate_views(events).keys())
        contents = self.content_store.get(engaged_ids) if engaged_ids else {}
        vectors = self.content_store.get_vectors(engaged_ids) if engaged_ids else {}
        return build_user_signals(
            user_id=user_id,
            events=events,
            contents=contents,
            vectors=vectors,
            now=now,
            cfg=self.cfg,
            demographics=demographics,
        )

    def refresh(
        self,
        user_id: str,
        source: EventSource,
        *,
        now: datetime,
        demographics: Optional[dict] = None,
    ) -> UserSignals:
        """Pull the user's recent events from the hot buffer, rebuild, persist."""
        events = source.fetch_events(user_id)
        signals = self.build(user_id, events, now=now, demographics=demographics)
        self.model_store.save_signals(signals)
        return signals
