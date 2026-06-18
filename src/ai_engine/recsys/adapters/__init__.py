"""Adapters (IO). Only the pure RudderStack normalizer is exported here; the
infra-bound stores (Redis/Qdrant/fastembed) are imported directly from their
modules so importing this package doesn't require those libraries installed.
"""
from .rudderstack import normalize_event, normalize_events, normalize_content_id

__all__ = ["normalize_event", "normalize_events", "normalize_content_id"]
