"""Recommendation engine (redesign).

Pragmatic ports-and-adapters: only EventSource / ContentStore / EmbeddingModel are
Protocols (multiple impls / test fakes). Engagement, signal building, scoring and
fusion are pure functions over typed models. See docs/recsys-architecture.md.
"""
