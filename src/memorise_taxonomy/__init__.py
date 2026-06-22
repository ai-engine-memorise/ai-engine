"""Shared MEMORISE tag taxonomy & match normalization.

Single source of truth imported by ai-engine (survey side), omeka-tools and
content-engine (content-ingest side). See ``taxonomy`` for the implementation.
"""
from .taxonomy import (
    ALIASES,
    DEFAULT_FACET,
    FacetAssignment,
    assign_facet,
    normalize_filter_value,
    normalize_key,
    normalize_label,
    review_vocab,
    to_tag,
    to_tags,
)

__all__ = [
    "ALIASES",
    "DEFAULT_FACET",
    "FacetAssignment",
    "assign_facet",
    "normalize_filter_value",
    "normalize_key",
    "normalize_label",
    "review_vocab",
    "to_tag",
    "to_tags",
]
