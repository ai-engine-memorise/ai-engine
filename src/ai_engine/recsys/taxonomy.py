"""Re-export of the shared :mod:`memorise_taxonomy` package.

The match-key normalization is now a single source of truth in ``memorise-taxonomy``
so the survey side (here) and the content-ingest side (omeka-tools / content-engine)
can never drift. This shim keeps existing ``ai_engine.recsys.taxonomy`` imports working.
"""
from memorise_taxonomy import (  # noqa: F401
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
