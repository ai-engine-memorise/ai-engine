"""Generic source: turn arbitrary contract-shaped records into ContentDocuments.

This is the "allow a certain contract to be ingested" path — anything that already
matches the contract (dicts or ContentDocuments) can be ingested without Omeka.
Source-specific formatters (e.g. omeka_to_documents in omeka-tools) emit such dicts.
"""
from __future__ import annotations
from typing import Iterable, Iterator, Union

from .contract import ContentDocument

Record = Union[dict, ContentDocument]


def from_records(records: Iterable[Record]) -> Iterator[ContentDocument]:
    for r in records:
        if isinstance(r, ContentDocument):
            yield r
        else:
            yield ContentDocument.model_validate(r)
