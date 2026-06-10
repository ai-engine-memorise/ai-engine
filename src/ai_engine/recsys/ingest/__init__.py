"""Content ingestion: format any source into the ContentDocument contract (step 1),
then ingest any contract-matching documents into Qdrant (step 2).

The contract (ContentDocument) is the seam: Omeka, CSV, JSON or any other source
just has to emit it; the ingestor is source-agnostic.
"""
from .contract import ContentDocument, build_point
from .ingestor import ContentIngestor
from .sources import from_records

__all__ = ["ContentDocument", "build_point", "ContentIngestor", "from_records"]
