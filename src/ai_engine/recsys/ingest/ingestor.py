"""The generic ingestor (step 2): ContentDocuments -> embed -> upsert Qdrant.

Source-agnostic. Takes an EmbeddingModel port (real fastembed or test fake) and a
Qdrant client. Writes the payload the recommender reads (tags[] + tag_labels) and
creates the indexes tag search / geo search rely on.
"""
from __future__ import annotations
import logging
from typing import Iterable

from ..contracts.ports import EmbeddingModel
from .contract import ContentDocument, build_point
from .sources import from_records

logger = logging.getLogger(__name__)


class ContentIngestor:
    def __init__(self, embedder: EmbeddingModel, client, collection: str, *, batch_size: int = 64):
        self.embedder = embedder
        self.client = client
        self.collection = collection
        self.batch_size = batch_size

    def ensure_collection(self) -> None:
        from qdrant_client.models import VectorParams, Distance
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=self.embedder.dim, distance=Distance.COSINE),
            )
            logger.info(f"Created collection '{self.collection}' (dim={self.embedder.dim})")

    def ensure_indexes(self) -> None:
        from qdrant_client.models import PayloadSchemaType
        # tag recall + geo search depend on these. Best-effort (ignore 'already exists').
        for field, schema in [
            ("tag_labels", PayloadSchemaType.KEYWORD),
            ("locations", PayloadSchemaType.GEO),
        ]:
            try:
                self.client.create_payload_index(self.collection, field, schema)
            except Exception as e:  # noqa: BLE001
                logger.debug(f"index {field}: {e}")

    def ingest(self, documents: Iterable[ContentDocument | dict]) -> int:
        """Embed + upsert. Accepts ContentDocuments or contract-shaped dicts."""
        self.ensure_collection()
        self.ensure_indexes()

        count = 0
        batch: list[ContentDocument] = []
        for doc in from_records(documents):
            batch.append(doc)
            if len(batch) >= self.batch_size:
                count += self._flush(batch)
                batch = []
        if batch:
            count += self._flush(batch)
        logger.info(f"Ingested {count} documents into '{self.collection}'")
        return count

    def _flush(self, batch: list[ContentDocument]) -> int:
        from qdrant_client.models import PointStruct
        points = [
            PointStruct(**build_point(doc, self.embedder.encode(doc.embed_text)))
            for doc in batch
        ]
        self.client.upsert(collection_name=self.collection, points=points)
        return len(points)
