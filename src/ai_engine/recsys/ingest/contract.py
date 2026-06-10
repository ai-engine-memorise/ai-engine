"""The ingestion CONTRACT.

Any source (Omeka, CSV, JSON, another CMS) formats its content into ContentDocument.
The ingestor consumes ContentDocument and nothing else. The Qdrant payload it writes
is exactly what the recommender's QdrantContentStore reads back (tags[] + tag_labels).
"""
from __future__ import annotations
from typing import Any, Optional, Union

from pydantic import BaseModel, Field

from ..contracts.enums import ContentType
from ..contracts.models import Tag, Vector


class ContentDocument(BaseModel):
    """Source-agnostic content record ready to embed + upsert."""
    id: str
    title: str = ""
    text: str = ""
    content_type: ContentType = ContentType.text_item
    tags: list[Tag] = Field(default_factory=list)

    creator: Optional[str] = None
    locations: list[dict] = Field(default_factory=list)
    geo_metadata: Optional[dict] = None
    time_metadata: Optional[dict] = None
    files_url: list[str] = Field(default_factory=list)
    public_url: Optional[str] = None

    # what to embed; defaults to title + text
    embed_text_override: Optional[str] = None
    # any extra payload to passthrough to Qdrant
    extra: dict = Field(default_factory=dict)

    @property
    def embed_text(self) -> str:
        if self.embed_text_override is not None:
            return self.embed_text_override
        return f"{self.title}\n{self.text}".strip() if self.text else self.title

    @property
    def tag_labels(self) -> list[str]:
        return [t.key for t in self.tags]

    @property
    def image_url(self) -> Optional[str]:
        return self.files_url[0] if self.files_url else None

    @property
    def word_count(self) -> int:
        return len(self.embed_text.split())

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": str(self.id),
            "title": self.title,
            "text": self.text,
            "content_type": self.content_type.value,
            "creator": self.creator,
            "tags": [{"facet": t.facet, "label": t.label, "weight": t.weight} for t in self.tags],
            "tag_labels": self.tag_labels,
            "locations": self.locations or None,
            "geo_metadata": self.geo_metadata,
            "time_metadata": self.time_metadata,
            "files_url": self.files_url or None,
            "image_url": self.image_url,
            "public_url": self.public_url,
            "text_length_words": self.word_count,
        }
        payload.update(self.extra)
        return payload


def _point_id(doc_id: str) -> Union[int, str]:
    return int(doc_id) if str(doc_id).isdigit() else str(doc_id)


def build_point(doc: ContentDocument, vector: Vector) -> dict[str, Any]:
    """Pure: ContentDocument + vector -> a Qdrant-point dict {id, vector, payload}.

    Point id = the source id (Omeka int id), so the recommender can retrieve by id.
    """
    return {"id": _point_id(doc.id), "vector": list(vector), "payload": doc.to_payload()}
