"""Qdrant-backed ContentStore. Requires `qdrant-client` (not needed for tests).

Tags live in the point payload (decided design): a `tags` list of {facet,label,weight}
plus a flat `tag_labels` ("facet:label") KEYWORD-indexed field used for tag recall.
"""
from __future__ import annotations
from typing import Optional, Sequence

from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Filter, FieldCondition, MatchAny  # type: ignore

from ..contracts.enums import ContentType
from ..contracts.models import Content, Tag, Candidate, Vector


def _payload_to_content(point_id, payload: dict) -> Content:
    payload = payload or {}
    tags = [
        Tag(facet=t.get("facet", "unknown"), label=t.get("label", ""), weight=float(t.get("weight", 1.0)))
        for t in (payload.get("tags") or [])
    ]
    files = payload.get("files_url") or []
    has_image = bool(payload.get("image_url") or (isinstance(files, list) and files))
    ctype = payload.get("content_type", ContentType.text_item.value)
    try:
        ctype = ContentType(ctype)
    except ValueError:
        ctype = ContentType.text_item
    return Content(
        id=str(point_id),
        content_type=ctype,
        title=payload.get("title") or "",
        text=payload.get("text") or "",
        tags=tags,
        word_count=int(payload.get("text_length_words") or payload.get("word_count") or 0),
        has_image=has_image,
    )


class QdrantContentStore:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    @staticmethod
    def _pid(content_id: str):
        return int(content_id) if str(content_id).isdigit() else content_id

    def get(self, ids: Sequence[str]) -> dict[str, Content]:
        if not ids:
            return {}
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._pid(i) for i in ids],
            with_payload=True, with_vectors=False,
        )
        return {str(p.id): _payload_to_content(p.id, p.payload) for p in res}

    def get_vectors(self, ids: Sequence[str]) -> dict[str, Vector]:
        if not ids:
            return {}
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._pid(i) for i in ids],
            with_payload=False, with_vectors=True,
        )
        out: dict[str, Vector] = {}
        for p in res:
            v = p.vector
            if isinstance(v, dict):  # named vectors -> take first
                v = next(iter(v.values()))
            out[str(p.id)] = v
        return out

    def search_vector(self, vector: Vector, *, limit: int) -> list[Candidate]:
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=vector, limit=limit, with_payload=False, with_vectors=False,
        )
        return [
            Candidate(content_id=str(p.id), generated_by="semantic", base_score=p.score or 0.0)
            for p in res.points
        ]

    def search_tags(self, tag_keys: Sequence[str], *, limit: int) -> list[Candidate]:
        if not tag_keys:
            return []
        flt = Filter(should=[FieldCondition(key="tag_labels", match=MatchAny(any=list(tag_keys)))])
        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=flt, limit=limit, with_payload=False, with_vectors=False,
        )
        return [Candidate(content_id=str(p.id), generated_by="tag", base_score=0.0) for p in points]
