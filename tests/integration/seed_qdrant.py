"""Seed a tiny TAGGED Qdrant collection for integration tests.

Doubles as a mini content-ingestion: it writes exactly the payload shape the real
ingestion will produce (tags[] + tag_labels) so tag-matching is exercised for real.

    python tests/integration/seed_qdrant.py            # uses QDRANT_API_URL or localhost
"""
from __future__ import annotations
import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, PointStruct, PayloadSchemaType,
)

DIM = 8
COLLECTION = os.getenv("COLLECTION_NAME", "recsys_itest")


def _axis(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


# id -> (vector, theme label, axis)
POINTS = [
    (101, _axis(0), "Forced Labor", "workshops"),
    (102, _axis(0), "Forced Labor", "work detail"),
    (103, _axis(0), "Forced Labor", "work detail"),
    (201, _axis(1), "Family", "children"),
    (202, _axis(1), "Family", "family life"),
]


def _payload(theme: str, sub: str) -> dict:
    tags = [
        {"facet": "theme_what", "label": theme, "weight": 1.0},
        {"facet": "theme_what", "label": sub, "weight": 0.8},
    ]
    return {
        "title": f"{theme}: {sub}",
        "text": f"{theme} story about {sub}",
        "content_type": "text_item",
        "text_length_words": 120,
        "tags": tags,
        "tag_labels": [f"{t['facet']}:{t['label']}" for t in tags],
    }


def seed(client: QdrantClient, collection: str = COLLECTION) -> None:
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection_name=collection,
        vectors_config=VectorParams(size=DIM, distance=Distance.COSINE),
    )
    client.create_payload_index(collection, "tag_labels", PayloadSchemaType.KEYWORD)
    client.upsert(
        collection_name=collection,
        points=[
            PointStruct(id=pid, vector=vec, payload=_payload(theme, sub))
            for pid, vec, theme, sub in POINTS
        ],
    )


def main() -> None:
    client = QdrantClient(
        url=os.getenv("QDRANT_API_URL", "http://localhost:6333"),
        api_key=os.getenv("QDRANT_API_KEY"),
    )
    seed(client)
    print(f"seeded {len(POINTS)} points into '{COLLECTION}'")


if __name__ == "__main__":
    main()
