from ai_engine.recsys.contracts.enums import ContentType
from ai_engine.recsys.contracts.models import Tag
from ai_engine.recsys.ingest import ContentDocument, build_point, from_records, ContentIngestor
from ai_engine.recsys.testing.fakes import InMemoryEmbeddingModel


class FakeQdrant:
    def __init__(self):
        self.points = []
        self.created = False
        self.dim = None
        self.indexes = []

    def collection_exists(self, name):
        return self.created

    def create_collection(self, collection_name, vectors_config):
        self.created = True
        self.dim = vectors_config.size

    def create_payload_index(self, collection, field, schema):
        self.indexes.append(field)

    def upsert(self, collection_name, points):
        self.points.extend(points)


def _doc(**kw):
    base = dict(id="101", title="Forced labour", text="workshop story",
                tags=[Tag(facet="theme_what", label="Forced Labor")])
    base.update(kw)
    return ContentDocument(**base)


def test_from_records_coerces_dicts():
    docs = list(from_records([
        {"id": "1", "title": "a", "tags": [{"facet": "medium_what", "label": "photograph"}]},
        _doc(),
    ]))
    assert all(isinstance(d, ContentDocument) for d in docs)
    assert docs[0].tag_labels == ["medium_what:photograph"]


def test_to_payload_shape():
    p = _doc(files_url=["http://x/img.jpg"]).to_payload()
    assert p["id"] == "101"
    assert p["tag_labels"] == ["theme_what:Forced Labor"]
    assert p["tags"][0] == {"facet": "theme_what", "label": "Forced Labor", "weight": 1.0}
    assert p["content_type"] == "text_item"
    assert p["image_url"] == "http://x/img.jpg"
    assert p["text_length_words"] >= 1


def test_build_point_id_and_vector():
    p = build_point(_doc(), [0.1, 0.2, 0.3])
    assert p["id"] == 101                    # numeric source id -> int point id
    assert p["vector"] == [0.1, 0.2, 0.3]
    assert build_point(_doc(id="abc"), [0.0])["id"] == "abc"   # non-numeric kept


def test_ingestor_embeds_and_upserts():
    import pytest
    pytest.importorskip("qdrant_client")  # PointStruct needed only here
    fake = FakeQdrant()
    ing = ContentIngestor(InMemoryEmbeddingModel(dim=8), fake, "c", batch_size=2)
    n = ing.ingest([
        _doc(id="101"),
        _doc(id="102", title="construction detail"),
        {"id": "201", "title": "family life", "tags": [{"facet": "theme_what", "label": "Family"}]},
    ])
    assert n == 3
    assert fake.created and fake.dim == 8
    assert "tag_labels" in fake.indexes and "locations" in fake.indexes
    assert {p.id for p in fake.points} == {101, 102, 201}
    assert all(len(p.vector) == 8 for p in fake.points)
    p201 = next(p for p in fake.points if p.id == 201)
    assert p201.payload["tag_labels"] == ["theme_what:Family"]
