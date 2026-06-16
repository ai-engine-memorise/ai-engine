"""Regression tests for ai_engine.common payload handling."""
import numpy as np

from ai_engine.common import clean_payload_field, Item


def test_clean_payload_field_passes_through_plain_python():
    # the bug: a missing fallback return turned normal list/dict payloads into None.
    assert clean_payload_field([{"a": 1}]) == [{"a": 1}]
    assert clean_payload_field({"k": "v"}) == {"k": "v"}
    assert clean_payload_field("str") == "str"
    assert clean_payload_field(None) is None


def test_clean_payload_field_converts_numpy():
    assert clean_payload_field(np.array([1, 2, 3])) == [1, 2, 3]


def test_item_preserves_locations_and_files_url():
    # locations / files_url come from Qdrant payloads as plain lists; they must survive
    # __post_init__ so image_url and geo data are not silently dropped.
    item = Item(
        id=1,
        title="t",
        public_url="u",
        locations=[{"lat": 52.7, "lon": 9.9}],
        files_url=["https://example/img.jpg"],
    )
    assert item.locations == [{"lat": 52.7, "lon": 9.9}]
    assert item.files_url == ["https://example/img.jpg"]
    assert item.image_url == "https://example/img.jpg"
