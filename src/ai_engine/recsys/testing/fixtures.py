"""A tiny, hand-built Bergen-Belsen content world with KNOWN structure, so synthetic
scenarios have a predictable right answer.

Three semantic clusters on orthogonal axes (6-dim vectors), each with matching
expert tags from the real taxonomy (theme_what facet):

    A = Forced Labor      axis 0
    B = Family / Children axis 1
    C = Liberation        axis 2
"""
from __future__ import annotations
from datetime import datetime, timedelta

from ..contracts.enums import ContentType
from ..contracts.models import Content, Tag, InteractionEvent
from .fakes import FakeContentStore


def _c(id, title, axis, theme_label, sub, latlon=None, years=None):
    vec = [0.0] * 6
    vec[axis] = 1.0
    lat, lon = latlon if latlon else (None, None)
    payload = {"title": title, "text": title,
               "tags": [{"facet": "theme_what", "label": theme_label}]}
    if latlon:
        payload["location"] = {"lat": lat, "lon": lon}
    if years:
        payload["time_metadata"] = {"dates_of_creation": [str(y) for y in years]}
    return Content(
        id=id,
        content_type=ContentType.text_item,
        title=title,
        text=title,
        word_count=120,
        has_image=False,
        lat=lat,
        lon=lon,
        years=[int(y) for y in (years or [])],
        tags=[
            Tag(facet="theme_what", label=theme_label, weight=1.0),
            Tag(facet="theme_what", label=sub, weight=0.8),
            Tag(facet="theme_how.type_of_stores", label="Personal stories", weight=0.5),
        ],
    ), vec, payload


# content_id -> (Content, vector, raw payload). Coordinates are spots on the
# Bergen-Belsen memorial site; creation years give the temporal view shape.
_WORLD = dict([
    ("101", _c("101", "Forced labour in the textile workshop", 0, "Forced Labor", "workshops", (52.7581, 9.9062), [1942, 1943])),
    ("102", _c("102", "Construction work detail", 0, "Forced Labor", "work detail", (52.7576, 9.9081), [1941, 1942])),
    ("103", _c("103", "Armament factory labour", 0, "Forced Labor", "work detail", (52.7583, 9.9079), [1943, 1944])),
    ("201", _c("201", "Children in the women's camp", 1, "Family", "children", (52.7562, 9.9105), [1944])),
    ("202", _c("202", "A family's life in the Star Camp", 1, "Family", "family life", (52.7568, 9.9111), [1943, 1944])),
    ("203", _c("203", "Siblings reunited after deportation", 1, "Family", "siblings", (52.7565, 9.9098), [1944, 1945])),
    ("301", _c("301", "Liberation by the British Army", 2, "Liberation", "camp liberation", (52.7597, 9.9041), [1945])),
    ("302", _c("302", "First days after liberation", 2, "Liberation", "liberator", (52.7599, 9.9048), [1945])),
])


def make_world() -> FakeContentStore:
    return FakeContentStore({cid: cv[0] for cid, cv in _WORLD.items()},
                            {cid: cv[1] for cid, cv in _WORLD.items()},
                            payloads={cid: cv[2] for cid, cv in _WORLD.items()})


def make_contents_and_vectors():
    contents = {cid: cv[0] for cid, cv in _WORLD.items()}
    vectors = {cid: cv[1] for cid, cv in _WORLD.items()}
    return contents, vectors


def make_payloads():
    """Raw payload dicts (location, time_metadata, …) matching the fixture world."""
    return {cid: cv[2] for cid, cv in _WORLD.items()}


def view_events(
    user_id: str,
    content_id: str,
    *,
    dwell: float,
    reason: str,
    base_ts: datetime,
    visits: int = 1,
) -> list[InteractionEvent]:
    """Emit a START + END pair (separate events) for one content."""
    out: list[InteractionEvent] = []
    for k in range(visits):
        t0 = base_ts + timedelta(minutes=k)
        out.append(InteractionEvent(
            user_id=user_id, event="CONTENT_VIEW_STARTED",
            content_id=content_id, session_id="s1", ts=t0,
        ))
        out.append(InteractionEvent(
            user_id=user_id, event="CONTENT_VIEW_ENDED",
            content_id=content_id, session_id="s1",
            ts=t0 + timedelta(seconds=dwell),
            dwell_seconds=dwell, end_reason=reason,
        ))
    return out
