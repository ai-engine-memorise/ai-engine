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


def _c(id, title, axis, theme_label, sub):
    vec = [0.0] * 6
    vec[axis] = 1.0
    return Content(
        id=id,
        content_type=ContentType.text_item,
        title=title,
        text=title,
        word_count=120,
        has_image=False,
        tags=[
            Tag(facet="theme_what", label=theme_label, weight=1.0),
            Tag(facet="theme_what", label=sub, weight=0.8),
            Tag(facet="theme_how.type_of_stores", label="Personal stories", weight=0.5),
        ],
    ), vec


# content_id -> (Content, vector)
_WORLD = dict([
    ("A1", _c("A1", "Forced labour in the textile workshop", 0, "Forced Labor", "workshops")),
    ("A2", _c("A2", "Construction work detail", 0, "Forced Labor", "work detail")),
    ("A3", _c("A3", "Armament factory labour", 0, "Forced Labor", "work detail")),
    ("B1", _c("B1", "Children in the women's camp", 1, "Family", "children")),
    ("B2", _c("B2", "A family's life in the Star Camp", 1, "Family", "family life")),
    ("B3", _c("B3", "Siblings reunited after deportation", 1, "Family", "siblings")),
    ("C1", _c("C1", "Liberation by the British Army", 2, "Liberation", "camp liberation")),
    ("C2", _c("C2", "First days after liberation", 2, "Liberation", "liberator")),
])


def make_world() -> FakeContentStore:
    contents = {cid: cv[0] for cid, cv in _WORLD.items()}
    vectors = {cid: cv[1] for cid, cv in _WORLD.items()}
    return FakeContentStore(contents, vectors)


def make_contents_and_vectors():
    contents = {cid: cv[0] for cid, cv in _WORLD.items()}
    vectors = {cid: cv[1] for cid, cv in _WORLD.items()}
    return contents, vectors


def view_events(
    user_id: str,
    content_id: str,
    *,
    dwell: float,
    reason: str,
    base_ts: datetime,
    visits: int = 1,
) -> list[InteractionEvent]:
    """Emit a START + END pair (path-B style separate events) for one content."""
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
