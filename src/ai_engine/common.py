from dataclasses import dataclass, fields, field, asdict
from datetime import datetime
from typing import Optional, List, Dict, Any, Literal
import pandas as pd
import numpy as np


def clean_payload_field(data: Any) -> Any:
    """Converts data containing NumPy types to native Python types."""
    if isinstance(data, (np.ndarray, pd.Series)):
        return data.tolist()


@dataclass
class Item:
    id: int
    title: str
    public_url: str

    text: Optional[str] = field(default="")
    creator: Optional[str] = field(default="")
    locations: Optional[List[Dict[str, Any]]] = field(default_factory=list)
    geo_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    time_metadata: Optional[Dict[str, Any]] = field(default_factory=dict)
    files_url: List[str] = field(default_factory=list)

    def __post_init__(self):
        self.locations = clean_payload_field(self.locations)
        self.files_url = clean_payload_field(self.files_url)
            
    @property
    def image_url(self) -> Optional[str]:
        files_url = self.files_url
        if isinstance(files_url, list) and files_url:
            return files_url[0]
        elif isinstance(files_url, str):
            return files_url
        return None
    
    @property
    def text_all(self) -> str:
        return f"{self.title}\n{self.text}" if pd.notnull(self.text) else self.title
    
    @property
    def char_count(self) -> int:
        """Returns the length of text_all in characters."""
        return len(self.text_all)

    @property
    def word_count(self) -> int:
        """Returns the word count of text_all."""
        # .split() handles various whitespace separators, including the '\n'
        return len(self.text_all.split())
    
    # --- loading methods

    @classmethod
    def from_payload(cls, payload: dict) -> "Item":
        allowed = {f.name for f in fields(cls)}
        data = {k: v for k, v in payload.items() if k in allowed}
        return cls(**data)


@dataclass
class User:
    id: int
    age: int
    gender: str
    nationality: str
    personal_connection: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


EventType = Literal['start', 'end']

@dataclass
class Event:
    id: int
    user_id: int
    item_id: int
    event_type: EventType
    ts: datetime

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    # @property
    # def duration(self):
    #     return self.time_end - self.timestart


# --- Search Standarization ----
# ------------------------------

SearchType = Literal["vector", "geo", "hybrid"]


def hit_to_item(
    hit: Any,          # Qdrant ScoredPoint or Record
    source: SearchType,
    query_text: Optional[str] = None,
) -> dict[str, Any]:
    """
    Convert a Qdrant point to a small, standardized dict without
    assuming anything about the payload structure.
    """
    payload = hit.payload or {}
    score = getattr(hit, "score", None)

    highlight = None
    if query_text:
        # You can decide later how/if to use this
        text_for_highlight = payload.get("text") or ""
        highlight = highlight_search_match(text_for_highlight, query_text)

    return {
        "id": hit.id,
        "source": source,
        "score": score,
        "payload": payload,
        "highlight": highlight,
    }


@dataclass
class SearchResult:
    search_type: SearchType
    query_text: Optional[str]
    lat: Optional[float]
    lon: Optional[float]
    radius_meters: Optional[float]
    items: list[dict[str, Any]]
    next_offset: Any = None

    def dict(self) -> dict[str, Any]:
        return {
            "search_type": self.search_type,
            "query": {
                "text": self.query_text,
                "lat": self.lat,
                "lon": self.lon,
                "radius_meters": self.radius_meters,
            },
            "items": self.items,
            "next_offset": self.next_offset,
        }


# ------------------------------

import re

def limit_text(text: str, lim: int = 80):
    """
    >>> limit_text("hello world", 5)
    'hello...'

    >>> limit_text("hello world", 100)
    'hello world'

    :param text: Text to limit
    :param lim: Max length
    :return: Limited text
    """
    if len(text) > lim:
        return text[:lim] + "..."
    return text


def highlight_search_match(text: str, query: str, before="<b>", after="</b>"):
    """
    >>> highlight_search_match("hello world", "world")
    'hello <b>world</b>'

    >>> highlight_search_match("hello world", "hello")
    '<b>hello</b> world'

    >>> highlight_search_match("hello world", "hell")
    '<b>hell</b>o world'

    >>> highlight_search_match("Hello world", "hell")
    '<b>Hell</b>o world'

    >>> highlight_search_match("hello world", "foo")
    'hello world'

    >>> highlight_search_match("hello world", "ello")
    'hello world'


    :param text: Found string
    :param query: Search query
    :param before: Tag before match
    :param after: Tag after match
    :return: Highlighted string
    """
    # Replace matches only on word boundaries
    return re.compile(r"\b(" + re.escape(query) + ")", re.IGNORECASE).sub(before + '\\1' + after, text)