from enum import Enum


class ContentType(str, Enum):
    text_item = "text_item"
    image_item = "image_item"
    video_item = "video_item"
    audio_item = "audio_item"
    exhibition = "exhibition"
    tag = "tag"
    poi = "poi"


class EndReason(str, Enum):
    """How a content view ended (RudderStack CONTENT_VIEW_ENDED.details.reason)."""
    next_button = "next_button"   # advanced on purpose -> engaged
    link = "link"                 # followed a link -> mildly engaged
    close_button = "close_button" # closed -> neutral/weak
    abandon = "abandon"           # left without ending -> negative


class Outcome(str, Enum):
    positive = "positive"
    negative = "negative"
    neutral = "neutral"
