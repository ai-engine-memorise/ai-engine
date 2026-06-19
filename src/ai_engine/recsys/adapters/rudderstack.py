"""Pure RudderStack -> InteractionEvent normalizer.

No IO, no infra: takes raw RudderStack `track` payloads (the same shape RudderStack
delivers to a webhook or writes to its warehouse) and maps them to canonical events.
This is the single place source-specific shape is handled, and it is fully testable
with plain dicts. A PostHog normalizer would live beside this and emit the same type.
"""
from __future__ import annotations
import re
from datetime import datetime, timezone
from typing import Iterable, Optional

from ..contracts.enums import EndReason
from ..contracts.models import InteractionEvent

_DIGITS = re.compile(r"\d+")


def normalize_content_id(raw_id: Optional[str]) -> Optional[str]:
    """'content_1234' -> '1234'; '841' -> '841'; None -> None.

    Bridges the event schema's string ids to the Qdrant integer point ids.
    """
    if raw_id is None:
        return None
    m = _DIGITS.search(str(raw_id))
    return m.group(0) if m else str(raw_id)


def _parse_ts(value) -> datetime:
    if isinstance(value, datetime):
        return value
    if value is None:
        return datetime.now(timezone.utc)
    s = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.now(timezone.utc)


def _end_reason(value) -> Optional[EndReason]:
    if not value:
        return None
    try:
        return EndReason(value)
    except ValueError:
        return None


def normalize_event(raw: dict) -> Optional[InteractionEvent]:
    """Map one RudderStack track/identify payload to an InteractionEvent (or None)."""
    if not isinstance(raw, dict):
        return None
    # identify call (app sends it AFTER the survey, traits = persona/demographics).
    # No `event` field -> route traits through the survey/demographics fold.
    if raw.get("type") == "identify" or (raw.get("traits") and not raw.get("event")):
        user_id = raw.get("userId") or raw.get("anonymousId")
        if not user_id:
            return None
        return InteractionEvent(
            user_id=str(user_id),
            event="IDENTIFY",
            ts=_parse_ts(raw.get("timestamp") or raw.get("sentAt")),
            survey_answers={k: v for k, v in (raw.get("traits") or {}).items() if v is not None},
            raw=raw,
        )

    event = raw.get("event")
    user_id = raw.get("userId") or raw.get("anonymousId")
    if not event or not user_id:
        return None

    props = raw.get("properties") or {}
    content = props.get("content") or {}
    details = props.get("details") or {}
    context = props.get("context") or {}

    # `content` / candidates may arrive as a dict ({content_id: ...}) OR as the bare
    # content_id string — RudderStack/clients send either. Accept both.
    def _cid(x):
        return x.get("content_id") if isinstance(x, dict) else x

    content_id = normalize_content_id(_cid(content))

    impressions = [
        normalize_content_id(_cid(c))
        for c in (context.get("candidates") or [])
        if _cid(c)
    ]
    impressions = [i for i in impressions if i]

    survey_answers: dict = {}
    for ans in (props.get("answers") or []):
        qid, val = ans.get("question_id"), ans.get("answer_value")
        if qid is not None and val is not None:
            if qid in survey_answers:           # multi-select -> collect into a list
                ex = survey_answers[qid]
                survey_answers[qid] = (ex if isinstance(ex, list) else [ex]) + [val]
            else:
                survey_answers[qid] = val
        if ans.get("question_type") == "rating" and val is not None:
            try:
                survey_answers["rating"] = float(val)
            except (TypeError, ValueError):
                pass

    # request_id: the app echoes the rec response's id on the resulting view, so the
    # bandit trainer can join this reward to the exact impression (its feature vector).
    request_id = (details.get("request_id") or context.get("request_id")
                  or props.get("request_id"))

    return InteractionEvent(
        user_id=str(user_id),
        event=str(event),
        ts=_parse_ts(raw.get("timestamp") or raw.get("sentAt")),
        session_id=context.get("session_id") or raw.get("sessionId"),
        request_id=request_id,
        content_id=content_id,
        dwell_seconds=details.get("dwell_seconds"),
        end_reason=_end_reason(details.get("reason")),
        query_text=details.get("query_text"),
        clicked_id=normalize_content_id(details.get("clicked_id")),
        impressions=impressions,
        survey_answers=survey_answers,
        raw=raw,
    )


def normalize_events(raws: Iterable[dict]) -> list[InteractionEvent]:
    out = [normalize_event(r) for r in raws]
    out = [e for e in out if e is not None]
    out.sort(key=lambda e: e.ts)
    return out
