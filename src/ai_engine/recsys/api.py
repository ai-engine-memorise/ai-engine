"""FastAPI surface for the recommendation engine.

- POST /api/ingest   : the ingest WEBHOOK. RudderStack POSTs user events here
                          (single object or list). Normalize -> buffer -> rebuild
                          the user model.
- GET  /api/recommend: serve recommendations for a user (reads the user model).
- GET  /api/usermodel: debug — inspect the current user model.

Mount `router` into the main service, or run `app` standalone. With no REDIS_URL /
QDRANT_API_URL set it runs fully in-memory on dev fixtures.
"""
from __future__ import annotations
import logging
import os
from datetime import datetime, timezone
from typing import Any, Optional, Union

from fastapi import APIRouter, FastAPI, Body, Query, Header, HTTPException, Depends
from pydantic import BaseModel, Field

from .composition import Components, build_components
from .adapters.rudderstack import normalize_events
from .contracts.models import UserSignals
from .survey import survey_affinity


class PreviewSpec(BaseModel):
    """A hand-authored user model for testing recs without going through events."""
    tag_affinity: dict[str, float] = Field(default_factory=dict)   # {"theme_what:forced labor": 1.0}
    like_items: list[str] = Field(default_factory=list)            # -> taste vector (centroid) + excluded as seen
    demographics: dict = Field(default_factory=dict)               # {"age_group":"25_34","gender":"female",...}
    limit: Optional[int] = None


def build_preview_signals(spec: PreviewSpec, content_store) -> UserSignals:
    taste = None
    if spec.like_items:
        vecs = content_store.get_vectors(spec.like_items)
        acc = None
        for v in vecs.values():
            if acc is None:
                acc = [0.0] * len(v)
            for i, x in enumerate(v):
                acc[i] += x
        if acc and any(acc):
            n = sum(x * x for x in acc) ** 0.5
            taste = [x / n for x in acc] if n else acc

    aff = dict(spec.tag_affinity)
    for k, w in survey_affinity(spec.demographics).items():
        aff[k] = aff.get(k, 0.0) + w
    folded: dict[str, float] = {}
    for k, v in aff.items():
        folded[k.lower()] = folded.get(k.lower(), 0.0) + v
    if folded:
        mx = max(folded.values())
        if mx > 0:
            folded = {k: v / mx for k, v in folded.items()}

    return UserSignals(
        user_id="preview",
        positives={i: 1.0 for i in spec.like_items},
        tag_affinity=folded,
        taste_vector=taste,
        demographics=spec.demographics,
    )

logger = logging.getLogger(__name__)


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Guard write endpoints with a shared secret (env INGEST_API_KEY).

    If INGEST_API_KEY is unset, requests are allowed (dev) but a warning is logged.
    """
    expected = os.getenv("INGEST_API_KEY")
    if not expected:
        logger.warning("INGEST_API_KEY unset: /api/ingest is UNAUTHENTICATED")
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _dump_items(items, include_content: bool) -> list:
    """Output contract: id / rank / relevance_score / role (+ breakdown, optional content).
    The distractor is OUTSIDE the rank ordering -> rank=null; only targets are ranked."""
    out = []
    rank = 0
    for i in items:
        d = i.model_dump()
        is_distractor = d.get("kind") == "distractor"
        if not is_distractor:
            rank += 1
        item = {
            "id": d["content_id"],
            "rank": None if is_distractor else rank,
            "relevance_score": d["final_score"],
            "role": "distractor" if is_distractor else "target",
            "breakdown": d.get("breakdown", {}),
        }
        if include_content and d.get("content") is not None:
            item["content"] = d["content"]
        out.append(item)
    return out


def make_router(components: Components) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["Recsys"])
    c = components

    @router.post("/ingest", dependencies=[Depends(_require_api_key)])
    def ingest(payload: Union[dict, list] = Body(...)) -> dict:
        raws = payload if isinstance(payload, list) else [payload]
        events = normalize_events(raws)
        c.event_log.append(events)    # durable append-only log (Parquet) — training/eval record
        users: set[str] = set()
        for e in events:
            c.event_buffer.append(e)  # type: ignore[attr-defined]
            users.add(e.user_id)
        now = datetime.now(timezone.utc)
        for u in users:
            demographics = c.demographics.get_demographics(u)
            c.updater.refresh(u, c.event_buffer, now=now, demographics=demographics)
        return {"status": "ok", "ingested": len(events), "users": sorted(users)}

    @router.get("/recommend")
    def recommend(
        user_id: str = Query(..., examples=["u1"]),
        limit: Optional[int] = Query(default=None, ge=1, le=50),
        filter: Optional[str] = Query(default=None, description="restrict candidates to a tag, e.g. a location: AiARLocationBarrack3"),
        include_content: bool = Query(default=True, description="false = compact (ids/scores only)"),
    ) -> dict:
        rec = c.recommender.recommend(user_id, filter=filter)
        items = rec.items[:limit] if limit else rec.items
        out = rec.model_dump()
        out["filter"] = filter
        out["items"] = _dump_items(items, include_content)
        return {"result": out}

    @router.get("/usermodel")
    def usermodel(user_id: str = Query(..., examples=["u1"])) -> dict:
        sig = c.model_store.get_signals(user_id)
        return {"result": sig.model_dump() if sig else None}

    @router.post("/recommend/preview")
    def recommend_preview(
        spec: PreviewSpec,
        filter: Optional[str] = Query(default=None),
        include_content: bool = Query(default=True, description="false = compact (ids/scores only)"),
    ) -> dict:
        """Recommend from a hand-authored user model (no events). For manual /
        programmatic testing + LLM evaluation."""
        signals = build_preview_signals(spec, c.content_store)
        rec = c.recommender.recommend_for_signals(signals, filter=filter)
        items = rec.items[:spec.limit] if spec.limit else rec.items
        out = rec.model_dump()
        out["filter"] = filter
        out["items"] = _dump_items(items, include_content)
        out["user_model"] = signals.model_dump()
        return {"result": out}

    return router


def create_app(components: Optional[Components] = None) -> FastAPI:
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="AI-Engine Recsys")
    # browser test UIs (ui4testing) call /api/* directly; allow cross-origin in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("AI_ENGINE_CORS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(make_router(components or build_components()))

    @app.get("/health", tags=["Health"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.recsys.api:app", host="0.0.0.0", port=8001, reload=True)
