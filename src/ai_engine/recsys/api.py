"""FastAPI surface for the recommendation engine.

- POST /recsys/ingest   : the ingest WEBHOOK. RudderStack POSTs user events here
                          (single object or list). Normalize -> buffer -> rebuild
                          the user model.
- GET  /recsys/recommend: serve recommendations for a user (reads the user model).
- GET  /recsys/usermodel: debug — inspect the current user model.

Mount `router` into the main service, or run `app` standalone. With no REDIS_URL /
QDRANT_API_URL set it runs fully in-memory on dev fixtures.
"""
from __future__ import annotations
from datetime import datetime, timezone
from typing import Any, Optional, Union

from fastapi import APIRouter, FastAPI, Body, Query

from .composition import Components, build_components
from .adapters.rudderstack import normalize_events


def make_router(components: Components) -> APIRouter:
    router = APIRouter(prefix="/recsys", tags=["Recsys"])
    c = components

    @router.post("/ingest")
    def ingest(payload: Union[dict, list] = Body(...)) -> dict:
        raws = payload if isinstance(payload, list) else [payload]
        events = normalize_events(raws)
        users: set[str] = set()
        for e in events:
            c.event_buffer.append(e)  # type: ignore[attr-defined]
            users.add(e.user_id)
        now = datetime.now(timezone.utc)
        for u in users:
            c.updater.refresh(u, c.event_buffer, now=now)
        return {"status": "ok", "ingested": len(events), "users": sorted(users)}

    @router.get("/recommend")
    def recommend(
        user_id: str = Query(..., examples=["u1"]),
        limit: Optional[int] = Query(default=None, ge=1, le=50),
    ) -> dict:
        rec = c.recommender.recommend(user_id)
        items = rec.items[:limit] if limit else rec.items
        out = rec.model_dump()
        out["items"] = [i.model_dump() for i in items]
        return {"result": out}

    @router.get("/usermodel")
    def usermodel(user_id: str = Query(..., examples=["u1"])) -> dict:
        sig = c.model_store.get_signals(user_id)
        return {"result": sig.model_dump() if sig else None}

    return router


def create_app(components: Optional[Components] = None) -> FastAPI:
    app = FastAPI(title="AI-Engine Recsys")
    app.include_router(make_router(components or build_components()))

    @app.get("/health", tags=["Health"])
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.recsys.api:app", host="0.0.0.0", port=8001, reload=True)
