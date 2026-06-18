"""FastAPI surface for Qdrant search.

Ported from the legacy ai-engine-api `service.py` so the recsys serving image
preserves the search endpoints (vector / geo / preference / profile) and the
Qdrant item lookup. Paths and response shapes are kept identical to the legacy
API so existing consumers do not break.

Routes are plain `def` (not `async`): the bodies call blocking Qdrant / embedding
I/O, so FastAPI runs them in its threadpool instead of stalling the event loop.
"""
from typing import Optional, List

from fastapi import APIRouter, Depends, Query
from loguru import logger

from ai_engine.config import COLLECTION_NAME
from ai_engine.search import GlobalSearch


def make_search_router(collection_name: str = COLLECTION_NAME) -> APIRouter:
    """Build the search router. Instantiates GlobalSearch once (loads the
    embedding model + Qdrant clients) and reuses its ProjectionBuilder, mirroring
    the legacy module-level `searcher` without per-request construction."""
    # Lazy import to avoid a circular import: recsys.api imports this module from
    # inside create_app(), by which point recsys.api is fully defined.
    from ai_engine.recsys.api import _require_api_key

    router = APIRouter()
    searcher = GlobalSearch(collection_name=collection_name)
    # reuse the projection builder already created by the user recommender instead
    # of newing one (DB engine + Qdrant client) per /profile request.
    projection_builder = searcher.user_recommender.projection_builder

    @router.get("/api/search", tags=["Search"])
    def read_item_search(
        q: str = Query(..., description="Search text", example="Bergen-Belsen"),
    ):
        results = searcher.search(text=q)
        return {"result": results.dict()}

    @router.get("/api/search/geo", tags=["Search"])
    def read_geo_search(
        lat: float = Query(..., example=52.7579),
        lon: float = Query(..., example=9.9048),
        radius_meters: float = Query(5000),
        q: Optional[str] = Query(
            default=None, description="Search text (optional)", example="Bergen-Belsen"
        ),
    ):
        results = searcher.search(text=q, lat=lat, lon=lon, radius_meters=radius_meters)
        return {"result": results.dict()}

    @router.get("/api/search/preference", tags=["Search"])
    def read_event_search(user_id: int = Query(..., example=10)):
        """Recommend by user history (Qdrant taste vector)."""
        results = searcher.user_recommender.recommend_for_user(user_id=user_id)
        return {"result": results.dict()}

    @router.get("/api/search/profile", tags=["Search"])
    def read_user_search(user_id: int = Query(..., example=10)):
        """Build a text profile for the user, then vector-search on it."""
        user_query = projection_builder.get_user_profile_as_text(user_id=user_id)
        results = searcher.search(text=user_query)
        return {"result": results.dict()}

    @router.get("/debug/item_info", tags=["Debug"], dependencies=[Depends(_require_api_key)])
    def read_item(item_id: List[int] = Query(..., example="2148")):
        # guarded: returns raw Qdrant payloads, so it sits behind the same API key
        # as the other inspection endpoints.
        item = searcher.common_searcher.get_item(item_id=item_id)
        logger.info(f"Fetched item: {item}")
        return {"result": item}

    return router
