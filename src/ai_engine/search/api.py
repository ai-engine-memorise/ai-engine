"""FastAPI surface for Qdrant search.

Ported from the legacy ai-engine-api `service.py` so the recsys serving image
preserves the search endpoints (vector / geo / preference / profile) and the
Qdrant item lookup. Paths and response shapes are kept identical to the legacy
API so existing consumers do not break.
"""
from typing import Optional, List

from fastapi import APIRouter, Query
from loguru import logger

from ai_engine.config import COLLECTION_NAME
from ai_engine.search import GlobalSearch
from ai_engine.projection_builder import ProjectionBuilder


def make_search_router(collection_name: str = COLLECTION_NAME) -> APIRouter:
    """Build the search router. Instantiates GlobalSearch once (loads the
    embedding model + Qdrant clients), mirroring the legacy module-level
    `searcher`."""
    router = APIRouter()
    searcher = GlobalSearch(collection_name=collection_name)

    @router.get("/api/search", tags=["Search"])
    async def read_item_search(
        q: str = Query(..., description="Search text", example="Bergen-Belsen"),
    ):
        results = searcher.search(text=q)
        return {"result": results}

    @router.get("/api/search/geo", tags=["Search"])
    async def read_geo_search(
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
    async def read_event_search(user_id: int = Query(..., example=10)):
        """Recommend by user history (Qdrant taste vector)."""
        results = searcher.user_recommender.recommend_for_user(user_id=user_id)
        return {"result": results.dict()}

    @router.get("/api/search/profile", tags=["Search"])
    async def read_user_search(user_id: int = Query(..., example=10)):
        """Build a text profile for the user, then vector-search on it."""
        db_projection = ProjectionBuilder(collection_name=collection_name)
        user_query = db_projection.get_user_profile_as_text(user_id=user_id)
        results = searcher.search(text=user_query)
        return {"result": results.dict()}

    @router.get("/debug/item_info", tags=["Debug"])
    async def read_item(item_id: List[int] = Query(..., example="2148")):
        item = searcher.common_searcher.get_item(item_id=item_id)
        logger.info(f"Fetched item: {item}")
        return {"result": item}

    return router
