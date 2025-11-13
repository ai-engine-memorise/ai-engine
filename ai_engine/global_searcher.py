from loguru import logger
from typing import Optional, List, Dict, Any
from ai_engine.help_searcher import CommonSearcher
from ai_engine.vector_searcher import VectorSearcher
from ai_engine.geo_searcher import GeoSearcher
from ai_engine.user_searcher import UserRecommender
from ai_engine.common import SearchResult

# Umbrella class for the Qdrant Fetching Logic Text + Vector + Geo

class GlobalSearch:
    def __init__(self, collection_name: str):
        self.common_searcher = CommonSearcher(collection_name=collection_name)
        self.geo_searcher = GeoSearcher(collection_name=collection_name)
        self.vector_searcher = VectorSearcher(collection_name=collection_name)
        self.user_recommender = UserRecommender(collection_name=collection_name)

    def search(
        self,
        text: Optional[str] = None,
        *,
        lat: Optional[float] = None,
        lon: Optional[float] = None,
        radius_meters: Optional[float] = None,
    ) -> SearchResult:
        """
        Unified search behavior:
        - q only                -> vector search
        - geo only (lat/lon)    -> geo search
        - q + geo               -> hybrid search
        """
        has_query = text is not None and str(text).strip() != ""
        has_geo = lat is not None and lon is not None

        # Nothing to search on
        if not has_query and not has_geo:
            logger.exception("Either 'text' or (lat & lon) must be provided")
            raise ValueError("Either 'text' or (lat & lon) must be provided")

        # 1) Vector only (just q)
        if has_query and not has_geo:
            return self.vector_searcher.search(text=text)

        # 2) Geo only (just lat/lon)
        if has_geo and not has_query:
            return self.geo_searcher.search(lat=lat, lon=lon, radius_meters=radius_meters)

        # 3) Hybrid (both q and geo)
        embedding = self.vector_searcher.encode(text=text)
        return self.geo_searcher.hybrid_search(
            query_vector=embedding,
            lat=lat,
            lon=lon,
            radius_meters=radius_meters,
        )