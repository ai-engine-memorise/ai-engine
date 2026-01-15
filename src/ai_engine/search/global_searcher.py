from loguru import logger
from typing import Optional, List, Dict, Any
from ai_engine.common import SearchResult
from .help_searcher import CommonSearch
from .geo_searcher import GeoSearch
from .vector_searcher import VectorSearch
from .user_searcher import UserRecommender
from ..config import COLLECTION_NAME

# Umbrella class for the Qdrant Fetching Logic Text + Vector + Geo

class GlobalSearch:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.common_searcher = CommonSearch(collection_name=collection_name)
        self.geo_searcher = GeoSearch(collection_name=collection_name)
        self.vector_searcher = VectorSearch(collection_name=collection_name)
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
    
    def similar(
        self,
        item_id: int,
        *,
        exclude_self: bool = True,
        vector_name: Optional[str] = None,  # only needed if your collection uses named vectors
    ) -> SearchResult:
        """
        Similar items for a given item_id:
        - fetch point vector by id
        - run vector similarity search
        """

        vectors = self.common_searcher.get_vector(item_id)  # {id: vector}
        if item_id not in vectors:
            raise ValueError(f"Item id {item_id} not found")

        query_vector = vectors[item_id]

        # If Qdrant returns named vectors: {"text": [...], ...}
        if isinstance(query_vector, dict):
            if not vector_name:
                raise ValueError(
                    f"Named vectors detected; pass vector_name. Available: {list(query_vector.keys())}"
                )
            query_vector = query_vector[vector_name]

        # This assumes your VectorSearch.search supports vector=...
        result = self.vector_searcher.search(vector=query_vector)

        if exclude_self and getattr(result, "items", None):
            result.items = [
                it for it in result.items
                if str(getattr(it, "id", "")) != str(item_id)
            ]

        return result

    def random(self):

        result = self.common_searcher.get_random_item()
        
        return result
        