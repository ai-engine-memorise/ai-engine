from loguru import logger
from typing import List
from dataclasses import asdict
from qdrant_client.models import RecommendStrategy

from ai_engine.projection_builder import ProjectionBuilder
from ai_engine.common import SearchResult, hit_to_item
from ai_engine.config import SEARCH_LIMIT, COLLECTION_NAME


class UserRecommender:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.projection_builder = ProjectionBuilder(collection_name=collection_name)
        self.common_searcher = self.projection_builder.common_searcher
        self.client = self.common_searcher.client
        self.collection_name = self.common_searcher.collection_name

    def _get_user_signals(self, user_id: int):
        events = self.projection_builder.get_user_projection(user_id=str(user_id))

        positives: List[str] = []
        negatives: List[str] = []

        for _, row in events.iterrows():
            item_id = row["item_id"]
            is_successful = row["is_successful"]
            if is_successful:
                positives.append(str(item_id))
            else:
                negatives.append(str(item_id))

        return positives, negatives, events

    def recommend_for_user(self, user_id: int, limit: int = SEARCH_LIMIT) -> SearchResult:
        positives, negatives, events = self._get_user_signals(user_id)

        if not positives:
            logger.info(f"No positive signals for user {user_id}, cannot recommend.")
            return []

        positive_ids = [int(i) for i in positives]
        negative_ids = [int(i) for i in negatives] if negatives else None

        results = self.client.recommend(
            collection_name=self.collection_name,
            positive=positive_ids,
            negative=negative_ids,
            with_payload=True,
            with_vectors=False,
            limit=limit,
            strategy=RecommendStrategy.AVERAGE_VECTOR,
        )

        items = [
            hit_to_item(hit=hit, source="user", query_text=None)
            for hit in results
        ]

        return SearchResult(
            search_type="user",
            query_text=None,
            lat=None,
            lon=None,
            radius_meters=None,
            items=items,
            next_offset=None,
        )

