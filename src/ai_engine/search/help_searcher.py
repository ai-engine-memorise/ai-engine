
from loguru import logger
from typing import Any, Dict, Optional, List, Union
from ai_engine.config import QDRANT_API_KEY, QDRANT_API_URL, COLLECTION_NAME
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny, CreateFieldIndex, PayloadSchemaType

class CommonSearch:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=QDRANT_API_URL,
            api_key=QDRANT_API_KEY
        )

    def get_item(self, item_id: Union[int, List[int]]) -> Optional[Dict[str, Any]]:
        """
        Fetch item by Point ID.
        """
        if isinstance(item_id, int):
            item_id = [item_id]

        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=item_id,
            with_payload=True,
            with_vectors=False,
        )

        if not res:
            return {}

        return {p.id: p.payload for p in res}

    def get_item_by_item_id(self, item_id: Union[int, List[int]]) -> Optional[List[Dict[str, Any]]]:
        """
        Fetch item by ID.
        """
        # self.client.create_payload_index(
        #     collection_name=self.collection_name,
        #     field_name="id",
        #     field_schema=PayloadSchemaType.KEYWORD,
        # )
        
        if isinstance(item_id, str):
            item_id = [item_id]
        
        scroll_filter = Filter(
            must=[
                FieldCondition(
                    key="id",
                    match=MatchAny(any=item_id),
                )
            ]
        )
      
        res, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=scroll_filter,
            with_payload=True,
            with_vectors=False,
            limit=len(item_id),
        )
        
        if not res:
            return []
        
        # return [p.payload for p in res]
        return {p.id: p.payload for p in res}
        # return [{"id": p.id, "payload": p.payload} for p in res]
    
    