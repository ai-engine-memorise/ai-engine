# %%
from loguru import logger
from typing import Optional, List, Dict, Any

import pandas as pd  # assuming you already use it

from ai_engine.db_interface import DB_Interface
from ai_engine.search import CommonSearch
from ai_engine.user_state import UserState
from ai_engine.common import Item

### Projection Builder Class
### Given user_events table it creates other projection tables:
### 1) user_item_stats: aggregates per user/item stats (views, dwell time, success count)
### 2) user_embeddings: precomputed user embeddings based on recent events

class ProjectionBuilder:
    def __init__(self, collection_name: str):
        self.user_state = UserState()
        self.db_engine = DB_Interface()
        self.common_searcher = CommonSearch(collection_name=collection_name)

    ### Public entrypoints
    #############################

    def get_user_projection(self, user_id: str) -> pd.DataFrame:
        """
        Compute user projection embedding based on recent events.
        """
        # Fetch user events
        events_df = self.db_engine.fetch_events(user_id=int(user_id))
        logger.info(events_df.dtypes)
        item_details = self._get_item_details(item_id=events_df['item_id'].astype(int).unique().tolist())
        
        # For each row in events_df, compute estimated reading time and success flag
        for idx, event in events_df.iterrows():
            item_id = int(event['item_id'])
            word_count = item_details[item_id]['word_count']
            has_image = item_details[item_id]['has_image']
            estimated_time = self.user_state.compute_reading_time(content_length_words=word_count, has_image=has_image)
            events_df.at[idx, 'estimated_reading_time'] = estimated_time
            is_successful = self.user_state.is_interaction_successful(
                dwell_time=event['dwell_seconds'],
                estimated_reading_time=estimated_time
            )
            events_df.at[idx, 'is_successful'] = is_successful

        return events_df

    ### Derived views
    #############################

    def _get_item_details(self, item_id: List[str]) -> Dict[str, Dict[str, Any]]:
        items = self.common_searcher.get_item(item_id=item_id)
        details = {}
        for item_id, item in items.items():
            logger.info(f"Retrieved {item_id}, {item}")
            item = Item.from_payload(item)
            details[item_id] = {
                "has_image": bool(item.image_url),
                "word_count": item.word_count,
            }
        return details


# %%

if __name__ == "__main__":
    # %%
    from ai_engine.config import COLLECTION_NAME
    # Example usage
    pb = ProjectionBuilder(collection_name=COLLECTION_NAME)

    item_id = ["1227"]
    items = pb.common_searcher.get_item(item_id=item_id)
    print(items) 

    user_history = pb.get_user_history(user_id="10")
    print(user_history)

    user_projection = pb.get_user_projection(user_id="10")
    print(user_projection)
