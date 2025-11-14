
# Use User info as Query to Qdrant
from typing import List, Iterable, Dict, Any

from qdrant_client import QdrantClient
from qdrant_client.models import Filter

from fastembed import TextEmbedding

from ai_engine.config import QDRANT_API_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL, SEARCH_LIMIT
from ai_engine.common import SearchResult, hit_to_item 

#################
#### Qdrant  ####
#################

BATCH_SIZE = 32

def iter_batch(iterable: Iterable[str], batch_size: int) -> Iterable[List[str]]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def user_to_query_text(user: Dict[Any, str]) -> str:
    """
    Convert user profile into a short natural-language description
    that the embedding model can understand.
    """
    base = f"{user.age}-year-old {user.gender} from {user.nationality}"
    tail = (
            "interested in content related to people from the same nationality and similar age."
        )
    return f"{base}, {tail}"

# -------------------------------------------------------------------------------------------------------------------------------

class VectorSearch:

    def __init__(self, collection_name: str):
        self.collection_name = collection_name
        self.model = TextEmbedding(model_name=EMBEDDING_MODEL)
        self.client = QdrantClient(
            url=QDRANT_API_URL, 
            api_key=QDRANT_API_KEY,
        )

    def search(self, text: str, filter_: dict = None) -> SearchResult:
        vector = next(iter(self.model.embed(text))).tolist()
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=vector,
            limit=SEARCH_LIMIT,
            query_filter=Filter(**filter_) if filter_ else None,
            with_payload=True,
            with_vectors=False,
        ) # QueryResponse.points == list[ScoredPoint]
        
        items = [
            hit_to_item(hit=hit, source="vector", query_text=text)
            for hit in res.points
        ]

        return SearchResult(
            search_type="vector",
            query_text=text,
            lat=None,
            lon=None,
            radius_meters=None,
            items=items,
            next_offset=None,
        )

    def encode_iter(self, texts: Iterable[str]) -> Iterable[list]:
        for vector in self.model.embed(texts, parallel=4):
            yield vector.tolist()

    def encode(self, text: str) -> list:
        return next(self.encode_iter([text]))
    
    def get_model_dim(self) -> int:
        return self.model._get_model_description(EMBEDDING_MODEL).dim

# ---------- Example usage ----------

if __name__ == '__main__':
    from ai_engine.common import User

    neural_searcher = VectorSearch(collection_name=COLLECTION_NAME)

    user = User(
        age = 24,
        gender = 'male',
        nationality = 'german',
        personal_connection = True
    )
    query_user = user_to_query_text(user=user)
    result = neural_searcher.search(query_user)
    for res in result:
        res_item = res['payload']
        res_score = res['score']
        print(f"> {res_item['id']}] {res_item['title']}{res_item['text'] or ""}\n")
        print(f"> {res_item['public_url']}\n")
        print(f"{res_score}\n\n")

