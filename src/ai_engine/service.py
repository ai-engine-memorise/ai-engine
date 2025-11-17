from typing import Optional, List
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ai_engine.config import COLLECTION_NAME, SEARCH_LIMIT
from ai_engine.db_interface import DB_Interface
from ai_engine.search import GlobalSearch
from ai_engine.projection_builder import ProjectionBuilder
from ai_engine.narrative import NarrativeGenerator
from ai_engine.common import Event, User

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

### Routes
#########################

### / Health Check
@app.get("/", tags=["Health"])
def root():
    return {
        "status": "ok",
        "message": "AI Engine Service is running."
    }

# Optional: serve index at root
@app.get("/demo", tags=["Health"], include_in_schema=False)
def demo():
    return FileResponse("static/index.html")

### / Search
searcher = GlobalSearch(collection_name=COLLECTION_NAME)

@app.get("/api/search", tags=["Search"])
async def read_item_search(q: str = Query(..., description="Search text", example="Bergen-Belsen")):
    results = searcher.search(text=q)
    return {
        "result": results
    }


@app.get("/api/search/geo", tags=["Search"])
async def read_geo_search(
    lat: float = Query(..., example=52.7579),
    lon: float = Query(..., example=9.9048),
    radius_meters: float = Query(5000),
    q: Optional[str] = Query(
        default=None,
        description="Search text (optional)",
        example="Bergen-Belsen",
    ),
):
    results = searcher.search(
            text=q,
            lat=lat,
            lon=lon,
            radius_meters=radius_meters
        )
    return {
        "result": results.dict()
    }


@app.get("/api/search/user", tags=["Search"])
async def read_user_search(
    user_id: int = Query(..., example=10),
    # lat: Optional[float] = Query(default=None, example=52.7579),
    # lon: Optional[float] = Query(default=None, example=9.9048),
    # radius_meters: float = Query(5000),
):
    """
    Search by user history and optionally by location.
    """
    user_recommender = searcher.user_recommender
    results = user_recommender.recommend_for_user(user_id=user_id)
    return {
        "result": results.dict()
    }
    # Get user state / history
    # Extract positive negative signals
    # Query qdrant on semantic taste vector
    # Rerank on engagement
    # Apply MMR for diversity

## Search by current:
#  - location, 
#  - user,
    #  - item,
    #  - item history

### / Narrative
narrative_generator = NarrativeGenerator()

class NarrativeRequest(BaseModel):
    items: List[dict]

@app.post("/api/narrative", tags=['Show'])
async def create_narrative(request: NarrativeRequest):
    """
    Turn a set of items into a narrative
    """
    narrative = narrative_generator.generate_narrative(request.items)
    return {
        "result": narrative
    }

### / Debug
@app.get("/debug/user_info", tags=["Debug"])
async def read_user(user_id: int = Query(..., example=10)):
    db_client = DB_Interface()
    db_projection = ProjectionBuilder(collection_name=COLLECTION_NAME)
    user_data = db_client.fetch_user(user_id=int(user_id))
    user_history = db_client.fetch_events_raw(user_id=int(user_id))
    user_dwell = db_client.fetch_events(user_id=int(user_id))
    user_metrics = db_projection.get_user_projection(user_id=str(user_id))
    return {
        "result": {
            "user_data": user_data.to_dict(orient="records"),
            "user_history": user_history.to_dict(orient="records"),
            "user_dwell": user_dwell.to_dict(orient="records"),
            "user_metrics": user_metrics.to_dict(orient="records"),
        }
    }

@app.get("/debug/item_info", tags=["Debug"])
async def read_item(item_id: List[int] = Query(..., example="2148")):
    item = searcher.common_searcher.get_item(item_id=item_id)
    logger.info(f"Fetched item: {item}")
    return {
        "result": item,
    }


@app.post("/db/users", tags=["Insert"])
async def create_user(user: User):
    """
    Dumb endpoint: inserts a user row via DB_Interface.register_user.
    """
    db_client = DB_Interface()
    new_id = db_client.register_user(user)
    if new_id == "failed":
        # you can customize the error contract later
        return {"status": "error", "id": None}
    return {"status": "ok", "id": new_id}


@app.post("/db/events", tags=["Insert"])
async def create_event(event: Event):
    """
    Dumb endpoint: inserts an event row via DB_Interface.register_event.
    """
    db_client = DB_Interface()
    new_id = db_client.register_event(event)
    if new_id == "failed":
        return {"status": "error", "id": None}
    return {"status": "ok", "id": new_id}



if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.service:app", host="0.0.0.0", port=8000, reload=True)