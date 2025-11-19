## We need to embedd content into vector form, being easy to change model
## Push the vectors to a Qdrant collection
from uuid import uuid4
import json
from dataclasses import asdict
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType

from sentence_transformers import SentenceTransformer

import pandas as pd
from ai_engine.config import QDRANT_API_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL
from ai_engine.common import Item

def to_serializable(obj):
    # NumPy arrays -> Python lists
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # NumPy scalar types -> Python scalars
    if isinstance(obj, np.generic):
        return obj.item()
    # You can add more special cases here (datetime, Decimal, etc.)
    raise TypeError(f"Type {type(obj)} not serializable")

# TODO: Add content medium as metadata (text, image, audio, video, mixed)
# TODO: Compute item length in words

if __name__ == '__main__':

    ##################
    ####  Omeka  #####
    ##################
    df = pd.read_parquet("../../../data/omeka_data.parquet")

    # Init Model
    logger.info(f"Initializing embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)  # 384-dim vector
    logger.info(f"Dataframe loaded. Total rows: {len(df)}")

    # Prepare vectors and payloads
    logger.info("Starting embedding and payload construction loop.")
    points = []
    for index, row in df.iterrows():
        item = Item(**row.to_dict())
        logger.info(f"embedding text for item {item.id}: {item.text_all}")
        embedding = model.encode(item.text_all, show_progress_bar=False).tolist()
        payload = asdict(item)
        payload["image_url"] = item.image_url
        try:
            payload = json.loads(json.dumps(payload, default=to_serializable))
        except TypeError as e:
            # A cleaner error handler if the json module itself fails to serialize a weird type
            logger.error(f"Failed to serialize payload for Item ID: {item.id}. Error: {e}")
            logger.info(item)
            continue # Skip this item or handle the error appropriately

        points.append(
            PointStruct(
                id=row['id'],    #str(uuid4()),  # or use row["id"] if it's unique
                vector=embedding,
                payload=payload
            )
        )
    logger.info(f"Total points prepared for upsert: {len(points)}")

    #################
    #### Qdrant  ####
    #################

    # Init client
    logger.info(f"Qdrant client initialized for URL: {QDRANT_API_URL}")
    client = QdrantClient(
        url=QDRANT_API_URL,
        api_key=QDRANT_API_KEY
    )

    vector_size = len(points[0].vector) if points else model.get_sentence_embedding_dimension()
    logger.info(f"Preparing Qdrant collection '{COLLECTION_NAME}' with vector size {vector_size}")

    # Check if collection exists
    if not client.collection_exists(collection_name=COLLECTION_NAME):
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE)
        )

    logger.info(f"Starting upsert of {len(points)} vectors to '{COLLECTION_NAME}'")
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points
    )
    logger.info("Upsert completed successfully.")

    logger.info("Creating payload index on 'time_metadata.dates_of_creation'")
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="time_metadata.dates_of_creation",
        field_schema="datetime"
    )
    client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="locations",
        field_schema="geo",
    )

    logger.info("Payload index created.")
    logger.info("Data ingestion process completed.")

