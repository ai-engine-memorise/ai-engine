## We need to embedd content into vector form, being easy to change model
## Push the vectors to a Qdrant collection
import os
import json
from uuid import uuid4
from dataclasses import asdict
from loguru import logger

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, PayloadSchemaType

from sentence_transformers import SentenceTransformer

import pandas as pd
import numpy as np
from ai_engine.config import QDRANT_API_URL, QDRANT_API_KEY, COLLECTION_NAME, EMBEDDING_MODEL
from ai_engine.common import Item

OMEKA_DATA_PATH = "../../../data/omeka_data.parquet"
FILTER_ITEMS_PATH = "../../../data/test_items.json"

with open(FILTER_ITEMS_PATH, 'r') as f:
    filter_ids_dict = json.load(f)
    filter_ids = set(filter_ids_dict['filter_ids'])

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

        # --- Filtering Logic Starts Here ---
        if item.id in filter_ids:
            logger.info(f"Skipping index {index} (ID: {row['id']}): Item from test collection.")
            continue
        
        #Skip if < 5 words in text_all
        # Simple word count by splitting the string by whitespace
        if (len(item.text_all.split()) < 5) and not item.image_url:
            logger.info(f"Skipping index {index} (ID: {row['id']}): Less than 5 words in text_all.")
            continue

        # Fix field 
        # TODO: fic in the omeka_data already
        if item.creator == "unbekannt":
            item.creator = ""

        # --- Filtering Logic Ends Here ---

        # logger.info(f"embedding text for item {item.id}: {item.text_all}")
        encoding_text = f"{item.text_all} - {item.creator}"
        embedding = model.encode(encoding_text, show_progress_bar=False).tolist()
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

