## We need to embedd content into vector form, being easy to change model
## Push the vectors to a Qdrant collection
# %%
from loguru import logger

from qdrant_client import QdrantClient
from ai_engine.config import QDRANT_API_URL, QDRANT_API_KEY, COLLECTION_NAME

if __name__ == '__main__':

    # Init client
    logger.info(f"Qdrant client initialized for URL: {QDRANT_API_URL}")
    client = QdrantClient(
        url=QDRANT_API_URL,
        api_key=QDRANT_API_KEY
    )

    logger.info("Creating payload index on 'time_metadata. dates_of_creation'")
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


