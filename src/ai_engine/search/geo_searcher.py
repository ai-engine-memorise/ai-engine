# Given Current Real or Virtual Location query on Qdrant Proximal Items
from qdrant_client import QdrantClient
from qdrant_client.models import (
    GeoRadius,
    GeoPoint,
    Filter,
    FieldCondition,
    SearchParams,
)
from ai_engine.config import QDRANT_API_URL, QDRANT_API_KEY, COLLECTION_NAME, FIELD_NAME_GEO, SEARCH_LIMIT
from ai_engine.common import SearchResult, hit_to_item 

#################
#### Qdrant  ####
#################

# -------------------------------------------------------------------------------------------------------------------------------

class GeoSearch:
    def __init__(self, collection_name: str = COLLECTION_NAME):
        """
        Simple helper class for doing geo-based searches in Qdrant.

        :param api_url: Qdrant API URL
        :param api_key: Qdrant API key
        :param collection_name: Name of the collection to search
        """
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=QDRANT_API_URL,
            api_key=QDRANT_API_KEY
        )

    def geo_filter(
        self,
        center_lat: float,
        center_lon: float,
        radius_meters: float,
    ) -> Filter:
        """
        Build a geo-radius filter.
        """
        return Filter(
            must=[
                FieldCondition(
                    key=FIELD_NAME_GEO,
                    geo_radius=GeoRadius(
                        center=GeoPoint(
                            lat=center_lat,
                            lon=center_lon
                        ),
                        radius=radius_meters
                    )
                )
            ]
        )

    def search(
        self,
        lat: float,
        lon: float,
        radius_meters: float = 5000,
    ):
        """
        Pure geo search using scroll (no vector similarity).

        :returns: (points, next_offset)
        """
        geo_filter = self.geo_filter(
            center_lat=lat,
            center_lon=lon,
            radius_meters=radius_meters,
        )

        points, next_offset = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=geo_filter,
            limit=SEARCH_LIMIT,
            with_payload=True,
            with_vectors=False,
        ) # [list[Record], qdrant_client.conversions.common_types.PointId | None]
        
        items = [
            hit_to_item(hit=p, source="geo", query_text=None)
            for p in points
        ]

        return SearchResult(
            search_type="geo",
            query_text=None,
            lat=lat,
            lon=lon,
            radius_meters=radius_meters,
            items=items,
            next_offset=next_offset,
        )

    def hybrid_search(
        self,
        query_vector,
        lat: float,
        lon: float,
        radius_meters: float = 5000,
        limit: int = SEARCH_LIMIT,
        search_params: SearchParams | None = None,
        query_text: str | None = None,
    ):
        """
        Example hybrid search: vector similarity + geo constraint.
        """
        geo_filter = self.geo_filter(
            center_lat=lat,
            center_lon=lon,
            radius_meters=radius_meters,
        )

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=geo_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
            search_params=search_params,
        ).points # list[ScoredPoint]

        items = [
            hit_to_item(hit=hit, source="hybrid", query_text=query_text)
            for hit in results
        ]

        return SearchResult(
            search_type="hybrid",
            query_text=query_text,
            lat=lat,
            lon=lon,
            radius_meters=radius_meters,
            items=items,
            next_offset=None,
        )


# ---------- Example usage ----------

if __name__ == '__main__':

    geo_searcher = GeoSearch(
        collection_name=COLLECTION_NAME,
    )
    
    # geo_searcher.client.create_payload_index(
    #     collection_name="omeka_items",
    #     field_name="time_metadata.dates_of_creation",
    #     field_schema="datetime"
    # )
    # geo_searcher.client.create_payload_index(
    #     collection_name="omeka_items",
    #     field_name="locations",
    #     field_schema="geo",
    # )
    
    # Bergen-Belsen Memorial coordinates
    center_lat = 52.757778
    center_lon = 9.907778
    radius_meters = 5000
    results, next_offset = geo_searcher.search(
        lat=center_lat,
        lon=center_lon,
        radius_meters=radius_meters,
    )
    
    for res in results:
        print(f"> {res.payload['title']} : \n{res.payload['locations']} \n{res.payload['public_url']}")
        print("")
