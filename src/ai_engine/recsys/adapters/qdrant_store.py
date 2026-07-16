"""Qdrant-backed ContentStore. Requires `qdrant-client` (not needed for tests).

Tags live in the point payload (decided design): a `tags` list of {facet,label,weight}
plus a flat `tag_labels` ("facet:label") KEYWORD-indexed field used for tag recall.
"""
from __future__ import annotations
import logging
import re
from typing import Optional, Sequence

from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import Filter, FieldCondition, MatchAny, GeoRadius, GeoPoint  # type: ignore

from ..contracts.enums import ContentType
from ..contracts.models import Content, Tag, Candidate, Vector
from ..taxonomy import normalize_filter_value

logger = logging.getLogger(__name__)


def _extract_latlon(payload: dict):
    """Pull (lat, lon) from the payload. Accepts a `locations` geo object/list
    ({lat,lon}) the content-engine GEO-indexed field, or flat lat/lon keys."""
    loc = payload.get("locations") or payload.get("location")
    if isinstance(loc, list) and loc:
        loc = loc[0]
    if isinstance(loc, dict) and loc.get("lat") is not None and loc.get("lon") is not None:
        try:
            return float(loc["lat"]), float(loc["lon"])
        except (TypeError, ValueError):
            pass
    lat, lon = payload.get("lat"), payload.get("lon")
    if lat is not None and lon is not None:
        try:
            return float(lat), float(lon)
        except (TypeError, ValueError):
            pass
    return None, None


def _extract_years(payload: dict) -> list[int]:
    """Creation years from `time_metadata.dates_of_creation` (list or scalar, any
    reasonable date string): every 4-digit year 18xx-20xx found, sorted, deduped."""
    tm = payload.get("time_metadata") or {}
    raw = tm.get("dates_of_creation") or tm.get("date") or []
    out = set()
    for token in (raw if isinstance(raw, list) else [raw]):
        for y in re.findall(r"\b(1[89]\d\d|20\d\d)\b", str(token)):
            out.add(int(y))
    return sorted(out)


def _payload_to_content(point_id, payload: dict) -> Content:
    """THE payload normalizer: the single place raw payload shape is interpreted
    (docs/debt-payload-scatter.md D1). Everything downstream consumes `Content`."""
    payload = payload or {}
    tags = [
        Tag(facet=t.get("facet", "unknown"), label=t.get("label", ""), weight=float(t.get("weight", 1.0)))
        for t in (payload.get("tags") or [])
    ]
    files = payload.get("files_url") or []
    image_url = payload.get("image_url") or payload.get("imageUrl") or payload.get("thumbnail_url")
    has_image = bool(image_url or (isinstance(files, list) and files))
    ctype = payload.get("content_type", ContentType.text_item.value)
    try:
        ctype = ContentType(ctype)
    except ValueError:
        ctype = ContentType.text_item
    lat, lon = _extract_latlon(payload)
    return Content(
        id=str(point_id),
        content_type=ctype,
        title=payload.get("title") or "",
        text=payload.get("text") or "",
        tags=tags,
        word_count=int(payload.get("text_length_words") or payload.get("word_count") or 0),
        has_image=has_image,
        lat=lat,
        lon=lon,
        image_url=image_url,
        public_url=payload.get("public_url") or payload.get("publicUrl"),
        years=_extract_years(payload),
    )


class QdrantContentStore:
    def __init__(self, client: QdrantClient, collection_name: str):
        self.client = client
        self.collection_name = collection_name

    @staticmethod
    def _pid(content_id: str):
        return int(content_id) if str(content_id).isdigit() else content_id

    def get(self, ids: Sequence[str]) -> dict[str, Content]:
        if not ids:
            return {}
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._pid(i) for i in ids],
            with_payload=True, with_vectors=False,
        )
        return {str(p.id): _payload_to_content(p.id, p.payload) for p in res}

    def raw_payloads(self, ids: Sequence[str]) -> dict[str, dict]:
        """Raw Qdrant payloads keyed by id (image_url / public_url / creator / time_metadata …)
       , the full content the recsys `Content` model drops. Used to open item detail cards."""
        if not ids:
            return {}
        try:
            res = self.client.retrieve(collection_name=self.collection_name,
                                       ids=[self._pid(i) for i in ids],
                                       with_payload=True, with_vectors=False)
        except Exception as exc:
            logger.warning("raw_payloads retrieve failed: %s", exc)
            return {}
        return {str(p.id): (p.payload or {}) for p in res}

    def get_vectors(self, ids: Sequence[str]) -> dict[str, Vector]:
        if not ids:
            return {}
        res = self.client.retrieve(
            collection_name=self.collection_name,
            ids=[self._pid(i) for i in ids],
            with_payload=False, with_vectors=True,
        )
        out: dict[str, Vector] = {}
        for p in res:
            v = p.vector
            if isinstance(v, dict):  # named vectors -> take first
                v = next(iter(v.values()))
            out[str(p.id)] = v
        return out

    def search_vector(self, vector: Vector, *, limit: int) -> list[Candidate]:
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=vector, limit=limit, with_payload=False, with_vectors=False,
        )
        return [
            Candidate(content_id=str(p.id), generated_by="semantic", base_score=p.score or 0.0)
            for p in res.points
        ]

    def search_tags(self, tag_keys: Sequence[str], *, limit: int) -> list[Candidate]:
        if not tag_keys:
            return []
        flt = Filter(should=[FieldCondition(key="tag_labels", match=MatchAny(any=list(tag_keys)))])
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt, limit=limit, with_payload=False, with_vectors=False,
            )
        except Exception as exc:  # collection has no `tag_labels` keyword index
            logger.warning("tag search failed (no `tag_labels` keyword index?): %s", exc)
            return []
        return [Candidate(content_id=str(p.id), generated_by="tag", base_score=0.0) for p in points]

    def sample(self, *, limit: int, exclude=()) -> list[Candidate]:
        from qdrant_client.models import SampleQuery, Sample
        ex = {str(e) for e in exclude}
        res = self.client.query_points(
            collection_name=self.collection_name,
            query=SampleQuery(sample=Sample.RANDOM),
            limit=limit + len(ex), with_payload=False, with_vectors=False,
        )
        out = []
        for p in res.points:
            if str(p.id) not in ex:
                out.append(Candidate(content_id=str(p.id), generated_by="distractor"))
            if len(out) >= limit:
                break
        return out

    def vocab(self, *, sample: int = 4000) -> dict:
        """Distinct tag vocabulary (facet -> labels, "facet:label" -> count) for the collection.
        Scrolls payloads once; used by the evaluation tool to build / validate synthetic personas."""
        from collections import Counter
        counts: Counter = Counter()
        facets: dict[str, set] = {}
        try:
            points, off = [], None
            fetched = 0
            while fetched < sample:
                page, off = self.client.scroll(
                    collection_name=self.collection_name, offset=off,
                    limit=min(512, sample - fetched), with_payload=["tags"], with_vectors=False,
                )
                if not page:
                    break
                points += page
                fetched += len(page)
                if off is None:
                    break
        except Exception as exc:
            logger.warning("vocab scroll failed: %s", exc)
            return {"facets": {}, "tags": [], "counts": {}}
        for p in points:
            for t in ((p.payload or {}).get("tags") or []):
                facet, label = t.get("facet", "unknown"), t.get("label", "")
                key = f"{facet}:{label}"
                counts[key] += 1
                facets.setdefault(facet, set()).add(label)
        return {"facets": {f: sorted(ls) for f, ls in facets.items()},
                "tags": [k for k, _ in counts.most_common()],
                "counts": dict(counts)}

    def search_filter(self, value: str, *, limit: int, exclude=()) -> list[Candidate]:
        # filter candidates by an exact tag value (e.g. a location tag AiARLocationBarrack3).
        # If the collection has no `tag_values` keyword index (e.g. content ingested without
        # the tag schema), Qdrant raises -> degrade to empty instead of 500ing the request.
        flt = Filter(must=[FieldCondition(key="tag_values", match=MatchAny(any=[normalize_filter_value(value)]))])
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt, limit=limit + len(exclude), with_payload=False, with_vectors=False,
            )
        except Exception as exc:  # missing index / unindexed field
            logger.warning("tag filter %r failed (no `tag_values` keyword index?): %s", value, exc)
            return []
        ex = {str(e) for e in exclude}
        return [Candidate(content_id=str(p.id), generated_by="filter")
                for p in points if str(p.id) not in ex][:limit]

    def search_geo(self, lat: float, lon: float, radius_m: float, *, limit: int, exclude=()) -> list[Candidate]:
        # geo radius filter on the GEO-indexed `locations` field — independent of tags.
        # If the collection has no `locations` geo index (content lacks coordinates),
        # Qdrant raises -> degrade to an empty result instead of 500ing the request.
        flt = Filter(must=[FieldCondition(
            key="locations", geo_radius=GeoRadius(center=GeoPoint(lat=lat, lon=lon), radius=radius_m))])
        try:
            points, _ = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=flt, limit=limit + len(exclude), with_payload=False, with_vectors=False,
            )
        except Exception as exc:  # missing index / unindexed field
            logger.warning("geo radius search failed (no `locations` geo index?): %s", exc)
            return []
        ex = {str(e) for e in exclude}
        return [Candidate(content_id=str(p.id), generated_by="geo")
                for p in points if str(p.id) not in ex][:limit]
