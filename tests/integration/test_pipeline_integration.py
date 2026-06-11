"""Layer-4 integration: replay RudderStack payloads through the REAL adapters
(Redis event buffer + Redis user-model store + Qdrant content store).

Skipped unless infra is up and RUN_INTEGRATION=1:
    docker compose -f tests/integration/docker-compose.yml up -d
    RUN_INTEGRATION=1 python -m pytest tests/integration -q

Proves the same chain as production minus RudderStack itself (whose webhook just
forwards these exact payloads):  ingest webhook -> Redis -> user model -> Qdrant
tag/semantic match -> recommendation.
"""
from __future__ import annotations
import os
import socket
from datetime import datetime, timezone, timedelta

import pytest

QDRANT_HOST, QDRANT_PORT = "localhost", 6333
REDIS_HOST, REDIS_PORT = "localhost", 6379
COLLECTION = "recsys_itest"


def _reachable(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(
    os.getenv("RUN_INTEGRATION") != "1"
    or not _reachable(QDRANT_HOST, QDRANT_PORT)
    or not _reachable(REDIS_HOST, REDIS_PORT),
    reason="integration infra not up (set RUN_INTEGRATION=1 and start docker-compose)",
)


@pytest.fixture(scope="module")
def client():
    pytest.importorskip("qdrant_client")
    pytest.importorskip("redis")
    pytest.importorskip("httpx")

    os.environ["QDRANT_API_URL"] = f"http://{QDRANT_HOST}:{QDRANT_PORT}"
    os.environ["COLLECTION_NAME"] = COLLECTION
    os.environ["REDIS_URL"] = f"redis://{REDIS_HOST}:{REDIS_PORT}/0"

    import sys
    sys.path.insert(0, os.path.dirname(__file__))  # make seed_qdrant importable

    import redis
    from qdrant_client import QdrantClient
    from fastapi.testclient import TestClient

    from seed_qdrant import seed
    from ai_engine.recsys.api import create_app
    from ai_engine.recsys.composition import build_components

    # fresh state
    seed(QdrantClient(url=os.environ["QDRANT_API_URL"]), COLLECTION)
    redis.from_url(os.environ["REDIS_URL"]).flushdb()

    return TestClient(create_app(build_components()))


def _view(event, cid, ts, reason=None, dwell=None):
    props = {"content": {"content_id": f"content_{cid}"}}
    details = {}
    if reason:
        details["reason"] = reason
    if dwell is not None:
        details["dwell_seconds"] = dwell
    if details:
        props["details"] = details
    return {"event": event, "userId": "itest_user", "timestamp": ts, "properties": props}


def test_full_pipeline_real_redis_and_qdrant(client):
    ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    payload = []
    for cid in ("101", "102"):
        payload.append(_view("CONTENT_VIEW_STARTED", cid, ts))
        payload.append(_view("CONTENT_VIEW_ENDED", cid, ts, reason="next_button", dwell=120))

    # webhook -> Redis buffer -> rebuild user model -> Redis model store
    r = client.post("/api/ingest", json=payload)
    assert r.status_code == 200 and r.json()["ingested"] == 4

    # user model materialized in REAL Redis
    um = client.get("/api/usermodel", params={"user_id": "itest_user"}).json()["result"]
    assert um and "101" in um["positives"] and "102" in um["positives"]
    assert any(k.startswith("theme_what:Forced Labor") for k in um["tag_affinity"])

    # recommend from REAL Qdrant: unseen Forced-Labor sibling surfaces, seen excluded
    rec = client.get("/api/recommend", params={"user_id": "itest_user"}).json()["result"]
    ids = [it["content_id"] for it in rec["items"]]
    assert rec["strategy"] == "warm"
    assert "103" in ids
    assert "101" not in ids and "102" not in ids
