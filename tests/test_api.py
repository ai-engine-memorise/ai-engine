from datetime import datetime, timezone, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from ai_engine.recsys.api import create_app
from ai_engine.recsys.composition import Components
from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.contracts.models import Content, Tag
from ai_engine.recsys.recommender import Recommender
from ai_engine.recsys.updater import UserModelUpdater
from ai_engine.recsys.testing.fakes import (
    FakeContentStore, FakeEventSource, InMemoryUserModelStore,
)
from ai_engine.recsys.adapters.demographics import (
    NullDemographicsProvider, StaticDemographicsProvider,
)

FORCED = Tag(facet="theme_what", label="Forced Labor", weight=1.0)
FAMILY = Tag(facet="theme_what", label="Family", weight=1.0)


def _content(cid, tag, axis):
    vec = [0.0, 0.0]
    vec[axis] = 1.0
    return Content(id=cid, title=cid, text=cid, word_count=120, tags=[tag]), vec


def _client(demographics=None):
    world = dict([
        ("101", _content("101", FORCED, 0)),
        ("102", _content("102", FORCED, 0)),
        ("103", _content("103", FORCED, 0)),
        ("201", _content("201", FAMILY, 1)),
        ("202", _content("202", FAMILY, 1)),
    ])
    contents = {k: v[0] for k, v in world.items()}
    vectors = {k: v[1] for k, v in world.items()}
    cfg = RecConfig()
    store = FakeContentStore(contents, vectors)
    buf = FakeEventSource()
    models = InMemoryUserModelStore()
    components = Components(
        cfg=cfg, content_store=store, event_buffer=buf, model_store=models,
        updater=UserModelUpdater(store, models, cfg),
        recommender=Recommender(store, models, cfg),
        demographics=demographics or NullDemographicsProvider(),
    )
    return TestClient(create_app(components))


def _view(event, cid, ts, reason=None, dwell=None):
    props = {"content": {"content_id": f"content_{cid}"}}
    details = {}
    if reason:
        details["reason"] = reason
    if dwell is not None:
        details["dwell_seconds"] = dwell
    if details:
        props["details"] = details
    return {"event": event, "userId": "u1", "timestamp": ts, "properties": props}


def test_ingest_webhook_then_recommend():
    client = _client()
    now = datetime.now(timezone.utc) - timedelta(minutes=1)
    ts = now.isoformat()

    payload = []
    for cid in ("101", "102"):
        payload.append(_view("CONTENT_VIEW_STARTED", cid, ts))
        payload.append(_view("CONTENT_VIEW_ENDED", cid, ts, reason="next_button", dwell=120))

    r = client.post("/api/ingest", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["ingested"] == 4
    assert body["users"] == ["u1"]

    # user model materialized
    um = client.get("/api/usermodel", params={"user_id": "u1"}).json()["result"]
    assert um is not None
    assert "101" in um["positives"] and "102" in um["positives"]

    # recommend surfaces the unseen Forced-Labor story, excludes seen
    rec = client.get("/api/recommend", params={"user_id": "u1"}).json()["result"]
    ids = [it["content_id"] for it in rec["items"]]
    assert rec["strategy"] == "warm"
    assert "103" in ids
    assert "101" not in ids and "102" not in ids


def test_recommend_preview_hand_authored_model():
    client = _client()
    # tag-only model -> forced-labour items surface
    r = client.post("/api/recommend/preview",
                    json={"tag_affinity": {"theme_what:Forced Labor": 1.0}, "limit": 5})
    assert r.status_code == 200
    ids = [i["content_id"] for i in r.json()["result"]["items"]]
    assert any(i in ids for i in ["101", "102", "103"])
    # like_items -> taste vector built + those items excluded as seen
    r2 = client.post("/api/recommend/preview", json={"like_items": ["101"]})
    res2 = r2.json()["result"]
    assert "101" not in [i["content_id"] for i in res2["items"]]
    assert res2["user_model"]["taste_vector"] is not None


def test_recommend_compact_omits_content():
    client = _client()
    r = client.post("/api/recommend/preview", json={"tag_affinity": {"theme_what:Forced Labor": 1.0}})
    full = r.json()["result"]
    r2 = client.post("/api/recommend/preview", params={"include_content": "false"},
                     json={"tag_affinity": {"theme_what:Forced Labor": 1.0}})
    compact = r2.json()["result"]
    assert full["items"] and "content" in full["items"][0]
    assert compact["items"] and "content" not in compact["items"][0]
    assert "content_id" in compact["items"][0] and "final_score" in compact["items"][0]


def test_recommend_unknown_user_cold_start():
    client = _client()
    rec = client.get("/api/recommend", params={"user_id": "nobody"}).json()["result"]
    assert rec["strategy"] == "cold"
    assert rec["items"]                                    # cold-start fallback, not empty
    assert rec["diagnostics"].get("cold_start_fallback") is True


def test_demographics_reach_user_model():
    demo = StaticDemographicsProvider({"u1": {"age": 20, "gender": "female", "nationality": "dutch"}})
    client = _client(demographics=demo)
    ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    client.post("/api/ingest", json=[_view("CONTENT_VIEW_STARTED", "101", ts)])
    um = client.get("/api/usermodel", params={"user_id": "u1"}).json()["result"]
    assert any(k.startswith("person_who") for k in um["tag_affinity"])  # cold-start bridge live


def test_ingest_requires_api_key_when_set(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret")
    client = _client()
    ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    body = [_view("CONTENT_VIEW_STARTED", "101", ts)]
    assert client.post("/api/ingest", json=body).status_code == 401
    assert client.post("/api/ingest", json=body, headers={"X-API-Key": "secret"}).status_code == 200
