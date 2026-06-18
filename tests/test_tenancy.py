"""Multi-tenancy: per-tenant isolation of user models via the X-Tenant-Id header."""
from datetime import datetime, timezone, timedelta

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient

from ai_engine.recsys.api import create_app
from ai_engine.recsys.tenancy import TenantRegistry, TenantSpec


def _ev(evt, uid, ts, reason=None, dwell=None):
    props = {"content": {"content_id": "A1"}}
    details = {}
    if reason:
        details["reason"] = reason
    if dwell is not None:
        details["dwell_seconds"] = dwell
    if details:
        props["details"] = details
    return {"event": evt, "userId": uid, "timestamp": ts, "properties": props}


def test_registry_default_and_unknown_fallback():
    reg = TenantRegistry({"default": TenantSpec("default"), "alpha": TenantSpec("alpha", collection="c_alpha")}, "default")
    assert reg.get("alpha").collection == "c_alpha"
    assert reg.get(None).tenant_id == "default"
    # unknown tenant -> auto-isolated slice (own id/prefix), inheriting the default catalogue
    assert reg.get("ghost").tenant_id == "ghost"


def test_tenant_admin_runtime_crud():
    client = TestClient(create_app())                    # manager-backed
    r = client.post("/api/tenants", json={"tenant_id": "westerbork-ar", "collection": "westerbork-ar"})
    assert r.status_code == 200 and r.json()["status"] == "saved"
    ids = {t["tenant_id"] for t in client.get("/api/tenants").json()["result"]}
    assert "westerbork-ar" in ids and "default" in ids   # runtime + config baseline
    client.delete("/api/tenants/westerbork-ar")
    ids2 = {t["tenant_id"] for t in client.get("/api/tenants").json()["result"]}
    assert "westerbork-ar" not in ids2


def test_per_tenant_key_pins_tenant_over_spoofed_header():
    """A per-tenant API key derives the tenant; a spoofed X-Tenant-Id is ignored, so the
    key can only ever write/read its OWN slice (cross-tenant write hole closed)."""
    client = TestClient(create_app())
    r = client.post("/api/tenants", json={"tenant_id": "alpha", "collection": "c_alpha",
                                          "api_keys": ["alpha-secret"]})
    assert r.status_code == 200

    ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    payload = [_ev("CONTENT_VIEW_STARTED", "u1", ts),
               _ev("CONTENT_VIEW_ENDED", "u1", ts, reason="next_button", dwell=120)]
    # present alpha's key but SPOOF the header as beta -> must land in ALPHA, not beta
    r = client.post("/api/ingest", json=payload,
                    headers={"X-API-Key": "alpha-secret", "X-Tenant-Id": "beta"})
    assert r.json()["ingested"] == 2

    # the key pins to alpha regardless of the header -> alpha has the model
    a = client.get("/api/usermodel", params={"user_id": "u1"},
                   headers={"X-API-Key": "alpha-secret", "X-Tenant-Id": "beta"}).json()["result"]
    assert a is not None
    # nothing leaked into beta
    b = client.get("/api/usermodel", params={"user_id": "u1"}, headers={"X-Tenant-Id": "beta"}).json()["result"]
    assert b is None


def test_generated_key_is_hashed_and_pins_tenant():
    """Server mints a key, returns it ONCE, persists only its hash; the key still pins
    the tenant and the admin list never leaks the key or its hash."""
    client = TestClient(create_app())
    body = client.post("/api/tenants",
                       json={"tenant_id": "gamma", "collection": "c_gamma",
                             "generate_api_key": True}).json()
    key = body["api_key"]
    assert key and "api_keys" not in body["result"] and "api_key_hashes" not in body["result"]

    g = next(t for t in client.get("/api/tenants").json()["result"] if t["tenant_id"] == "gamma")
    assert "api_keys" not in g and "api_key_hashes" not in g and g["api_keys_count"] == 1

    ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    payload = [_ev("CONTENT_VIEW_STARTED", "u9", ts),
               _ev("CONTENT_VIEW_ENDED", "u9", ts, reason="next_button", dwell=99)]
    # generated key pins to gamma even with a spoofed header
    assert client.post("/api/ingest", json=payload,
                       headers={"X-API-Key": key, "X-Tenant-Id": "delta"}).json()["ingested"] == 2
    a = client.get("/api/usermodel", params={"user_id": "u9"}, headers={"X-API-Key": key}).json()["result"]
    assert a is not None


def test_unknown_api_key_rejected():
    """A key that matches neither a per-tenant key nor the global key is 401, even in dev."""
    client = TestClient(create_app())
    r = client.post("/api/ingest", json=[], headers={"X-API-Key": "bogus", "X-Tenant-Id": "default"})
    assert r.status_code == 401


def test_user_models_isolated_by_tenant():
    client = TestClient(create_app())                    # no fixed components -> tenant manager + X-Tenant-Id
    ts = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    payload = [_ev("CONTENT_VIEW_STARTED", "u1", ts),
               _ev("CONTENT_VIEW_ENDED", "u1", ts, reason="next_button", dwell=120)]

    r = client.post("/api/ingest", json=payload, headers={"X-Tenant-Id": "alpha"}).json()
    assert r["ingested"] == 2

    a = client.get("/api/usermodel", params={"user_id": "u1"}, headers={"X-Tenant-Id": "alpha"}).json()["result"]
    b = client.get("/api/usermodel", params={"user_id": "u1"}, headers={"X-Tenant-Id": "beta"}).json()["result"]
    d = client.get("/api/usermodel", params={"user_id": "u1"}).json()["result"]   # default tenant
    assert a is not None                                 # tenant alpha has the model
    assert b is None                                     # tenant beta is isolated
    assert d is None                                     # default tenant is isolated too
