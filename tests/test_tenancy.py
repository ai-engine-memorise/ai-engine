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
