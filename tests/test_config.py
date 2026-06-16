"""Tests for the runtime config endpoints (settings page backend)."""
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from test_api import _client  # reuse the in-memory client fixture


def test_get_config_returns_full_recconfig():
    client = _client()
    cfg = client.get("/api/config").json()["result"]
    assert "fusion" in cfg and "mmr_lambda" in cfg
    assert "semantic" in cfg["fusion"]


def test_put_config_applies_scalar_and_nested():
    client = _client()
    r = client.put("/api/config", json={"mmr_lambda": 0.42, "fusion": {"tag": 0.99}})
    assert r.status_code == 200
    out = r.json()["result"]
    assert out["mmr_lambda"] == 0.42
    assert out["fusion"]["tag"] == 0.99
    # persisted: a fresh GET reflects it
    assert client.get("/api/config").json()["result"]["mmr_lambda"] == 0.42


def test_put_config_rejects_invalid():
    client = _client()
    r = client.put("/api/config", json={"mmr_lambda": "not-a-number"})
    assert r.status_code == 422


def test_reset_reverts_to_baseline():
    client = _client()
    client.put("/api/config", json={"mmr_lambda": 0.123})
    assert client.get("/api/config").json()["result"]["mmr_lambda"] == 0.123
    r = client.post("/api/config/reset")
    assert r.status_code == 200
    # back to the RecConfig default (0.7)
    assert client.get("/api/config").json()["result"]["mmr_lambda"] == 0.7


def test_settings_page_served():
    client = _client()
    r = client.get("/settings")
    assert r.status_code == 200
    assert "recsys" in r.text and "X-API-Key" in r.text
