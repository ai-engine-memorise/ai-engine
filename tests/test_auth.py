"""Tests for the write-endpoint API-key guard (_require_api_key)."""
import pytest
from fastapi import HTTPException

from ai_engine.recsys.api import _require_api_key


def test_fails_closed_when_key_unset_and_not_dev(monkeypatch):
    monkeypatch.delenv("INGEST_API_KEY", raising=False)
    monkeypatch.delenv("AI_ENGINE_DEV", raising=False)
    with pytest.raises(HTTPException) as ei:
        _require_api_key(x_api_key=None)
    assert ei.value.status_code == 503


def test_allows_unauthenticated_in_dev(monkeypatch):
    monkeypatch.delenv("INGEST_API_KEY", raising=False)
    monkeypatch.setenv("AI_ENGINE_DEV", "1")
    assert _require_api_key(x_api_key=None) is None


def test_accepts_matching_key(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret")
    assert _require_api_key(x_api_key="secret") is None


def test_rejects_wrong_or_missing_key(monkeypatch):
    monkeypatch.setenv("INGEST_API_KEY", "secret")
    with pytest.raises(HTTPException) as ei:
        _require_api_key(x_api_key="nope")
    assert ei.value.status_code == 401
    with pytest.raises(HTTPException) as ei2:
        _require_api_key(x_api_key=None)
    assert ei2.value.status_code == 401
