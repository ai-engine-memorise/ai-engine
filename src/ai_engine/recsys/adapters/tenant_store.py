"""Runtime tenant registry store.

Holds tenants created / edited at RUNTIME (via the admin API / inspector), so a new client
is onboarded WITHOUT a redeploy. The ComponentManager merges the `TENANTS_PATH` git baseline
with this store (store wins).

Backing, in preference order (see ComponentManager.tenant_store):
  - FileTenantStore  -> a JSON file on the api PVC (TENANT_STORE_PATH). DURABLE: survives pod
                        restarts and is immune to the recsys-redis LRU/flush, so /admin edits
                        are permanent with zero redeploy. This is the production path.
  - RedisTenantStore -> the recsys-redis (ephemeral here: `--save ""` + emptyDir + allkeys-lru),
                        so NOT durable on this deploy. Kept for setups with a persistent Redis.
  - InMemoryTenantStore -> dev / test (lost on restart).
"""
from __future__ import annotations
import json
import os
import tempfile
import threading
from typing import Optional


class RedisTenantStore:
    def __init__(self, client, *, prefix: str = "tenantcfg"):
        self.client = client
        self.prefix = prefix

    def _key(self, tid: str) -> str:
        return f"{self.prefix}:{tid}"

    def get(self, tenant_id: str) -> Optional[dict]:
        raw = self.client.get(self._key(tenant_id))
        return json.loads(raw) if raw else None

    def all(self) -> list[dict]:
        out = []
        for k in self.client.scan_iter(match=f"{self.prefix}:*"):
            raw = self.client.get(k)
            if raw:
                try:
                    out.append(json.loads(raw))
                except Exception:
                    pass
        return out

    def set(self, spec: dict) -> None:
        self.client.set(self._key(spec["tenant_id"]), json.dumps(spec))

    def delete(self, tenant_id: str) -> None:
        self.client.delete(self._key(tenant_id))


class FileTenantStore:
    """Durable runtime store: one JSON object {tenant_id: spec} on a persistent volume.

    Re-reads the file per op so a restarted / second replica sees the latest, and writes
    atomically (temp file + os.replace) so a crash mid-write can't corrupt the registry.
    """
    def __init__(self, path: str):
        self.path = path
        self._lock = threading.Lock()
        d = os.path.dirname(path)
        if d:
            os.makedirs(d, exist_ok=True)

    def _load(self) -> dict[str, dict]:
        try:
            with open(self.path, encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, dict) else {}
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    def _save(self, data: dict[str, dict]) -> None:
        d = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    def get(self, tenant_id: str) -> Optional[dict]:
        return self._load().get(tenant_id)

    def all(self) -> list[dict]:
        return list(self._load().values())

    def set(self, spec: dict) -> None:
        with self._lock:
            data = self._load()
            data[spec["tenant_id"]] = dict(spec)
            self._save(data)

    def delete(self, tenant_id: str) -> None:
        with self._lock:
            data = self._load()
            if data.pop(tenant_id, None) is not None:
                self._save(data)


class InMemoryTenantStore:
    """Dev / test fallback (no Redis)."""
    def __init__(self) -> None:
        self._d: dict[str, dict] = {}

    def get(self, tenant_id: str) -> Optional[dict]:
        return self._d.get(tenant_id)

    def all(self) -> list[dict]:
        return list(self._d.values())

    def set(self, spec: dict) -> None:
        self._d[spec["tenant_id"]] = dict(spec)

    def delete(self, tenant_id: str) -> None:
        self._d.pop(tenant_id, None)
