"""Runtime tenant registry store.

The DURABLE baseline is `TENANTS_PATH` (the deploy configmap). This store holds tenants
created / edited at RUNTIME (via the admin API / inspector), so a new client can be onboarded
WITHOUT a redeploy. The ComponentManager merges baseline + this store (store wins).

Like the config override, this lives in Redis: persistent enough for live edits, but the
configmap stays the source of truth, promote a tenant there to make it permanent.
"""
from __future__ import annotations
import json
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
