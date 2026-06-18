"""Multi-tenancy: serve many clients (UIs / memorial sites) from one engine, isolated.

Each tenant gets its OWN Qdrant collection (content can't cross) and its OWN Redis
key-prefix (user models / events / impressions / config can't cross), plus its own
bandit policy and cluster model. Shared infra, hard logical isolation — the standard
SaaS "pooled" model.

The engine core (recommender, scorers, user model) is already tenant-agnostic: it
operates on whatever stores it is handed. Tenancy is purely an edge concern:

    request -> X-Tenant-Id header -> current_tenant contextvar
            -> ComponentManager.get(tenant) -> tenant-scoped Components

A `TenantProxy` lets every endpoint keep using `c.recommender` / `c.cfg` etc. while
those resolve to the CURRENT request's tenant, so no per-endpoint changes are needed.
"""
from __future__ import annotations
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

# set per request from the X-Tenant-Id header; "default" preserves single-tenant behaviour.
current_tenant: ContextVar[str] = ContextVar("current_tenant", default="default")


@dataclass
class TenantSpec:
    """Everything that distinguishes one client's slice of the engine."""
    tenant_id: str
    collection: Optional[str] = None              # Qdrant collection (None -> dev fakes)
    redis_prefix: Optional[str] = None            # defaults to tenant_id
    bandit_state_path: Optional[str] = None
    cluster_model_path: Optional[str] = None
    config_overrides: dict = field(default_factory=dict)

    @property
    def prefix(self) -> str:
        return self.redis_prefix or self.tenant_id


class TenantRegistry:
    """tenant_id -> TenantSpec. Loaded from TENANTS_PATH (JSON) or a single 'default'
    tenant built from the existing env (so single-tenant deployments are unchanged)."""

    def __init__(self, specs: dict[str, TenantSpec], default_id: str = "default", strict: bool = False):
        self._specs = specs
        self.default_id = default_id
        self.strict = strict          # when loaded from TENANTS_PATH: unknown tenants are rejectable

    def get(self, tenant_id: Optional[str]) -> TenantSpec:
        tid = tenant_id or self.default_id
        if tid in self._specs:
            return self._specs[tid]
        # unknown tenant -> auto-isolated slice: its OWN redis prefix (no cross-tenant sharing),
        # inheriting the default's catalogue + config. (strict registries can reject these.)
        base = self._specs.get(self.default_id) or TenantSpec(self.default_id)
        return TenantSpec(tenant_id=tid, collection=base.collection,
                          config_overrides=dict(base.config_overrides))

    def ids(self) -> list[str]:
        return list(self._specs)

    def known(self, tenant_id: Optional[str]) -> bool:
        return (tenant_id or self.default_id) in self._specs

    @classmethod
    def from_env(cls) -> "TenantRegistry":
        path = os.getenv("TENANTS_PATH")
        if path and os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
            specs = {t["tenant_id"]: TenantSpec(**t) for t in data.get("tenants", [])}
            default_id = data.get("default", "default")
            specs.setdefault(default_id, TenantSpec(tenant_id=default_id))
            return cls(specs, default_id, strict=True)
        # back-compat single tenant from the current env
        spec = TenantSpec(
            tenant_id="default",
            collection=os.getenv("COLLECTION_NAME"),
            bandit_state_path=os.getenv("BANDIT_STATE_PATH"),
            cluster_model_path=os.getenv("CLUSTER_MODEL_PATH"),
        )
        return cls({"default": spec}, "default")


class TenantProxy:
    """Delegates attribute access to the CURRENT request's tenant Components, so the
    API endpoints can keep referencing `c.<attr>` unchanged across tenants."""

    def __init__(self, manager):
        object.__setattr__(self, "_manager", manager)

    def __getattr__(self, name):
        return getattr(self._manager.get(current_tenant.get()), name)


class TenantASGIMiddleware:
    """Pure ASGI middleware that sets `current_tenant` from the X-Tenant-Id header.

    A pure ASGI middleware runs in the SAME context as the request, so the contextvar
    propagates into the (threadpool-run) endpoint — unlike BaseHTTPMiddleware, which
    runs the endpoint in a separate task and would lose it.
    """
    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        token = current_tenant.set(headers.get("x-tenant-id") or "default")
        try:
            await self.app(scope, receive, send)
        finally:
            current_tenant.reset(token)
