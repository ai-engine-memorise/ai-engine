"""Multi-tenancy: serve many clients (UIs / memorial sites) from one engine, isolated.

Each tenant gets its OWN Qdrant collection (content can't cross) and its OWN Redis
key-prefix (user models / events / impressions / config can't cross), plus its own
bandit policy and cluster model. Shared infra, hard logical isolation — the standard
"pooled" multi-tenant model.

The engine core (recommender, scorers, user model) is already tenant-agnostic: it
operates on whatever stores it is handed. Tenancy is purely an edge concern:

    request -> X-Tenant-Id header -> current_tenant contextvar
            -> ComponentManager.get(tenant) -> tenant-scoped Components

A `TenantProxy` lets every endpoint keep using `c.recommender` / `c.cfg` etc. while
those resolve to the CURRENT request's tenant, so no per-endpoint changes are needed.
"""
from __future__ import annotations
import hmac
import json
import os
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional

# set per request from the X-Tenant-Id header; "default" preserves single-tenant behaviour.
current_tenant: ContextVar[str] = ContextVar("current_tenant", default="default")

# how the request authenticated (set by the middleware, read by the API-key guard):
#   "*"        -> the global INGEST_API_KEY (superuser; may target any tenant via header)
#   "<tenant>" -> a per-tenant key; the request is PINNED to that tenant (header ignored)
#   None       -> no/invalid key (public reads only; guarded routes -> 401)
auth_tenant: ContextVar[Optional[str]] = ContextVar("auth_tenant", default=None)


@dataclass
class TenantSpec:
    """Everything that distinguishes one client's slice of the engine."""
    tenant_id: str
    collection: Optional[str] = None              # Qdrant collection (None -> dev fakes)
    redis_prefix: Optional[str] = None            # defaults to tenant_id
    bandit_state_path: Optional[str] = None
    cluster_model_path: Optional[str] = None
    config_overrides: dict = field(default_factory=dict)
    api_keys: list[str] = field(default_factory=list)   # per-tenant keys; presenting one PINS the request to this tenant

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
    """Pure ASGI middleware: the per-request TRUST BOUNDARY for tenant + auth.

    Derives the tenant so it can't be spoofed by a client header:
      - a valid per-tenant API key  -> tenant is taken FROM THE KEY (X-Tenant-Id ignored),
        so a tenant's key can only ever read/write its own slice.
      - the global INGEST_API_KEY    -> superuser; tenant from X-Tenant-Id (admin/ops).
      - otherwise                    -> X-Tenant-Id (public reads; guarded routes 401 later).

    `key_resolver(key) -> tenant_id | None` maps a per-tenant key to its tenant. A pure ASGI
    middleware runs in the SAME context as the request, so both contextvars propagate into the
    (threadpool-run) endpoint — unlike BaseHTTPMiddleware, which would lose them.
    """
    def __init__(self, app, key_resolver: Optional[Callable[[str], Optional[str]]] = None):
        self.app = app
        self.key_resolver = key_resolver

    async def __call__(self, scope, receive, send):
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        headers = {k.decode("latin1").lower(): v.decode("latin1") for k, v in scope.get("headers", [])}
        key = headers.get("x-api-key")
        hdr_tenant = headers.get("x-tenant-id") or "default"

        key_tenant = self.key_resolver(key) if (key and self.key_resolver) else None
        if key_tenant is not None:                       # per-tenant key -> authoritative tenant
            tenant, auth = key_tenant, key_tenant
        elif key and _is_global_key(key):                # superuser key -> tenant from header
            tenant, auth = hdr_tenant, "*"
        else:                                            # no/unknown key -> header (public) , unauthenticated
            tenant, auth = hdr_tenant, None

        t1 = current_tenant.set(tenant)
        t2 = auth_tenant.set(auth)
        try:
            await self.app(scope, receive, send)
        finally:
            current_tenant.reset(t1)
            auth_tenant.reset(t2)


def _is_global_key(key: str) -> bool:
    expected = os.getenv("INGEST_API_KEY")
    return bool(expected) and hmac.compare_digest(key, expected)
