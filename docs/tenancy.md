# Multi-tenancy (SaaS)

One engine serves many clients (UIs / memorial sites), each **hard-isolated**: its own
Qdrant collection (content can't cross), its own Redis key-prefix (user models, events,
impressions, config can't cross), and its own bandit policy + cluster model. Shared infra,
logical isolation — the standard SaaS "pooled" model.

## How a request is routed

```
request --X-Tenant-Id: <id>--> current_tenant contextvar
        --> ComponentManager.get(<id>) --> tenant-scoped Components
```

Every request (and event) carries `X-Tenant-Id`. A pure-ASGI middleware sets a contextvar;
a `TenantProxy` resolves `c.<attr>` to that tenant's Components, so the API endpoints are
unchanged. No header → the **`default`** tenant (single-tenant behaviour, unchanged).

The engine core (recommender, scorers, user model) is tenant-agnostic — tenancy is purely
an edge concern (`tenancy.py` + `composition.ComponentManager`).

## What's isolated per tenant

| Axis | Keying |
|---|---|
| Content catalogue | its own Qdrant `collection` |
| User models / events / impressions | Redis prefix `{tenant}:umodel|evt|imp` |
| Runtime config (settings page) | Redis key `{tenant}:recsys:config` |
| Bandit policy (θ) | `bandit_state_path` per tenant |
| Cluster model | `cluster_model_path` per tenant |
| Durable event log | `EVENT_LOG_DIR/{tenant}/...` |

## Registering tenants

Set `TENANTS_PATH` to a JSON file:

```json
{
  "default": "westerbork",
  "tenants": [
    {"tenant_id": "westerbork",   "collection": "westerbork",
     "bandit_state_path": "/data/westerbork/bandit.json",
     "cluster_model_path": "/data/westerbork/clusters.json"},
    {"tenant_id": "bergen-belsen", "collection": "bergen-belsen",
     "bandit_state_path": "/data/bb/bandit.json"}
  ]
}
```

- **No `TENANTS_PATH`** → a single `default` tenant built from the existing env
  (`COLLECTION_NAME`, `BANDIT_STATE_PATH`, `CLUSTER_MODEL_PATH`). Existing deployments are
  unchanged.
- **Unknown tenant id** → an auto-isolated slice (its own Redis prefix, inheriting the
  default catalogue). When `TENANTS_PATH` is set the registry is `strict` (you can reject
  unknown ids at the edge to bound the in-memory cache).

## The contract each UI must honour

`user_id` only means something **within** a tenant; `content_id` only resolves within that
tenant's collection. So every UI must send a consistent `X-Tenant-Id` (and agree on the
`user_id` / `content_id` namespaces for that tenant).

## Tooling

- **Inspector** (`/inspector`) and **Settings** (`/settings`) both have a **tenant** field
  that sends `X-Tenant-Id`, so you can inspect / configure one client at a time.

## Known limits (v1)

- In-process `/api/metrics` counters are **global** (not per-tenant) — fine for ops, not for
  per-client billing. Prometheus labels are the prod upgrade.
- Demographics provider is shared; per-tenant DB wiring is future work.
