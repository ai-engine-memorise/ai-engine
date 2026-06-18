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

## Configuring a client (what developers must do)

The `X-Tenant-Id` header is documented on every `/api/*` operation in **Swagger / OpenAPI**
(`/docs`). Every request AND every event must carry it.

### 1. Register the tenant
Add it to `TENANTS_PATH` (collection + per-tenant paths), then redeploy. Ingest the tenant's
content into its Qdrant collection (content-engine, per tenant).

### 2. Frontend / grid (serving calls)
Add the header to every `/api/*` call:
```js
fetch(`${API}/api/recommend?user_id=${uid}`, {
  headers: { "X-Tenant-Id": TENANT, "X-API-Key": KEY }   // X-API-Key only on guarded routes
})
```

### 3. RudderStack (events -> /api/ingest)
The tenant rides on the webhook. Recommended: **one RudderStack Source per tenant.**
1. Create (or reuse) a **Source** for the tenant; note its write key (the app / survey-viewer uses it).
2. Add a **Webhook destination** on that Source, URL = `https://<host>/api/ingest`.
3. On that destination add custom **Headers**:
   - `X-Tenant-Id: <tenant>`
   - `X-API-Key: <INGEST_API_KEY>` (if set)
4. All events from that Source now land in the right tenant. No per-event logic.

If you instead run **one shared Source** for all tenants, a webhook can't vary a header per
event, so put `tenant_id` in the event payload and enable the tenant-from-payload fallback below.

### 4. Survey-viewer
Point it at the tenant's RudderStack **write key**. The tenant attaches at that Source's webhook
destination header (step 3). No survey-viewer code change.

## Tenant-from-payload fallback (optional)

By default the tenant is read from the `X-Tenant-Id` **header**. When events arrive via a single
shared webhook that cannot set a per-event header, the engine can instead read `tenant_id` from the
event body (e.g. `context.tenant_id`) and route each event to its tenant. Header takes precedence;
payload is the fallback. Not enabled by default (small change to the ingest path). Prefer
per-tenant Sources where possible.

## Tooling

- **Inspector** (`/inspector`) and **Settings** (`/settings`) both have a **tenant** field
  that sends `X-Tenant-Id`, so you can inspect / configure one client at a time.

## Known limits (v1)

- In-process `/api/metrics` counters are **global** (not per-tenant) — fine for ops, not for
  per-client billing. Prometheus labels are the prod upgrade.
- Demographics provider is shared; per-tenant DB wiring is future work.
