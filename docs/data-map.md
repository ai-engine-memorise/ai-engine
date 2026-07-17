# Data Map — where every piece of data lives and how it feeds recommendations

Physical companion to [`data-and-scoring.md`](./data-and-scoring.md) (which explains the
scoring math). This answers: **which stores exist, what is in each, where it sits in
production, how long it lives, and which part of the recommendation path reads it.**

```
   memorial app (RudderStack)                    Omeka / content pipeline
            │ events                                      │ items + embeddings
            ▼                                             ▼
   POST /api/ingest ──────────────┐              ┌── Qdrant (content store)
            │                     │              │   vectors + payloads
            ▼                     ▼              │
   redis event buffer      parquet event log     │
   (30-day window)         (append-only truth)   │
            │                                    │
            ▼                                    │
   UserSignals (redis user-model store) ─────────┤
   taste/recency vectors, tag affinity/aversion  │
            │                                    │
            ▼                                    ▼
   GET /api/recommend:  candidates (Qdrant) → scorers → fusion/bandit → MMR → distractor
            │
            ├─→ served log (parquet) + impression store (redis, 24h)
            └─→ request_id echoed by the app on the next view → bandit reward join
```

---

## 1. The stores

### 1.1 Qdrant — the content store (read-only for the engine)

| | |
|---|---|
| What | One collection per tenant (e.g. `omeka-items`, `westerbork-ar-ai`): one point per content item — embedding vector (384-dim) + payload (`title`, `text`, `tags[{facet,label,weight}]`, `location{lat,lon}`, `time_metadata`, `image_url`, `public_url`, `creator`) |
| Where (prod) | `http://qdrant.qdrant:6333` in-cluster (`QDRANT_API_URL`); its own Longhorn storage via the qdrant HelmRelease |
| Written by | The content/ingestion pipeline (omeka-tools) — **never by the recsys** |
| Used for | Candidate generation (vector + filtered + random sample), `score_semantic` / `score_recency` (vectors), `score_tag` / `score_aversion` (payload tags), `score_geo` (payload location), and everything the dashboard shows about an item |
| Dev fallback | No `QDRANT_API_URL` → built-in 8-item fixture world (`testing/fixtures.py`) |

### 1.2 Redis — the live visitor state (fast, semi-durable)

`redis://recsys-redis:6379/0` (`REDIS_URL`). Since the AOF change, backed by Longhorn PVC
`recsys-redis-data` (2Gi), `appendonly yes`, `volatile-lru` (only TTL'd keys evictable).
Deployment strategy is Recreate-equivalent (surge 0), so the RWO volume is never double-mounted.

| Keys | Content | TTL | Read by |
|---|---|---|---|
| `evt:<user_id>` | Raw `InteractionEvent`s, 30-day window | none (trimmed by window) | model rebuilds (`updater.refresh`), dashboard counts |
| `umodel:<user_id>` | `UserSignals` JSON — `positives`, `negatives`, `viewed`, `recent_views`, `tag_affinity`, `tag_aversion`, `taste_vector`, `recency_vector`, `behavior`, `demographics` | 7 days (refreshed on every ingest) | **the serve path**: `Recommender` reads exactly this at request time |
| `imp:<request_id>` | Served feature vectors per item | 24 h | online bandit update when the reward event arrives |

Dev fallback: no `REDIS_URL` → in-memory fakes (wiped on restart).

### 1.3 Parquet event log — the durable source of truth

`EVENT_LOG_DIR=/app/logs` on the api PVC (`ai-engine-api-pvc`, Longhorn), per-tenant partition
(`/app/logs/<tenant>/` when multi-tenant).

| Path | Content | Written when |
|---|---|---|
| `date=YYYY-MM-DD/part-<uuid>.parquet` | Every normalized event, append-only, immutable parts | each `/api/ingest` batch |
| `served/date=…/…` | One row per recommendation response: `request_id`, `user_id`, `ts`, `strategy`, ranked items JSON, per-scorer breakdown | each `/api/recommend` |

Used for: **recovery** (`/api/replay-events` rebuilds all redis state — the Config page
"rebuild models from event log" button), **export** (`/api/export/events` tar.gz),
**explainability** (`/api/served/explain` joins a request_id back to what/why),
**offline bandit training / eval**. Not read on the serve path.
Dashboard caveat: the served scan for the visitor table is cached 5 min (`_dash_cached`).

### 1.4 Small files on the api PVC (`/app/logs`)

| File | Content | Writer → Reader |
|---|---|---|
| `models/clusters.json` | Explainable visitor clusters (FCM, k=4) with taxonomy profiles + Falk hints | nightly `cluster-train` CronJob → Cohort segments + `/api/clusters` (re-read per request) |
| `registry/tenants.json` | Runtime tenant registry, API keys **hashed** | Access Tokens page → tenancy resolution |
| bandit state (per tenant, `bandit_state_path`) | Learned LinUCB theta/covariance | online updates → ranking when `ranking_mode=bandit` |

Baseline (read-only, from ConfigMap): `TENANTS_PATH=/app/config/tenants.json`.

### 1.5 Postgres (optional) — survey demographics

`DB_NAME=ai_engine_dev`: `PostgresDemographicsProvider` looks up survey demographics by
visitor id at model-refresh time. Without it, demographics come only from `IDENTIFY` /
survey events. Stored into `UserSignals.demographics` (inspection) and folded into
`tag_affinity` via `survey.py` (country/age/theme → taxonomy tags).

---

## 2. What the recommender actually reads at serve time

One request = `GET /api/recommend?user_id=…`:

1. **`umodel:<uid>` from redis** — the whole visitor model. Cold visitor (no positives):
   survey-seeded `tag_affinity` + diverse global sample instead of personal vectors.
2. **Qdrant** — candidate pools (semantic neighbours of `taste_vector`, tag-filtered,
   random exploration sample; `pool_per_generator` each) with payloads.
3. Scorers combine model × candidate: `semantic`, `tag`, `recency`, `aversion`, `geo`
   (weights = Config → Fusion, or the learned bandit theta when `ranking_mode=bandit`).
4. MMR diversity re-rank (`mmr_lambda`), optional distractor slot.
5. Response logged to the served parquet + `imp:<request_id>` for the reward join.

**Not read at serve time**: parquet logs, clusters.json, Postgres, the event buffer.
That's why dashboard slowness or a cron failure can never affect recommendation latency.

---

## 3. Durability & recovery cheat-sheet

| Store | Survives pod restart? | Survives loss? |
|---|---|---|
| Qdrant | yes (own storage) | re-ingest content pipeline |
| Redis | yes (AOF on PVC since v0.6.21-era fix) | **replay from parquet log** (Config → rebuild) |
| Parquet log | yes (PVC) | this IS the source of truth — export it off-site (`/api/export/events`) |
| clusters.json | yes (PVC) | next nightly cron (or a one-shot Job) retrains |
| Bandit state | yes (PVC) | decays back to prior = fusion weights; retrains online |
| Tenant registry | yes (PVC) | recreate tenants; deploy-baseline tenants come from Git |

Historical note: before the AOF fix redis ran on an emptyDir — every restart wiped models
and buffers (the "all visitors show 0 views" symptom). Replay restores them.
