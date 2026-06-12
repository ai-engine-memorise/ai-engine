# AI-Engine Inspector — visibility panel

A single static page that visualizes the engine's inner workings. The engine is glass-box,
so the panel is mostly a **viewer over existing `/api/*`** plus three thin read endpoints.

Open it at **`http://<host>:<port>/inspector`** (served by the recsys app itself).

Styled to the MEMORISE docs aesthetic (archival greige + bronze, Montserrat headings, light/dark).

## Two scopes

The tabs split into **Across users** (cohort / global state) and **Within user** (single-visitor deep
dive, driven by the `user_id` field) — full visibility at both levels.

**Across users**
| Tab | Source | Shows |
|---|---|---|
| **Cohort** | `GET /api/clusters` | cluster cards: size bar (share of visitors), Falk hint, breadth, top tags (k-means or fcm) |
| **Policy** | `GET /api/policy` | bandit **prior θ₀ vs learned θ** per feature + the learned shift; mode, α, ridge |
| **Traffic** | `GET /api/metrics`, `GET /api/served/recent` | counters (ingests, recommends, cold-rate, avg pool, distractor-rate) + recent served-impression tail |

**Within user**
| Tab | Source | Shows |
|---|---|---|
| **Request** | `GET /api/recommend` | each rec as a **stacked breakdown bar** (semantic/affinity/tag/recency/aversion/geo), distractor flag, strategy, ranking mode, generators, pool size |
| **User model** | `GET /api/usermodel/explain` | Falk type + Pekarik pref + engagement style, prose summary, interest/aversion bars (with evidence counts), trajectory, soft cluster membership |

Header: base URL, `X-API-Key` (PII-guarded tabs), `user_id`, theme toggle. Settings persist in localStorage.
The `/inspector` route reads the file fresh per request, so HTML edits need only a browser refresh.

## New endpoints (all thin reads)

- `GET /api/policy` → `{mode, feature_order, prior, theta, trained, alpha, ridge, explore}`
- `GET /api/metrics` → in-process serving counters (+ derived rates). *Not Prometheus — that's the prod upgrade.*
- `GET /api/served/recent?n=50` → tail of the durable served log (needs `EVENT_LOG_DIR`; PII-guarded).

## Notes

- The panel needs nothing new to *decide* — `diagnostics` / `breakdown` / `features` / `explain` /
  `clusters` were already emitted. It's the viewer, not new instrumentation.
- `/metrics` counters are in-process (reset on restart). For production observability, add a Prometheus
  `/metrics` exporter + Grafana instead; the in-process counters are for local visibility.
- The HTML is vanilla JS, no build step (same pattern as the survey-viewer). File:
  `src/ai_engine/recsys/static/inspector.html`.
