# AI-Engine Inspector — visibility panel

A single static page that visualizes the engine's inner workings. The engine is glass-box,
so the panel is mostly a **viewer over existing `/api/*`** plus three thin read endpoints.

Open it at **`http://<host>:<port>/inspector`** (served by the recsys app itself).

## Tabs

| Tab | Source | Shows |
|---|---|---|
| **Request** | `GET /api/recommend` | each rec as a **stacked breakdown bar** (semantic/affinity/tag/recency/aversion/geo), distractor flag, strategy, ranking mode, generators, pool size |
| **User model** | `GET /api/usermodel/explain` | Falk type + Pekarik pref + engagement style, prose summary, interest/aversion bars (with evidence counts), trajectory, soft cluster membership |
| **Segments** | `GET /api/clusters` | cluster cards: top tags, Falk hint, size, breadth (k-means or fcm) |
| **Policy** | `GET /api/policy` | bandit **prior θ0 vs learned θ** per feature + the learned shift; mode, α, ridge |
| **Traffic** | `GET /api/metrics`, `GET /api/served/recent` | counters (ingests, recommends, cold-rate, avg pool, distractor-rate) + recent served-impression tail |

Header has: base URL, `X-API-Key` (for the PII-guarded tabs), and `user_id`. Settings persist in localStorage.

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
