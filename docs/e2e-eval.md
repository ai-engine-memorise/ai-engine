# AI-Engine — end-to-end evaluation plan

A hands-on checklist to validate the whole system with a colleague. Each point is
**Action → Expect → Result (pass / fail / note)**. Most can be done from the **Inspector**
(`/inspector`); a few use `curl`. Budget ~45 min.

Base URL below assumes local: `http://localhost:8010`. Replace with the deployed URL if testing remote.

---

## 0. Setup

| # | Action | Expect |
|---|--------|--------|
| 0.1 | Start infra: Qdrant (`:6333`), Redis (`:6379`). Confirm the content collection exists and has items. | `curl localhost:6333/collections` lists the collection (e.g. `ui-test`) |
| 0.2 | Start the engine with infra env (`QDRANT_API_URL`, `REDIS_URL`, `COLLECTION_NAME`, `EVENT_LOG_DIR`; for learning add `RECSYS_RANKING_MODE=bandit RECSYS_BANDIT_ONLINE=true BANDIT_STATE_PATH=...`). | `curl /health` → `{"status":"ok"}` |
| 0.3 | Open `/inspector`. Set the **base URL** and (if `INGEST_API_KEY` is set) the **X-API-Key** in the header. | Header dot shows **● connected** (green) |

> Tip: pick one **test user id** (e.g. `eval_alice`) and use it throughout so the per-user tabs all line up.

---

## A. Serving & ranking (the core promise)

| # | Action | Expect |
|---|--------|--------|
| A.1 | **Cold start** — Inspector → Request tab with a brand-new `user_id`, click Recommend. | Returns a full list (never empty); `strategy: cold`; `diagnostics.cold_start_fallback: true` |
| A.2 | **Breakdown** — look at the breakdown bars. | Each item shows a stacked bar (semantic/affinity/tag/recency/aversion/geo); colours match the legend |
| A.3 | **Distractor present** — find the row flagged `distr`. | Exactly one distractor, at **rank 3 or 4**, `role: distractor`; `diagnostics.distractor.slot` matches its rank |
| A.4 | **Ranking is sensible** — eyeball the top items vs the user's interests (after warming in section B). | Top items thematically match; the distractor is visibly off-theme |
| A.5 | **Exclude seen** — recommend again for a user who has viewed items (section B). | Already-viewed content does **not** reappear |
| A.6 | **Compact mode** — `curl "/api/recommend?user_id=eval_alice&include_content=false"`. | Items have `id/rank/relevance_score/role` but **no** `content` blob |
| A.7 | **Location filter** — `curl "/api/recommend?user_id=eval_alice&filter=<AiARLocation...>"`. | Only content tagged with that location; `diagnostics.generators` includes `filter` |
| A.8 | **Filter never leaks** — confirm every returned id is in that location. | No out-of-location items; if the location has ≤ 9 items, all shown and `diagnostics.distractor.placed:false` |
| A.9 | **Geo (only if content has coords)** — add `&near_lat=..&near_lon=..&geo_radius_m=500`. | Candidates restricted to the radius; nearer items rank higher (geo in breakdown). If no coords: empty/again no crash |

---

## B. User model & personalization

| # | Action | Expect |
|---|--------|--------|
| B.1 | **Ingest a deep read** — POST `/api/ingest` a `CONTENT_VIEW_STARTED` + `CONTENT_VIEW_ENDED` (reason `next_button`, dwell ~120s) for 2-3 items of one theme, userId `eval_alice`. | Response `{ingested:N, users:["eval_alice"]}` |
| B.2 | **User model built** — Inspector → User model tab. | Persona card appears; `warm`; interests list the theme just read with evidence counts |
| B.3 | **History** — Inspector → History tab. | Event timeline + engaged-content table with the right dwell / end-reason / outcome (`positive`) |
| B.4 | **Negative signal** — ingest an `abandon` (dwell ~2s) on a different theme; reopen User model. | That theme appears under **Aversions**; recs down-weight it |
| B.5 | **Recommend reacts** — Request tab again. | Unseen items of the liked theme surface near the top |
| B.6 | **Survey persona** — emit the survey (`SURVEY_*`) or `identify` traits for a user; check User model. | Demographics + `person_who` / theme affinity seeded even with no views (cold-start bridge) |

---

## C. Explainability & cohort

| # | Action | Expect |
|---|--------|--------|
| C.1 | **Persona** — User model tab for `eval_alice`. | A Falk type (Explorer/Hobbyist/Experience-Seeker/Recharger/Facilitator) with rationale + confidence; Pekarik preference; engagement style; a readable prose summary |
| C.2 | **Persona is grounded** — read the prose + interests. | Claims match the actual reading behaviour (no invented themes); persona colour is consistent across tabs |
| C.3 | **Reference** — Personas tab. | All visitor types + experience prefs + engagement styles + a Methods section explaining the signals |
| C.4 | **Clusters** — Cohort tab (needs a trained cluster model: `python explain/cluster_train.py --method fcm`). | Cluster cards with size bar, top tags, Falk hint; expandable **member list** |
| C.5 | **Drill into a user** — click a member chip in a cluster. | Loads that user into the User model tab |
| C.6 | **Cohort content** — Content tab. | Content seen across visitors (views/liked/disliked + like-rate), popular themes, and **content preferences per cluster** |

---

## D. Learning (the bandit)

| # | Action | Expect |
|---|--------|--------|
| D.1 | **Mode** — Policy tab. | Shows `mode: bandit` (or `static`); prior vs learned weights, zero-centred bars |
| D.2 | **Prior == static** — with a fresh state, learned == prior. | No shift yet; `trained: prior only` |
| D.3 | **Online update** — serve a recommend (note the `request_id`), then ingest a `CONTENT_VIEW_ENDED` echoing that **`request_id`** for a served item. | Ingest response `bandit_updates ≥ 1`; Policy tab now `trained`, weights shifted |
| D.4 | **Idempotency** — re-POST the exact same reward event. | Second time `bandit_updates: 0` (no double-count) |
| D.5 | **Training health** — Policy tab banner. | Verdict (`cold` / `learning` / `converged`), update count, per-weight confidence bars, and which weights still lack data |
| D.6 | **Need-more-data signal** — with few updates. | Verdict `cold` / "needs more data"; some weights flagged "no data (still at prior)" |

---

## E. Durability & ops

| # | Action | Expect |
|---|--------|--------|
| E.1 | **Served log** — Traffic tab (or `curl /api/served/recent`). | Recent impressions with user / strategy / ranking / item count |
| E.2 | **Event log on disk** — check `EVENT_LOG_DIR`. | `date=YYYY-MM-DD/*.parquet` (events) and `served/date=*/*.parquet` (impressions) accumulate |
| E.3 | **Metrics** — Traffic tab. | Counters: ingests, recommends, cold-rate, avg pool, distractor-rate |
| E.4 | **Survives restart** — restart the engine, recommend again for `eval_alice`. | User model still present (Redis); bandit state reloads from `BANDIT_STATE_PATH` |
| E.5 | **Settings** — `/settings` page; change a tunable (e.g. `distractor_probability` to 0.5), save. | Takes effect without redeploy; recommend now injects a distractor ~half the time |

---

## F. Security & guards

| # | Action | Expect |
|---|--------|--------|
| F.1 | **Ingest guard** — with `INGEST_API_KEY` set, POST `/api/ingest` **without** the key. | `401` |
| F.2 | **PII guard** — GET `/api/usermodel` / `/usermodel/explain` / `/usermodel/history` / `/content/stats` without the key. | `401` (these expose demographics / individual data) |
| F.3 | **Serving open** — GET `/api/recommend` without the key. | `200` (the app's serving path stays open) |

---

## G. Robustness (try to break it)

| # | Action | Expect |
|---|--------|--------|
| G.1 | Recommend for a totally unknown user. | Non-empty cold-start list, no error |
| G.2 | Filter by a tag that matches nothing. | Empty `items`, `diagnostics.reason: empty_filter`, **no 500** |
| G.3 | Ingest a malformed / partial event (missing fields). | Skipped gracefully, other events still ingested |
| G.4 | Geo query when content has no coordinates. | Empty or proximity-0, **no crash** (degrades) |
| G.5 | Recommend for a location with very few items. | All items shown, distractor `placed:false` with a reason (not silent) |
| G.6 | **Exhausted location** — view ALL of a small location's stories, then recommend with that `filter`. | Location **re-shows** its (now-seen) stories rather than returning empty; `diagnostics.seen_fallback: true`. (Disable via `filter_reshow_when_exhausted=false` → empty.) |

---

## Scoring

For each point mark **PASS / FAIL / NOTE**. A point is a **blocker** if it breaks the core loop
(A.1, A.3, B.1, B.2, D.3) — those must pass. The rest are quality/coverage.

Capture for any FAIL: the request, the response (or screenshot), and `diagnostics`.

| Section | Pass | Fail | Notes |
|---|---|---|---|
| A. Serving & ranking | / 9 | | |
| B. User model | / 6 | | |
| C. Explainability & cohort | / 6 | | |
| D. Learning | / 6 | | |
| E. Durability & ops | / 5 | | |
| F. Security | / 3 | | |
| G. Robustness | / 5 | | |

> The Inspector covers most of A-E visually; keep it open on a second screen while the
> colleague drives the `curl` / app actions.
