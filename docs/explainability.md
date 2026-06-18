# Explainability — persona, evidence, and visitor clusters

The recommender is a **glass box**: the user model is structured (tag affinity over the expert
taxonomy, aversions, sequence, engagement summary), and every recommendation carries a per-scorer
`breakdown`. This doc covers the layer on top of that: turning the model into an **interpretable
persona** and **explainable visitor segments**, grounded in museum-visitor theory.

Module: `src/ai_engine/recsys/explain/` (pure, no IO). Endpoints: `/api/usermodel/explain`, `/api/clusters`.

## Theory grounding

| Framework | Used for | Signal it maps to |
|---|---|---|
| **Falk (2009), *Identity and the Museum Visitor Experience*** | `visitor_type` | breadth of interests (distinct themes) × engagement depth (dwell/completion) × pace |
| **Pekarik, Doering & Karns (1999), "Satisfying Experiences in Museums"** | `experience_preference` | which taxonomy facet family dominates affinity (medium → object, theme → cognitive, personal-story/person → introspective) |
| **Tintarev & Masthoff** (explanation aims) | the whole design | **scrutability** — every claim carries its evidence so a visitor can inspect/correct it |
| Csikszentmihalyi (flow) | `engagement_style` | dwell ratio vs reading-time estimate (already computed as engagement strength) |

### Falk visitor types (heuristic, transparent)

```
Hobbyist          narrow + deep + repeat        (1-breadth)·depth·cognitive·views
Explorer          broad + engaged               breadth·depth
Experience-Seeker broad + light/skims           breadth·(1-min(dwell,completion))
Recharger         few, slow, introspective      depth·(1-breadth)·few_views·introspective
Facilitator       social/accompanied visit      from personal_connection demographic
```

`breadth` = number of **distinct content themes** engaged (not affinity sub-labels, which make
one theme look broad). Pick = argmax; `confidence` = margin to the runner-up; `rationale` names the
drivers. Heuristics are intentionally simple and auditable — calibrate against real visitor data later.

## What the persona contains (`PersonaExplanation`)

- `interests` / `aversions` — top taxonomy tags **with `evidence`** (the content ids that drove each).
- `engagement_style` — deep_reader | completionist | skimmer | sampler | contemplative (from `behavior`).
- `experience_preference` — Pekarik object | cognitive | introspective | social.
- `visitor_type` — Falk type + confidence + rationale + per-type scores.
- `trajectory` — recent thematic arc (dominant theme per recent view, most-recent first).
- `summary` — optional prose (`explain/verbalize.py`: deterministic template, or `verbalize_llm`
  which words the SAME structured facts so nothing is invented).

A browser who never "liked" anything is **not** treated as cold — their skim behavior + thematic
trajectory still produce a persona (Experience-Seeker / skimmer).

## Explainable clusters (`explain/clusters.py`)

K-means OR fuzzy-c-means over the **tag-affinity vectors**. The cluster *is* the explanation: each
centroid is a tag-weight profile in the taxonomy, so a segment reads as "Forced Labor + Resistance,
narrow (Hobbyist-like)" rather than an opaque embedding. Pure-Python (no sklearn).

- **k-means** (`cluster_users` / `assign`) — hard buckets. Simplest.
- **fuzzy-c-means** (`cluster_users_fuzzy` / `assign_fuzzy`) — **soft membership** (rows sum to 1).
  Fits Falk's *overlapping* identities: a visitor deep in two themes reads as `c0:0.45 c1:0.55`
  instead of being forced into one bucket. `m` = fuzziness (2.0 default; →1 approaches k-means).

Compare them on archetypes (incl. a deliberately blended visitor): `python explain/compare_clustering.py`.
A single-theme visitor is crisp (~0.98) under both; only fuzzy reveals the blended one is split.
Pick the method at train time (`cluster_train.py --method fcm`); the API assigns with the matching
function automatically (reads `method` from the saved model).

Offline-trained from the live user models:

```bash
REDIS_URL=redis://localhost:6379 python explain/cluster_train.py --k 4 --out ./data/clusters.json
CLUSTER_MODEL_PATH=./data/clusters.json uvicorn ai_engine.recsys.api:app   # GET /api/clusters
```

When `CLUSTER_MODEL_PATH` is set, `/api/usermodel/explain` also returns the visitor's `cluster`.

## Endpoints

- `GET /api/usermodel/explain?user_id=&verbalize=true` — persona (+ cluster if model loaded). Guarded
  by `INGEST_API_KEY` (exposes demographics).
- `GET /api/clusters` — the segment profiles. Guarded.

## Scope / next

- Heuristic Falk/Pekarik mappings — replace thresholds with data-calibrated boundaries once labelled
  visitor data exists. The structure (signal → type) stays.
- Cluster-aware recommendation (e.g. a per-segment bandit θ) is the bridge to the contextual bandit's
  "per-segment policy" option — see `bandit/README.md`.
- Memorial-specific empathy/identification axis (dark-heritage / prosthetic-memory literature) can be
  added as a third lens beside Falk + Pekarik.
