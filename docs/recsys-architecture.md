# Recommendation Engine Redesign

Status: **design** (no implementation yet). Tracks the agreed architecture for hardening the
recommender and adding behavioral + tag-based recommendation for the in-memorial app.

## Goal

In-memorial app collects user behavior via **RudderStack + PostHog** (event taxonomy already
defined in `experiments/event-schema/`). Fuse:

- content **embeddings** (semantic),
- domain-**expert tags** on content (structured; content is mainly stories),
- **behavioral signals** (dwell, completion reason, revisits, search clicks, impressions),
- **survey + demographics** (cold start),

into explainable, diversity-aware recommendations.

## Principle: ports & adapters (hexagonal)

Pure typed core knows only Protocols. All IO (Qdrant, Postgres, RudderStack, PostHog, embedding
model) lives behind adapters. Every boundary is typed → every stage testable with fakes, no network.

```
ai_engine/
  contracts/        # pure typed core. NO io imports.
    models.py       # Item/Content, Tag, InteractionEvent, EngagementScore, UserSignals,
                    #   Candidate, ScoredCandidate, Recommendation
    ports.py        # Protocols: EventSource, ContentStore, EmbeddingModel  (ONLY these — see Scope)
    enums.py        # ContentType, EndReason, Outcome, SearchType
    config.py       # RecConfig (weights, decay, thresholds) — all tunables typed
  signals/
    engagement.py   # PURE fn: raw event -> EngagementScore (continuous, not binary)
    signal_builder.py # PURE fn: list[InteractionEvent] -> UserSignals (recency decay, taste vec, tag affinity)
  candidates/
    semantic.py     # SemanticGenerator (qdrant recommend / taste vector)
    tag.py          # TagGenerator (expert-tag affinity recall)
    geo.py          # GeoGenerator
  ranking/
    scorers.py      # PURE fns: score_semantic/affinity/tag/recency/aversion -> [0,1]
    fusion.py       # PURE fns: weighted_fuse + mmr_rerank
  recommender.py    # orchestrator (constructor-injected EventSource + ContentStore + EmbeddingModel)
  adapters/
    qdrant_store.py       # ContentStore (refactor current CommonSearch/Vector/Geo)
    postgres_events.py    # EventSource (refactor current fetch_events LEAD window)
    rudderstack_events.py # EventSource (new)
    posthog_events.py     # EventSource (new)
    fastembed_model.py    # EmbeddingModel
  testing/
    fakes.py        # FakeContentStore, FakeEventSource, InMemoryEmbeddingModel
    fixtures.py     # golden BB events/items/tags
```

## Data flow

```
raw rudderstack/posthog/pg rows
  -> EventSource.fetch_events        : list[InteractionEvent]   (dwell pairing in normalizer)
  -> EngagementScorer.score          : list[EngagementScore]    (continuous strength [-1,1])
  -> SignalBuilder                   : UserSignals              (recency decay, taste_vec, tag_affinity, soft-neg)
  -> CandidateGenerators (union)     : list[Candidate]          (semantic + tag + geo)
  -> Scorers                         : list[ScoredCandidate]    (each scorer returns [0,1], breakdown kept)
  -> WeightedFusion + MMR            : Recommendation           (explainable)
```

## Scope: pragmatic first pass (not full hexagonal yet)

Port **only what has ≥2 real implementations or needs a deterministic test fake**:

- `EventSource` — Protocol. 3 impls coming (Postgres, RudderStack, PostHog). Pays off now.
- `ContentStore` — Protocol. Real Qdrant + `FakeContentStore` for offline tests.
- `EmbeddingModel` — Protocol. Real fastembed + `InMemoryEmbeddingModel` (deterministic) for tests.

Everything else stays **plain pure functions over typed models** — no Protocol wrapper:

- engagement scoring, signal building, the four scorers, fusion + MMR.
- Already testable (pure, no IO). A Protocol there would add indirection with no second impl.

Typed domain models (`Content`, `Tag`, `InteractionEvent`, `UserSignals`, `Candidate`,
`ScoredCandidate`, `Recommendation`) used **throughout** regardless.

Full hexagonal end-state (Scorer/Ranker/Generator as plugin Protocols, registry, etc.) and the
triggers + steps to migrate there: see [`future-hexagonal.md`](./future-hexagonal.md).

## Decisions (locked)

- **Tags in Qdrant payload + payload index.** Flat `tag_labels` (`facet:label`, KEYWORD index) for
  recall via `Filter(should=...)`. Graded tag ranking done in pure-Python `TagScorer`, not Qdrant.
- **Rule-based weighted fusion + MMR.** No learned ranker yet. Hard contract: every `Scorer.score`
  returns `[0,1]` so the weighted sum is valid without rescaling (fused score may go negative via the
  `aversion` penalty — that is intentional, only the per-scorer outputs are bounded).
- **Durable training substrate + learned ranking policy (contextual bandit).** With `EVENT_LOG_DIR`
  set, two append-only Parquet datasets accumulate: `date=*/` (ingested events = reward) and
  `served/date=*/` (recommendations served = action + the per-item FEATURE VECTOR, keyed by
  `request_id` echoed back on later CONTENT_VIEW events). Joining them yields `(context, action,
  reward)` tuples. A linear contextual bandit (`ranking/bandit.py`, LinUCB-style, pure-Python) learns
  `E[reward | x] = θ·x` over the SAME per-scorer features, with the static `FusionWeights` as the
  ridge **prior** (`θ0 = weights`) — so `ranking_mode="bandit"` at the prior is byte-for-byte the
  static ranking, then learns away from it. Trained OFFLINE (`bandit/train.py`) from the logs; the
  state JSON is loaded at startup (`BANDIT_STATE_PATH`). Feature vectors are logged in BOTH modes, so
  a bandit can be fit from traffic served while still static. Online incremental updates: future work.

  ```
  RECSYS_RANKING_MODE=static   # default; logs features for later training
  python bandit/train.py --log ./data/eventlog --out ./data/bandit_state.json
  RECSYS_RANKING_MODE=bandit  BANDIT_STATE_PATH=./data/bandit_state.json   # serve learned θ
  ```
- **Engagement is continuous**, not the current binary `dwell >= estimate`.
- **Dwell pairing moves out of SQL into the EventSource normalizer**, shared across all 3 sources.
  One contract test asserts every adapter emits identical `InteractionEvent` from equivalent raw.
- **Cold/warm routing** on `cold_start_min_events`; cold path uses survey+demographics -> tag_affinity.

## Key formulas

Engagement strength (per content, weights from `RecConfig.engagement`):

```
dwell_ratio = min(dwell / est_reading_time, dwell_cap) / dwell_cap            # [0,1]
completion  = {next_button:1.0, link:0.6, close_button:0.0, abandon:-0.5}     # by end_reason
revisit     = 1 - exp(-visits / 2)
survey      = (rating - 3) / 2                                                # 1..5 -> [-1,1]
strength    = wd*dwell_ratio + wc*completion + wr*revisit + ws*survey
```

Recency decay: `w = strength * 0.5 ** (age_days / half_life_days)`.

Soft negatives: content in `impressions` but never viewed -> negative @ `soft_negative_weight * decay`.

Scorers (all -> [0,1]):

```
semantic   = (cosine(taste_centroid, cand_vec) + 1) / 2        # whole-history centroid (blurred)
affinity   = max_{liked i} (cosine(cand_vec, vec_i)+1)/2 * w_i # item-kNN: max-sim to ANY one like (sharp)
tag        = Σ user_affinity[l]*cand_tag_weight[l]  /  Σ user_affinity[l]
recency    = (cosine(recency_vec, cand_vec) + 1) / 2           # closeness to MOST-RECENT view (sequence)
aversion   = Σ user_aversion[l]*cand_tag_weight[l]  /  Σ user_aversion[l]   # PENALTY (negative weight)
geo        = exp(-distance_m / geo_scale_m)          # proximity to the REQUEST location (per-request)
popularity = log1p(views) / log1p(max_views)        # off by default
```

**Geo and the tag filter are independent.** Geo is never a tag. Two orthogonal mechanisms:
- `filter=<tag>` restricts candidates to a tag value (e.g. `AiARLocationBarrack3`) — discrete place.
- `near_lat/near_lon` (+ optional `geo_radius_m`) — the user's CURRENT GPS location (a per-request
  signal, NOT stored in the user model). Without a radius it only re-ranks by proximity (`geo` scorer,
  weight 0.20); with a radius it also restricts candidates via the Qdrant `locations` GEO index.

Either may be used alone; given together they compose by **intersection (AND)**. `geo_scale_m`
(default 300m) sets the proximity falloff; `geo_radius_m` the optional hard cutoff.

Default fusion weights: `semantic 0.30, affinity 0.25, tag 0.25, recency 0.10, aversion -0.25`.
`affinity` keeps multi-interest users' distinct tastes sharp (centroid alone averages them
into mush). `aversion` (negative weight) downranks themes of abandoned content. `affinity`
needs the liked items' vectors — fetched at serve time (one `get_vectors`), NOT stored in the
user model, so the model doesn't grow with #likes.

Fusion + MMR:

```
fused_i = Σ_s fusion_weight[s] * score_s(i)
select argmax_i [ λ*fused_i - (1-λ)*max_{j in selected} cosine(vec_i, vec_j) ]
```

## Tag payload shape (Qdrant)

```json
{
  "id": "841",
  "content_type": "text_item",
  "tags": [
    {"facet": "theme", "label": "forced_labour", "weight": 1.0},
    {"facet": "prisoner_group", "label": "hungarian_jews", "weight": 0.8}
  ],
  "tag_labels": ["theme:forced_labour", "prisoner_group:hungarian_jews"]
}
```

## EventSource normalization map

| Raw (RudderStack / PostHog)                                   | InteractionEvent          |
|---------------------------------------------------------------|---------------------------|
| `CONTENT_VIEW_STARTED.content.content_id`                     | `content_id`              |
| `STARTED.ts` … `ENDED.ts` (same content+session)              | `dwell_seconds` (paired)  |
| `CONTENT_VIEW_ENDED.details.reason`                           | `end_reason`              |
| `CONTENT_VIEW_STARTED.context.candidates[].content_id`        | `impressions`             |
| `CONTENT_LOOKUP.details.query_text / clicked_id`              | `query_text` + click      |
| `SURVEY_ANSWERED.answers[]`                                   | `survey_answers`          |

## Serving model: online (path B)

User data flows `app → RudderStack → {PostHog (analytics), ai-engine (serving)}`. PostHog stays
the analytics/eval sink; it is NOT queried at request time. Serving uses **path B (online)**:

```
RudderStack ─webhook→ ai-engine ─normalize→ EventBuffer(Redis) ─┐
                                                                 ├─ UserModelUpdater.refresh
                            UserSignals (the user model) ◄───────┘  (build_user_signals, pure)
                                    │ save
                              UserModelStore(Redis)
                                    │ get   (fast read, no rebuild)
                              Recommender.recommend ─→ Recommendation
```

`UserModelStore` is a port: in-memory/recompute fake for tests, Redis for prod — swap without
touching ranking. Rebuild-from-buffer keeps `build_user_signals` the single brain.

## Build status (implemented)

Pragmatic slice is code + passing tests under `src/ai_engine/api/`:

- `contracts/` — models, enums, `RecConfig`, ports (`EventSource`, `ContentStore`, `EmbeddingModel`,
  `UserModelStore`).
- `signals/` — pure `engagement.py`, `signal_builder.py` (the user model).
- `ranking/` — pure `scorers.py` (semantic+tag, [0,1]), `fusion.py` (weighted + MMR).
- `updater.py` (ingest side) + `recommender.py` (serve side).
- `adapters/` — pure `rudderstack.py` normalizer; infra-guarded `redis_store.py`, `qdrant_store.py`,
  `fastembed_model.py`.
- `testing/` fakes + BB fixtures. `tests/` — 31 passing: engagement, scorers, fusion, signal builder,
  end-to-end recommender scenarios + invariants, RudderStack normalization.

Not yet wired: the FastAPI webhook endpoint + real Redis/Qdrant config; geo scorer; learned ranker.

## Open questions

1. **Facet taxonomy** — experts provide closed facet set, or derive seeds from BERTopic `topics.jsonl`?
2. **Survey -> tag map** — reuse existing `survey:kwb:survey` v3 instrument's `question_id/answer_id`?
3. **Geo** — ranking signal (boost nearby) vs hard POI pre-filter?

## Test strategy

- Pure fns (engagement, scorers, fusion): unit + property tests (dwell monotonicity, MMR keeps top-1).
- Protocol fakes: full `Recommender` pipeline runs offline.
- Golden fixtures: canned BB events/items/tags -> snapshot `Recommendation`.
- Shared adapter contract test across all `EventSource` impls.
