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
    scorers.py      # PURE fns: score_semantic/score_tag/score_geo/score_popularity -> [0,1]
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
- **Rule-based weighted fusion + MMR.** No learned ranker yet — collect impression/click data first.
  Hard contract: every `Scorer.score` returns `[0,1]` so the weighted sum is valid without rescaling.
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
semantic   = (cosine(taste_vec, cand_vec) + 1) / 2
tag        = Σ user_affinity[l]*cand_tag_weight[l]  /  Σ user_affinity[l]
geo        = exp(-distance_m / geo_scale)
popularity = log1p(views) / log1p(max_views)        # off by default
```

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

## Open questions

1. **Facet taxonomy** — experts provide closed facet set, or derive seeds from BERTopic `topics.jsonl`?
2. **Survey -> tag map** — reuse existing `survey:kwb:survey` v3 instrument's `question_id/answer_id`?
3. **Geo** — ranking signal (boost nearby) vs hard POI pre-filter?

## Test strategy

- Pure fns (engagement, scorers, fusion): unit + property tests (dwell monotonicity, MMR keeps top-1).
- Protocol fakes: full `Recommender` pipeline runs offline.
- Golden fixtures: canned BB events/items/tags -> snapshot `Recommendation`.
- Shared adapter contract test across all `EventSource` impls.
