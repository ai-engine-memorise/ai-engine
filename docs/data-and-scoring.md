# Data & Scoring Reference

What the recommender collects about **visitors** and **content**, and exactly how it turns
that into a ranked recommendation list. This is a ground-truth reference for the *current*
implementation under `src/ai_engine/recsys/` — file:line citations point at the source. For the
design rationale and the hexagonal layering, see [`recsys-architecture.md`](./recsys-architecture.md);
for how the served explanation is built, see [`explainability.md`](./explainability.md).

> Domain note: the app is an in-memorial WW2 museum app; "visitors" (museum users) are the
> subjects. Content is mainly short **stories**, plus images, exhibitions, and POIs.

---

## 1. Data collected about visitors

### 1.1 Raw interaction events

Everything observed about a visitor arrives as events. The canonical model is `InteractionEvent`
(`contracts/models.py:44-58`); raw RudderStack/webhook payloads are normalized into it by
`adapters/rudderstack.py` (`normalize_event`, line 51).

| Field | Type | Meaning |
|-------|------|---------|
| `user_id` | str | RudderStack `userId` or `anonymousId` |
| `event` | str | e.g. `CONTENT_VIEW_ENDED`, `CONTENT_LOOKUP`, `SURVEY_SUBMITTED`, `IDENTIFY` |
| `ts` | datetime | UTC timestamp |
| `session_id` | str? | Browser session (sequence grouping) |
| `request_id` | str? | The rec-response id, echoed back on later views — joins impressions → outcome for bandit training |
| `content_id` | str? | Viewed item (normalized `content_1234` → `1234`) |
| `dwell_seconds` | float? | Time on content (explicit, or computed from start/end pair) |
| `end_reason` | enum? | How the view ended (see below) |
| `query_text` | str? | Search/lookup text |
| `clicked_id` | str? | Item clicked from a rec list |
| `impressions` | list[str] | Items shown but not engaged → **soft negatives** |
| `survey_answers` | dict | `question_id → answer` (str / list / float) |
| `raw` | dict | Untransformed source payload |

**`EndReason`** (`contracts/enums.py:14-19`) maps to a completion value used in scoring:
`next_button → 1.0`, `link → 0.6`, `close_button → 0.0`, `abandon → -0.5`.

Events persist append-only as Parquet (`adapters/event_log.py`): ingested events under
`date=YYYY-MM-DD/`, and served recommendations under `served/date=YYYY-MM-DD/` (records the
`request_id` and per-item feature vectors so `(context, action, reward)` tuples can be reconstructed).

### 1.2 Engagement strength (per viewed item)

A raw view becomes a **continuous** strength in `[-1, 1]` (not binary) —
`signals/engagement.py:53-71`:

```
strength = w_dwell*dwell_ratio + w_completion*completion + w_revisit*revisit + w_survey*survey
```

Weights (`contracts/config.py`, `EngagementWeights`): `dwell 0.4, completion 0.3, revisit 0.2, survey 0.1`.

| Component | Formula | Range |
|-----------|---------|-------|
| `dwell_ratio` | `dwell / est_reading_time`, capped at `dwell_cap_ratio` (2.0), then normalized | [0,1] |
| `est_reading_time` | `word_count / reading_speed_wps (+ img_extra_time if image)` — computed, not stored | seconds |
| `completion` | `_COMPLETION[end_reason]` | {-0.5, 0, 0.6, 1.0} |
| `revisit` | `1 - exp(-visits / 2)` | [0,1) |
| `survey` | `(rating - 3) / 2` (1–5 Likert) | [-1,1] |

Outcome classification (`engagement.py:74-79`): `strength ≥ positive_threshold (0.30)` → **positive**;
`≤ negative_threshold (-0.05)` → **negative**; else **neutral**.

### 1.3 The user model: `UserSignals`

Built by `signals/signal_builder.py` (`build_user_signals`, line 91); model at `contracts/models.py:68-81`.

| Field | Derivation |
|-------|-----------|
| `positives` | `{content_id → max(strength,0) * decay}` for positive outcomes |
| `negatives` | `{content_id → abs(strength) * decay}` for negative outcomes **and** soft negatives |
| `viewed` | all content_ids seen (dedup) |
| `recent_views` | content_ids, most-recent-first (sequence awareness) |
| `tag_affinity` | `{facet:label → [0,1]}` from positively-engaged tags + survey + demographics, max-normalized |
| `tag_aversion` | `{facet:label → [0,1]}` from negatively-engaged tags |
| `taste_vector` | L2-normalized weighted centroid of liked items' embeddings (null if no positives) |
| `recency_vector` | embedding of the most-recent viewed item |
| `behavior` | summary stats: `n_views, n_positive, n_negative, avg_dwell_ratio, completion_rate, revisit_rate, depth` |
| `demographics` | `age, gender, nationality, personal_connection, province` |

**Recency decay** (`signal_builder.py:77-81`): `w = strength * 0.5 ** (age_days / half_life_days)`,
`half_life_days = 14`. Applied to both positives and negatives.

**Soft negatives** (`signal_builder.py:148-153`): an item in an `impressions` set but never engaged
gets `negatives[id] = soft_negative_weight (0.30) * decay`.

**Tag affinity assembly** — three sources accumulate into `tag_affinity`, then max-normalized to [0,1]:
1. **Engagement** (`signal_builder.py:141`): each tag on a liked item gets `+= positives[cid] * tag.weight`.
2. **Survey** (`survey.py:119-150`): personalization answers map *directly* to taxonomy tags at weight 1.0
   (e.g. `q:personalization_theme → theme_what:{label}`, `…interest → theme_how.type_of_stores:…`,
   `…area → place_where.camp_areas:…`).
3. **Demographics** (`signal_builder.py:239-258`): age → `person_who.age_group` (0.5), gender →
   `person_who.gender_and_age` (0.3), nationality → `person_who.city_village_country:From: …` (0.4),
   province → `place_where.province_netherlands` (0.5).

### 1.4 Surveys & demographics

- **Pre-survey (demographic):** age, gender, nationality, province, personal_connection → stored in
  `demographics` and seeded into `tag_affinity`.
- **Personalization survey:** theme / interest / area; values must match taxonomy labels and feed
  `tag_affinity` at the strong weight 1.0.
- Carried on `SURVEY_SUBMITTED`, `SURVEY_ANSWERED`, and RudderStack `IDENTIFY` traits.
- Providers in `adapters/demographics.py`: `Null` (default), `Static` (tests), `Postgres` (visitor table).

### 1.5 Cold vs warm

`is_cold = not positives` (`contracts/models.py:83-84`) — true until the visitor has ≥1 positively-engaged
item. **Warm** path generates candidates from the taste vector + top tags; **cold** path falls back to a
diverse global `sample()`. (`cold_start_min_positives` exists in config but the live check is the
empty-positives test above.)

---

## 2. Data stored about content (items)

### 2.1 Content model

`contracts/models.py:27-37`:

| Field | Type | Notes |
|-------|------|-------|
| `id` | str | |
| `content_type` | enum | `text_item, image_item, video_item, audio_item, exhibition, tag, poi` (`enums.py:4-11`) |
| `title` | str | |
| `text` | str | body |
| `tags` | list[Tag] | structured expert tags |
| `word_count` | int | drives reading-time estimate |
| `has_image` | bool | adds `img_extra_time` to estimate |
| `lat`, `lon` | float? | item location |

There is **no** stored `est_reading_time`; it is derived from `word_count`/`has_image` at scoring time.

### 2.2 Tags & taxonomy

A `Tag` is `{facet, label, weight=1.0}` with canonical key `facet:label` (`models.py:15-24`). Facets
are hierarchical (`memorise_taxonomy/taxonomy.py:115-164`):

- `theme_what` — historical themes (with parent rollups)
- `theme_how` — `ai_engine_themes`, `type_of_stores`
- `place_where` — `camp_areas`, `transit_destinations`, `province_netherlands`
- `person_who` — `reason_for_imprisonment`, `age_group`, `gender_and_age`, `mothertongue`, `city_village_country`
- `time_when` — `time_period`; `medium_what`; `language_how` — `tone`

Labels are normalized (`normalize_label`, casefold/strip-accents/de-alias) so content tags and user-model
keys align despite casing/typos. Flat Omeka strings are classified into facets via `assign_facet`.

### 2.3 Embeddings & Qdrant index

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`, **384-dim** (`adapters/fastembed_model.py:9-19`).
- **Embedded text:** title + body (`text_all`), historically suffixed with creator.
- **Qdrant payload** carries `title, text, content_type, word_count/text_length_words, tags`, plus:
  - `tag_labels` — flat `facet:label` strings, **KEYWORD** index → tag recall via `MatchAny`.
  - `locations` — `{lat,lon}` list, **GEO** index → radius search (fallback to flat `lat`/`lon`).
  - `time_metadata.dates_of_creation` — datetime index.
- **Recall ops** (`adapters/qdrant_store.py`): `search_vector` (cosine), `search_tags` (MatchAny over
  tag_labels), `search_filter` (exact tag value), `search_geo` (GeoRadius), `sample` (random).

---

## 3. How recommendations are scored

Orchestrated in `recommender.py` (`recommend` → `recommend_for_signals`, lines 51-217).

### 3.1 Candidate generation

`pool_per_generator = 30` per source.

- **Unconstrained (warm):** union of **semantic** (vector search on `taste_vector`) + **tag** (MatchAny over
  top ~20 affinity tags), minus already-seen (`seen = positives ∪ negatives ∪ viewed`).
- **Constrained:** a `filter=<tag value>` and/or `near_lat/near_lon (+radius)` request restricts via
  `search_filter` / `search_geo`; given both, they intersect (AND).
- **Fallbacks:** cold-start → global `sample()`; filter exhausted + `filter_reshow_when_exhausted` →
  re-show seen content *within the filter only*.

### 3.2 Scorers — all return `[0, 1]`

`ranking/scorers.py`. `cosine ∈ [-1,1]` is remapped via `(cosine + 1) / 2`.

| Scorer | Formula | Signal |
|--------|---------|--------|
| `semantic` | `(cos(taste_vector, cand) + 1)/2` | whole-history taste centroid (blurred) |
| `affinity` | `max_i w_i * (cos(cand, liked_i) + 1)/2` | item-kNN: sharp max-sim to any one like |
| `tag` | `Σ affinity[t]*cand_weight[t] / Σ affinity[t]` | tag overlap |
| `recency` | `(cos(recency_vector, cand) + 1)/2` | closeness to most-recent view |
| `aversion` | `Σ aversion[t]*cand_weight[t] / Σ aversion[t]` | overlap with disliked themes (penalty via weight) |
| `geo` | `exp(-distance_m / geo_scale_m)` | proximity to request location (`geo_scale_m = 300`) |

Liked-item vectors for `affinity` are fetched at serve time (one `get_vectors`), **not** stored in the
user model, so the model doesn't grow with #likes.

### 3.3 Fusion + MMR

**Weighted fusion** (`fusion.py:15-26`): `fused = Σ weight[s] * score[s]`, per-scorer products kept in
`breakdown` for explainability. Default `FusionWeights` (`config.py:12-20`):

```
semantic 0.30, affinity 0.25, tag 0.25, recency 0.10, aversion -0.25, geo 0.20
```

`aversion`'s negative weight makes it a penalty (fused score may go negative — intentional; only
per-scorer outputs are bounded).

**MMR rerank** (`fusion.py:29-58`, `mmr_lambda = 0.7`): greedy
`argmax_i [ λ*fused_i - (1-λ)*max_{j∈selected} cos(vec_i, vec_j) ]` — trades relevance for diversity.

### 3.4 Distractor (novelty injection)

`recommender.py:176-215`. One deliberately off-profile item, `kind="distractor"`, inserted at a random
slot from `[3, 4]`. `distractor_enabled=True`, `distractor_probability=1.0`,
`distractor_strategy="max_dissimilar"` (negate taste vector and find nearest; or `unexplored_theme` /
`random`). Within a constrained query it picks the lowest-relevance item *from the same filter set* —
never leaks outside. Reserves one slot, so `rel_limit = final_limit - 1`.

### 3.5 Learned ranking — contextual bandit (optional)

`ranking/bandit.py`, LinUCB-style, pure-Python. Off by default (`ranking_mode="static"`).

- **Feature vector** (d=6, `FEATURE_ORDER`): `(semantic, affinity, tag, recency, aversion, geo)` — the
  same per-scorer values. Logged on **every** request, even in static mode, so a bandit can be trained
  from static-served traffic.
- **Model:** `E[reward|x] = θ·x`, with `θ = A⁻¹b`. **Prior** `θ0 = FusionWeights` via ridge
  (`bandit_ridge=1.0`) — so the bandit at its prior is byte-for-byte the static ranking, then learns away.
- **Exploration:** optional UCB bonus `α * sqrt(xᵀA⁻¹x)` (`bandit_alpha=0.3`, `bandit_explore`).
- **Training:** offline batch from the joined event logs (loaded at startup via `BANDIT_STATE_PATH`), or
  online incremental updates (`bandit_online=False` by default).

### 3.6 End-to-end order

```
load UserSignals → generate candidates (seen-filtered) → fetch content + vectors
  → score each (per-scorer [0,1], log feature vector) → fuse (or bandit override)
  → MMR rerank to rel_limit → inject distractor → Recommendation(items, strategy, diagnostics)
```

`strategy ∈ {warm, cold, cold_start_fallback, seen_fallback}`; `diagnostics` carry pool size, generators
used, filter, ranking mode, and distractor info.

---

## 4. Key config defaults (`contracts/config.py`)

| Param | Default | Param | Default |
|-------|---------|-------|---------|
| `engagement.dwell/completion/revisit/survey` | 0.4/0.3/0.2/0.1 | `half_life_days` | 14.0 |
| `fusion.semantic/affinity/tag/recency/aversion/geo` | 0.30/0.25/0.25/0.10/-0.25/0.20 | `soft_negative_weight` | 0.30 |
| `reading_speed_wps` | 4.2 | `pool_per_generator` | 30 |
| `img_extra_time` | 1.3 | `final_limit` | 10 |
| `dwell_cap_ratio` | 2.0 | `mmr_lambda` | 0.7 |
| `positive_threshold` | 0.30 | `geo_scale_m` / `geo_radius_m` | 300 / 1000 |
| `negative_threshold` | -0.05 | `cold_start_min_positives` | 1 |
| `distractor_enabled/probability` | true/1.0 | `distractor_slots` | [3,4] |
| `ranking_mode` | "static" | `bandit_alpha/ridge` | 0.3/1.0 |

Env overrides (`composition.py:86-128`): `RECSYS_W_*`, `RECSYS_MMR_LAMBDA`, `RECSYS_FINAL_LIMIT`,
`RECSYS_DISTRACTOR_*`, `RECSYS_RANKING_MODE`, `RECSYS_BANDIT_*`, `BANDIT_STATE_PATH`, plus infra
(`REDIS_URL`, `QDRANT_API_URL`).
