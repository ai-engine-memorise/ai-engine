# Tech debt: payload-schema scatter & duplicated aggregation

Findings from the 2026-07-16 dashboard sprint. The sprint added cohort statistics,
demographic filters, the collection map and several endpoints at high velocity; this
document records the seams it left so they get consolidated deliberately, not
rediscovered as bugs.

**Motivating incident (same day):** the collection map was missing in production.
`/api/content/spread` had its own coordinate extractor (dict-walk only) while the
production payload stores coordinates in the geo-indexed `locations` field as an
object *or list*. The serving adapter already had a correct extractor
(`qdrant_store._extract_latlon`); the endpoint simply didn't use it (fixed in
v0.6.15 by reusing it). That bug is the pattern this document is about.

## D1. Payload-schema knowledge lives in 4+ places — **H** — DONE (v0.6.16)

What a raw Qdrant payload looks like (`locations` object/list, `image_url` /
`imageUrl` / `thumbnail_url`, `public_url`, `time_metadata.dates_of_creation`,
`creator`, `tags`) is independently re-implemented by:

| Place | What it parses |
|---|---|
| `adapters/qdrant_store.py::_extract_latlon` | coordinates (the correct one) |
| `adapters/qdrant_store.py::_payload_to_content` | tags, title, text, lat/lon |
| `api.py::served_explain` | `image_url` / `public_url` fallbacks |
| `api.py::content_spread::_years` | creation-year regex |
| `static/dashboard.html::_latlon` (JS) | coordinates, again, client-side |
| `static/dashboard.html` item modal / eval rows (JS) | image/url/creator/date fallbacks |

Every new payload consumer re-answers "what fields exist" and diverges (see incident).

**Fix:** one server-side normalizer. Extend `Content` (or a sibling
`ContentMeta`) with `image_url`, `public_url`, `years: list[int]`, populate them in
`_payload_to_content` and mirror in `testing/fakes.py`. Endpoints stop touching raw
payloads except the metadata JSON view; `dashboard.html` drops `_latlon` and every
`p.image_url||p.imageUrl||…` chain and consumes normalized fields only.

## D2. Cohort demand aggregation triplicated — **H** (perf-relevant) — DONE (v0.6.16)

`content_stats`, `cohort_stats` and `content_spread` each hand-roll the same loop:
iterate `model_store.iter_signals()`, call `event_buffer.fetch_events(uid)`,
`aggregate_views(...)`, accumulate per-content/per-visitor counters. Three copies,
three chances to drift, and it is the slowest scan the API does (the reason the
dashboard needed lazy loading in the first place).

**Fix:** extract `_cohort_view_stats(c) -> {content_id: ViewAgg}` (plus per-visitor
variant) into one helper. That helper is also the natural seam for the planned
short-TTL cache: cache it once, all three endpoints get fast together.

## D3. Demographic semantics (py) vs labels (JS) — **M** — DONE (v0.6.16)

`survey.canon_demo_value` decides what an answer *means* (language folding, junk,
`no_answer`); `dashboard.html::humanVal` decides how it *reads* (`65_plus` → `65+`,
`under_16` → `<16`). Age-bucket knowledge now exists in both layers.

**Fix (cheap):** have `cohort_stats` return `{value, label}` pairs so the server owns
labels too and `humanVal` shrinks to a fallback. Do together with D1's response
reshaping.

## D4. Dev shims inside prod code paths — **M** — DONE (v0.6.16)

- `api.py::content_spread` carries a `digits()` id fallback that exists only because
  `normalize_content_id` digit-strips the fixture ids (`A1` → `1`). Production ids
  are numeric; the shim is dead weight + confusion there.
- **Fix:** give the fixture world numeric ids (`"101"…"308"`) so dev matches prod id
  shape, then delete the fallback. (Fixture titles/vectors unchanged; a handful of
  tests reference `A1`-style ids and need the same rename.)

## D5. `dashboard.html` size — **L** (watch, don't act yet)

Single inline-everything file is a *documented* choice (inert shell, no build step,
no secrets) and stays. But it crossed ~2600 lines this sprint. If it keeps growing:
split `<script>` into `static/dashboard.js` (same inert-shell property, still no
build step) before considering anything heavier. Not worth churn today.

## Plan — executed 2026-07-16, same day

All four steps landed in one pass; acceptance criteria verified (suite 136 green,
all tabs render error-free, one extractor, one aggregation loop, zero payload
field-chains left in dashboard.html). D5 remains a watch item.

### Original plan

One consolidation PR, no behavior change, ordered so each step is independently
shippable:

1. **D1** normalizer in `_payload_to_content` + fakes; migrate `served_explain`,
   `content_spread`; delete JS `_latlon` + fallback chains. *(~1.5h)*
2. **D2** `_cohort_view_stats` helper; migrate the three endpoints. *(~45m)*
3. **D4** numeric fixture ids; delete `digits()` shim; fix affected tests. *(~30m)*
4. **D3** `{value,label}` in `cohort_stats`; shrink `humanVal`. *(~30m)*

**Acceptance:** full test suite green; dashboard visually unchanged (cohort, content,
traffic, rec-detail, map); `grep -c "image_url" static/dashboard.html` drops to ~0;
exactly one lat/lon extractor and one views-aggregation loop in `src/`.
