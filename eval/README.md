# LLM evaluation of recommendations

Phase-0 eval: judge recommendation relevance with an LLM, before real click data exists.

```
personas.json → POST /api/recommend/preview → top-k recs → Claude judges each (1-5) → report.json
```

## Run

```bash
pip install anthropic requests
export ANTHROPIC_API_KEY=...
# ai-engine recsys API must be running with a populated Qdrant collection
python eval/run.py --api http://localhost:8010 --k 8
python eval/run.py --model claude-haiku-4-5-20251001     # cheaper judge
```

Output: per-persona **mean relevance** + **precision@k** to stdout, full breakdown
(each rec + the LLM's score & reason) to `report.json`.

## Personas

`personas.json` — each has a `description` (what the LLM judges against) and a `model`
(the hand-authored user model fed to `/recommend/preview`):
- `tag_affinity`: `{"theme_what:Forced Labor": 1.0}` — tag-driven.
- `like_items`: `["2410"]` — semantic (taste vector from those items).
- `demographics`: `{"age_group":"25_34","gender":"female","nationality":"netherlands"}` — cold-start.

Edit/add personas freely. Tag labels must exist in the ingested content (canonical casing,
e.g. `theme_what:Forced Labor`).

## Notes

- Re-ingest content with the HTML-stripping fix first, or the LLM judges against markup-laden text.
- This complements the synthetic-scenario invariants in `tests/` (which check behaviour, not relevance).
- A weak score for a persona usually means the catalogue lacks matching content, not a recommender bug —
  check the returned `user_model` + `diagnostics.pool_size`.
