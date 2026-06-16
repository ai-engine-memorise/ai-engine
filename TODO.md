# ai-engine — review TODO

Findings from the 2026-06-16 codebase review (recsys serving + search + lib + infra).
Severity: **H** high / **M** med / **L** low. Status: `[ ]` open · `[x]` done · `[~]` deferred (with reason).

> **v0.2.2 (this pass):** fixed all live search bugs, security guard, concurrency hardening,
> search correctness, dead code, and repo hygiene. 112 unit tests pass (+7 new).
> Deferred items are behavior decisions or refactors that need product input / cluster
> validation — each noted below.

## A. LIVE in serving image — active, user-facing

- [x] **H** `/api/search/preference` 500 for users with no positives → `user_searcher.py` now returns empty `SearchResult`.
- [x] **H** `common.py` `clean_payload_field` missing `return data` → fixed; `Item.locations`/`files_url`/images preserved (regression test added).
- [x] **H** `/debug/item_info` unauthenticated → now guarded by `_require_api_key`.
- [x] **M** search routes `async def` → blocking I/O → changed to `def` (threadpool).
- [x] **M** `/profile` built a new `ProjectionBuilder` per request → now reuses the router-scoped one.
- [x] **M** `/api/search` shape mismatch → returns `{"result": results.dict()}` like the siblings.

## B. Concurrency / online state (online path dormant: ranking_mode=static, bandit_online=False)

- [x] **H** shared bandit `policy` updated with no lock → added `_BANDIT_LOCK` (writer side) + per-event try/except so one bad event can't abort ingest.
- [~] **H** `redis_store.py` `consume` non-atomic get-then-set → deferred until `bandit_online` is enabled (needs `HDEL`/Lua atomic). Documented in code.
- [~] **H** per-worker `BANDIT_STATE_PATH` last-writer-wins → deferred (online off; replicas=1). Reader-side lock note left at `_BANDIT_LOCK`.

## C. Security / auth

- [x] **M** API-key compared with `!=` → now `hmac.compare_digest`.
- [x] **M** fail-open when `INGEST_API_KEY` unset → now fails closed (503) unless `AI_ENGINE_DEV=1`. (Tests run dev-mode via conftest.)
- [x] **L** Keycloak token POSTs had no `timeout` → added `timeout=30`.
- [~] **L** `config.py:33-39` SDU hosts/realm defaults → left; changing defaults breaks local dev. Low IP value (hosts already public DNS).
- [~] **L** `db_interface.py` table names f-string-interpolated → env-sourced (not user input); deferred, add identifier allowlist if those env vars ever become dynamic.

## D. Correctness — search / signals

- [x] **H** `global_searcher.similar()` `exclude_self` on dict items + `None` vector → robust id extraction + None guard.
- [x] **M** `help_searcher.get_item_by_item_id(int)` → `len()` TypeError → normalizes int/str/list.
- [x] **M** `geo_searcher.hybrid_search` deprecated `client.search` + hardcoded `limit=10` → migrated to `query_points` + `SEARCH_LIMIT`.
- [x] **M** `signal_builder` rating last-in-list → now latest-by-ts.
- [x] **L** `recommender.py` all-zero positives silently zeroed affinity → filter strictly-positive before `mx`.
- [~] **M** `user_searcher.py:43` deprecated `client.recommend` → still functional; deferred (mirror geo migration later).
- [~] **M** soft-negatives → hard `seen` exclusion (shown-once item banned forever). **Behavior decision**: keep as-is or make re-eligible/down-weight. Left for product call (changing it alters recommendations).

## E. Resource management / duplication

- [~] **M** `GlobalSearch` builds 4 QdrantClients → startup-time only (not per-request); deferred shared-client refactor. Per-request `ProjectionBuilder` already removed (A).
- [~] **M** `db_interface.py` engine-per-instance → now ~1 instance at startup (reused); deferred shared module engine.
- [~] **L** `user_to_query_text` duplicated → deferred (cosmetic).

## F. Dead code / cleanup

- [x] **M** `ingest_content.py` `import sentence_transformers` (torch) + data-file load at import → moved both into `__main__`; module now import-safe.
- [x] **L** dead `popularity` weight → removed from `FusionWeights` + `weighted_fuse`.
- [~] **L** `Candidate.base_score` unused / broken `__main__` demos / `SearchResult.dict()` name → left (harmless, low value).

## G. Repo hygiene / CI / build

- [x] **H** Deleted `.github/workflows/release.yml` (wrong pkg `ai_cdss`, fails every tag, misleading PyPI name).
- [x] **H** `src/ai_engine/events.db` → `git rm --cached` + `.gitignore *.db`.
- [x] **M** `Dockerfile.recsys` → pinned deps to ranges, reordered (deps layer before `src`), added `.dockerignore`. Non-root `USER` deferred (needs `fsGroup` for the `/app/logs` PVC + fastembed cache — validate on cluster first).
- [x] **M** `pyproject.toml` → moved `packages` under `[tool.poetry]`, added missing runtime deps (fastapi/uvicorn/redis/requests/pyarrow), added dev group + pytest config.
- [x] **M** Tests added: `test_common.py` (payload regression), `test_auth.py` (guard). Bandit-join / Keycloak / db_interface tests deferred (need heavy mocking).

## H. TODOs in source (tracked, not blocking)

- [~] **L** `ingest_content.py` content-medium / item-length / omeka_data TODOs → left as backlog markers.
