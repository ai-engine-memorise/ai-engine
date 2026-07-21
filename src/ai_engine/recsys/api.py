"""FastAPI surface for the recommendation engine.

- POST /api/ingest   : the ingest WEBHOOK. RudderStack POSTs user events here
                          (single object or list). Normalize -> buffer -> rebuild
                          the user model.
- GET  /api/recommend: serve recommendations for a user (reads the user model).
- GET  /api/usermodel: debug, inspect the current user model.

Mount `router` into the main service, or run `app` standalone. With no REDIS_URL /
QDRANT_API_URL set it runs fully in-memory on dev fixtures.
"""
from __future__ import annotations
import logging
import os
import threading
import time
from datetime import datetime, timezone
from typing import Any, Optional, Union
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Body, Query, Header, HTTPException, Depends
from pydantic import BaseModel, Field

from .composition import Components, build_components
from .adapters.rudderstack import normalize_events
from .contracts.models import UserSignals, InteractionEvent
from .survey import survey_affinity, canon_demo_value, demo_label


class PreviewSpec(BaseModel):
    """A hand-authored user model for testing recs without going through events."""
    tag_affinity: dict[str, float] = Field(default_factory=dict)   # {"theme_what:forced labor": 1.0}
    like_items: list[str] = Field(default_factory=list)            # -> taste vector (centroid) + excluded as seen
    demographics: dict = Field(default_factory=dict)               # {"age_group":"25_34","gender":"female",...}
    limit: Optional[int] = None


class EvalRun(BaseModel):
    """Run a synthetic persona across scenarios (module-level so FastAPI reads it as a body)."""
    spec: PreviewSpec = Field(default_factory=PreviewSpec)
    user_id: Optional[str] = None            # use an EXISTING visitor's live model instead of a spec
    scenarios: Optional[list[dict]] = None   # [{name, filter?, near_lat?, near_lon?, geo_radius_m?}]
    cold: bool = False                        # also run an empty (cold-start) baseline


def build_preview_signals(spec: PreviewSpec, content_store) -> UserSignals:
    taste = None
    if spec.like_items:
        vecs = content_store.get_vectors(spec.like_items)
        acc = None
        for v in vecs.values():
            if acc is None:
                acc = [0.0] * len(v)
            for i, x in enumerate(v):
                acc[i] += x
        if acc and any(acc):
            n = sum(x * x for x in acc) ** 0.5
            taste = [x / n for x in acc] if n else acc

    aff = dict(spec.tag_affinity)
    for k, w in survey_affinity(spec.demographics).items():
        aff[k] = aff.get(k, 0.0) + w
    folded: dict[str, float] = {}
    for k, v in aff.items():
        folded[k.lower()] = folded.get(k.lower(), 0.0) + v
    if folded:
        mx = max(folded.values())
        if mx > 0:
            folded = {k: v / mx for k, v in folded.items()}

    return UserSignals(
        user_id="preview",
        positives={i: 1.0 for i in spec.like_items},
        tag_affinity=folded,
        taste_vector=taste,
        demographics=spec.demographics,
    )

logger = logging.getLogger(__name__)

# serializes online bandit mutation (writer side). When bandit_online is enabled,
# the reader side (Recommender.policy.rank_scores) should be guarded with the same
# lock; today the online path is off by default (ranking_mode=static).
_BANDIT_LOCK = threading.Lock()


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Guard for write / inspection routes.

    The tenant + auth were resolved at the TRUST BOUNDARY (TenantASGIMiddleware) into the
    `auth_tenant` contextvar, which is:
      - "*"        -> the global INGEST_API_KEY (superuser)
      - "<tenant>" -> a valid per-tenant key, already PINNED to that tenant by the middleware
      - None       -> no/invalid key
    A per-tenant key therefore can only ever act on its OWN tenant (the middleware ignores a
    spoofed X-Tenant-Id), which closes the cross-tenant write hole.
    """
    import hmac
    from .tenancy import auth_tenant
    if auth_tenant.get() is not None:           # middleware validated: global "*" or a per-tenant key
        return
    # auth_tenant is None either because no/invalid key was presented OR the middleware isn't
    # installed (a fixed single-tenant app). Self-validate the global key directly so that path
    # still works.
    expected = os.getenv("INGEST_API_KEY")
    if expected:
        if x_api_key and hmac.compare_digest(x_api_key, expected):
            return
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    if x_api_key:                               # a key was sent but nothing to match it against
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")
    # no key presented AND none configured: dev-only passthrough, else fail closed
    if os.getenv("AI_ENGINE_DEV") == "1":
        logger.warning("INGEST_API_KEY unset (AI_ENGINE_DEV=1): write endpoints UNAUTHENTICATED")
        return
    raise HTTPException(status_code=503, detail="server auth not configured (INGEST_API_KEY unset)")


def _dump_items(items, include_content: bool) -> list:
    """Output contract: id / rank / relevance_score / role (+ breakdown, features, optional content).
    EVERY item is ranked by its position in the served ordering (1-based), distractor
    INCLUDED; the distractor is still flagged via role='distractor' so the UI can mark it."""
    out = []
    for rank, i in enumerate(items, start=1):
        d = i.model_dump()
        is_distractor = d.get("kind") == "distractor"
        item = {
            "id": d["content_id"],
            "rank": rank,
            "relevance_score": d["final_score"],
            "role": "distractor" if is_distractor else "target",
            "breakdown": d.get("breakdown", {}),
            "features": d.get("features", []),
        }
        if include_content and d.get("content") is not None:
            item["content"] = d["content"]
        out.append(item)
    return out


def _served_record(request_id: str, user_id: str, items: list, out: dict, filter: Optional[str]) -> dict:
    """Compact impression row for the durable served log (bandit training join key).
    Logs ids/ranks/roles + the per-item FEATURE VECTOR (the bandit context); content
    is recoverable from the content store."""
    distractor = next((it["id"] for it in items if it.get("role") == "distractor"), None)
    return {
        "request_id": request_id,
        "user_id": user_id,
        "ts": datetime.now(timezone.utc).isoformat(),
        "strategy": out.get("strategy"),
        "filter": filter,
        "cold_start": bool((out.get("diagnostics") or {}).get("cold_start_fallback")),
        "ranking": (out.get("diagnostics") or {}).get("ranking"),
        "distractor_id": distractor,
        "items": [{"id": it["id"], "rank": it["rank"], "role": it["role"],
                   "relevance_score": it.get("relevance_score"),
                   "breakdown": it.get("breakdown", {}),
                   "features": it.get("features", [])} for it in items],
    }


def _tail_served(base: str, n: int, light: bool = True, user_id: Optional[str] = None) -> list[dict]:
    """Latest n served-impression rows, newest first.

    log_served writes one parquet PER request into served/date=YYYY-MM-DD/, so the
    dataset is many tiny files. Reading them all to return a short tail is O(history),
    and even one busy day is O(requests/day). Strategy: walk date dirs newest-first
    (they sort chronologically), and within each list files WITHOUT reading them , 
    rank by mtime (≈ serve time) and read only the newest ~n parquet files. Cost is
    O(n) reads + cheap stat/sort over the newest day(s), not O(history).

    `light` (default) strips the per-item `breakdown`/`features` arrays from the rows:
    the traffic table needs only the item count and the per-impression metadata, and
    the drill-in (/served/explain) re-reads the full row from parquet, so the heavy
    float vectors never need to ride along in the list payload.
    """
    import glob
    import json
    import pyarrow.parquet as pq

    # newest day dirs first; collect candidate files (path only, no parquet read yet)
    # until we have >= n, so a short tail touches only the newest partition(s).
    day_dirs = sorted(glob.glob(os.path.join(base, "served", "date=*")), reverse=True)
    candidates: list[str] = []
    # a per-user tail can't know which files match without reading them, so pull a
    # deeper (but bounded) candidate window when filtering
    want_files = n if user_id is None else min(3000, max(200, n * 50))
    for d in day_dirs:
        candidates.extend(glob.glob(os.path.join(d, "*.parquet")))
        if len(candidates) >= want_files:
            break

    def _mtime(f: str) -> float:
        try:
            return os.path.getmtime(f)
        except OSError:
            return 0.0

    candidates.sort(key=_mtime, reverse=True)  # newest write (≈ serve) first

    rows: list = []
    for f in candidates:
        try:
            batch = pq.read_table(f).to_pylist()
        except Exception:
            continue
        if user_id is not None:
            batch = [r for r in batch if str(r.get("user_id") or "") == user_id]
        rows.extend(batch)
        if len(rows) >= n:    # mtime-desc -> stop after enough matching rows
            break
    rows.sort(key=lambda r: r.get("ts") or "", reverse=True)  # exact order by logged ts
    out = []
    for r in rows[:n]:
        items = r.get("items")
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                items = []
        if light and isinstance(items, list):
            items = [{k: it[k] for k in ("id", "rank", "role") if k in it}
                     for it in items if isinstance(it, dict)]
        out.append({**r, "items": items})
    return out


def _dash_cached(c, key: str, ttl: float, build):
    """Tiny per-tenant TTL cache for dashboard read models, hung off Components.

    Dashboard endpoints recompute heavy scans on every page view; serving never
    reads through here. Ingest drops the volatile keys (see /api/ingest) so the
    numbers stay live, and the TTL bounds staleness for everything else."""
    cache = getattr(c, "_dash_cache", None)
    if cache is None:
        cache = {}
        try:
            c._dash_cache = cache
        except Exception:          # exotic Components impl: fall back to uncached
            return build()
    hit = cache.get(key)
    now = time.monotonic()
    if hit is not None and now - hit[0] < ttl:
        return hit[1]
    val = build()
    cache[key] = (now, val)
    return val


def _cohort_view_aggs(c) -> list[tuple]:
    """[(signals, {content_id: ViewAgg})] for every visitor in the model store.

    THE cohort demand scan (docs/debt-payload-scatter.md D2): one implementation
    shared by /content/stats, /cohort/stats and /content/spread. Cached for 60s
    (invalidated on ingest) — at production scale the raw scan is one redis
    round-trip per visitor and dominated dashboard load times."""
    def build():
        from .signals.signal_builder import aggregate_views
        sigs = list(c.model_store.iter_signals() if hasattr(c.model_store, "iter_signals") else [])
        out = []
        for s in sigs:
            try:
                aggs = aggregate_views(c.event_buffer.fetch_events(s.user_id))
            except Exception:
                aggs = {}
            out.append((s, aggs))
        return out
    return _dash_cached(c, "cohort_aggs", 60.0, build)


def _iter_catalogue(c, cap: int = 2000):
    """Iterate the catalogue as normalized `Content` (fake store dict, or a capped
    Qdrant scroll through the payload normalizer). No raw payloads leave this point."""
    store = c.content_store
    if hasattr(store, "_contents"):
        yield from list(store._contents.values())[:cap]
        return
    client = getattr(store, "client", None)
    coll = getattr(store, "collection_name", None)
    if client is None or not coll:
        return
    from .adapters.qdrant_store import _payload_to_content
    off, n = None, 0
    while n < cap:
        try:
            points, off = client.scroll(collection_name=coll, limit=256, offset=off,
                                        with_payload=True, with_vectors=False)
        except Exception:
            return
        if not points:
            return
        for p in points:
            yield _payload_to_content(p.id, p.payload)
            n += 1
            if n >= cap:
                return
        if off is None:
            return


def _log_base(c) -> Optional[str]:
    """The CURRENT tenant's event-log directory (where served/ + date=*/ live).
    Per-tenant ParquetEventLog has `.base`; fall back to the global env for single-tenant."""
    base = getattr(getattr(c, "event_log", None), "base", None)
    return str(base) if base else os.getenv("EVENT_LOG_DIR")


def _durable_metrics(base: str) -> Optional[dict]:
    """Counts from the DURABLE Parquet log (survive restarts), not the in-process counters."""
    import glob
    try:
        import pyarrow.parquet as pq
    except ImportError:
        return None
    recs = cold = distr = items_total = 0
    for f in glob.glob(os.path.join(base, "served", "**", "*.parquet"), recursive=True):
        try:
            t = pq.read_table(f).to_pylist()
        except Exception:
            continue
        for r in t:
            recs += 1
            if r.get("strategy") == "cold":
                cold += 1
            if r.get("distractor_id"):
                distr += 1
            it = r.get("items")
            if isinstance(it, str):
                import json as _j
                try:
                    it = _j.loads(it)
                except Exception:
                    it = []
            items_total += len(it or [])
    ingests = 0
    for f in glob.glob(os.path.join(base, "date=*", "*.parquet")):
        try:
            ingests += pq.read_table(f).num_rows
        except Exception:
            continue
    return {"ingests": ingests, "recommends": recs, "cold": cold, "warm": recs - cold,
            "distractor_placed": distr, "items_total": items_total}


def _load_cluster_model(path: Optional[str] = None) -> Optional[dict]:
    """Lazily load the offline-trained cluster model (per-tenant path, else env)."""
    import json
    path = path or os.getenv("CLUSTER_MODEL_PATH")
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def _online_bandit_update(c: Components, events) -> int:
    """Online LinUCB: for each reward event echoing a request_id, look up the served
    feature vector and nudge theta by the realized engagement. Persists to BANDIT_STATE_PATH.
    No-op unless ranking_mode=bandit AND bandit_online. Good for the low-data regime."""
    cfg = c.cfg
    policy = getattr(c.recommender, "policy", None)
    if not (cfg.bandit_online and cfg.ranking_mode == "bandit" and policy is not None):
        return 0
    from .signals.engagement import estimate_reading_time, engagement_strength
    updated = 0
    with _BANDIT_LOCK:
        for e in events:
            try:
                if e.event != "CONTENT_VIEW_ENDED" or not e.request_id or not e.content_id:
                    continue
                x = c.impressions.get(e.request_id).get(e.content_id)
                if not x:
                    continue
                content = c.content_store.get([e.content_id]).get(e.content_id)
                est = estimate_reading_time(content.word_count, content.has_image, cfg) if content else 0.0
                reward = engagement_strength(dwell_seconds=e.dwell_seconds, est_reading_time=est,
                                             end_reason=e.end_reason, visits=1, survey_rating=None, cfg=cfg)
                policy.update(x, reward)
                c.impressions.consume(e.request_id, e.content_id)   # idempotency: one reward per impression
                updated += 1
            except Exception:
                logger.exception("online bandit update failed for one event; skipping")
                continue
        if updated:
            path = getattr(c, "bandit_state_path", None) or os.getenv("BANDIT_STATE_PATH")
            if path:
                try:
                    import json
                    tmp = path + ".tmp"
                    with open(tmp, "w", encoding="utf-8") as fh:
                        json.dump(policy.to_dict(), fh)
                    os.replace(tmp, path)
                except Exception:
                    logger.exception("failed to persist bandit state")
    return updated


def _tenant_doc(x_tenant_id: Optional[str] = Header(
        default=None, alias="X-Tenant-Id",
        description="Tenant id. Selects an isolated slice: its own Qdrant collection, "
                    "Redis namespace (user models / events / impressions / config), bandit policy, "
                    "clusters and event log. Omit to use the 'default' tenant. Resolution happens in "
                    "the ASGI tenant middleware; this declaration only documents the header.")) -> Optional[str]:
    return x_tenant_id


def make_router(components: Components) -> APIRouter:
    # Grouped by concern under one /api prefix. Guarded groups carry a single
    # router-level auth dependency instead of repeating it on every route.
    # _tenant_doc surfaces the X-Tenant-Id header in OpenAPI/Swagger for every /api route.
    router = APIRouter(prefix="/api", dependencies=[Depends(_tenant_doc)])
    # serving (recommend/search) is public by default so the UI sends only X-Tenant-Id.
    # Set SERVING_REQUIRES_KEY=1 to also require a (per-tenant) key here, closing the
    # cross-tenant READ hole — the UI then sends its tenant key too.
    _serving_deps = [Depends(_require_api_key)] if os.getenv("SERVING_REQUIRES_KEY") == "1" else []
    serving = APIRouter(tags=["serving"], dependencies=_serving_deps)                 # app-facing
    write = APIRouter(tags=["write"], dependencies=[Depends(_require_api_key)])       # ingest + preview
    usermodel_routes = APIRouter(tags=["usermodel"], dependencies=[Depends(_require_api_key)])
    ops = APIRouter(tags=["ops"], dependencies=[Depends(_require_api_key)])           # policy/metrics/served/stats/clusters
    admin = APIRouter(tags=["admin"], dependencies=[Depends(_require_api_key)])       # runtime config
    evalr = APIRouter(prefix="/eval", tags=["evaluation"], dependencies=[Depends(_require_api_key)])  # synthetic-visitor testing
    c = components
    # in-process counters for the inspector panel (v1; Prometheus /metrics is the prod upgrade)
    metrics = {"ingests": 0, "recommends": 0, "cold": 0, "warm": 0,
               "pool_total": 0, "distractor_requested": 0, "distractor_placed": 0}

    def _record(rec_out: dict) -> None:
        metrics["recommends"] += 1
        diag = rec_out.get("diagnostics") or {}
        metrics["cold" if rec_out.get("strategy") == "cold" else "warm"] += 1
        metrics["pool_total"] += int(diag.get("pool_size") or 0)
        d = diag.get("distractor")
        if isinstance(d, dict):
            metrics["distractor_requested"] += 1
            if d.get("content_id"):
                metrics["distractor_placed"] += 1

    @write.post("/ingest")
    def ingest(payload: Union[dict, list] = Body(...)) -> dict:
        raws = payload if isinstance(payload, list) else [payload]
        events = normalize_events(raws)
        c.event_log.append(events)    # durable append-only log (Parquet) — training/eval record
        users: set[str] = set()
        for e in events:
            c.event_buffer.append(e)  # type: ignore[attr-defined]
            users.add(e.user_id)
        now = datetime.now(timezone.utc)
        for u in users:
            demographics = c.demographics.get_demographics(u)
            c.updater.refresh(u, c.event_buffer, now=now, demographics=demographics)
        metrics["ingests"] += len(events)
        bandit_updates = _online_bandit_update(c, events)
        # new events change visitor rows and cohort aggregates immediately;
        # the served-log scan cache is unaffected by ingest and keeps its TTL
        cache = getattr(c, "_dash_cache", None)
        if cache:
            cache.pop("users_rows", None)
            cache.pop("cohort_aggs", None)
            cache.pop("fb_scores", None)
        return {"status": "ok", "ingested": len(events), "users": sorted(users),
                "bandit_updates": bandit_updates}

    @serving.get("/recommend")
    def recommend(
        user_id: str = Query(..., examples=["u1"]),
        limit: Optional[int] = Query(default=None, ge=1, le=50),
        filter: Optional[str] = Query(default=None, description="restrict candidates to a tag, e.g. a location: AiARLocationBarrack3"),
        near_lat: Optional[float] = Query(default=None, description="user's current latitude (geo proximity, independent of tag filter)"),
        near_lon: Optional[float] = Query(default=None, description="user's current longitude"),
        geo_radius_m: Optional[float] = Query(default=None, ge=0, description="also restrict to this radius (metres) around near_lat/lon"),
        include_content: bool = Query(default=True, description="false = compact (ids/scores only)"),
    ) -> dict:
        near = (near_lat, near_lon) if near_lat is not None and near_lon is not None else None
        rec = c.recommender.recommend(user_id, filter=filter, near=near, geo_radius_m=geo_radius_m)
        items = rec.items[:limit] if limit else rec.items
        out = rec.model_dump()
        out["filter"] = filter
        out["items"] = _dump_items(items, include_content)
        request_id = uuid4().hex
        out["request_id"] = request_id   # app echoes this on CONTENT_VIEW -> joins impression to outcome
        c.event_log.log_served(_served_record(request_id, user_id, out["items"], out, filter))
        # stash served feature vectors so a later reward event can update the bandit online
        c.impressions.put(request_id, {it["id"]: it["features"] for it in out["items"] if it.get("features")})
        _record(out)
        return {"result": out}

    @serving.get("/whoami")
    def whoami() -> dict:
        """The tenant + auth scope resolved from the request (key-derived, never spoofable).
        Public so the dashboard can label the active tenant without exposing any data."""
        from .tenancy import current_tenant, auth_tenant
        a = auth_tenant.get()
        scope = "global" if a == "*" else ("tenant" if a else "public")
        collection = getattr(c.content_store, "collection_name", None)   # fakes -> None
        return {"result": {"tenant": current_tenant.get() or "default", "scope": scope,
                           "collection": collection}}

    @usermodel_routes.get("/usermodel")
    def usermodel(user_id: str = Query(..., examples=["u1"])) -> dict:
        # debug/inspection endpoint — exposes demographics (PII), so it is guarded
        # by the same INGEST_API_KEY. The serving /recommend stays open for the app.
        sig = c.model_store.get_signals(user_id)
        return {"result": sig.model_dump() if sig else None}

    @usermodel_routes.get("/usermodel/explain")
    def usermodel_explain(
        user_id: str = Query(..., examples=["u1"]),
        verbalize: bool = Query(default=True, description="include a prose summary"),
    ) -> dict:
        """Glass-box persona: Falk visitor type + Pekarik preference + interests/aversions
        (with evidence) + engagement style + thematic trajectory, derived from the model."""
        from .explain.persona import explain_user
        from .explain.verbalize import verbalize as _verbalize
        sig = c.model_store.get_signals(user_id)
        if sig is None:
            return {"result": None}
        # pull tags for the user's touched content -> interest evidence + trajectory
        ids = list(dict.fromkeys(list(sig.positives) + list(sig.negatives) + sig.recent_views))
        contents = c.content_store.get(ids) if ids else {}
        exp = explain_user(sig, contents)
        if verbalize:
            exp.summary = _verbalize(exp)
        out = exp.model_dump()
        model = _load_cluster_model(getattr(c, "cluster_model_path", None))
        if model:                                  # place the visitor in a learned segment
            from .explain.clusters import assign, assign_fuzzy
            out["cluster"] = (assign_fuzzy(sig, model) if model.get("method") == "fcm"
                              else assign(sig, model))
        return {"result": out}

    @usermodel_routes.get("/usermodel/extras")
    def usermodel_extras(user_id: str = Query(..., examples=["u1"])) -> dict:
        """Holistic visitor profile extras that complement /usermodel/explain:
        demographics, the raw survey answers (presurvey + personalization), and the
        dynamic user model (tag affinities/aversions, engagement behaviour, and the
        content the visitor engaged with positively together with that content's
        tags, i.e. *why* we think the visitor likes certain things)."""
        from .survey import extract_demographics, split_survey_answers

        sig = c.model_store.get_signals(user_id)
        if sig is None:
            return {"result": None}

        # Raw survey answers, merged across all of this visitor's events (latest wins).
        answers: dict = {}
        for ev in (c.event_buffer.fetch_events(user_id) or []):
            for k, val in (getattr(ev, "survey_answers", None) or {}).items():
                if val is not None and val != "":
                    answers[k] = val
        grouped = split_survey_answers(answers)

        # personalization answers arrive as internal codes: AR location tags, content
        # ids, taxonomy slugs. Humanize for the profile card (ids -> content titles).
        import re as _re2
        def _pretty_pers(key: str, val):
            vals = val if isinstance(val, list) else [val]
            out = []
            if key.endswith("interest") and all(str(x).isdigit() for x in vals):
                got = c.content_store.get([str(x) for x in vals])
                for x in vals:
                    ct = got.get(str(x))
                    out.append(getattr(ct, "title", None) or str(x))
            else:
                for x in vals:
                    t = _re2.sub(r"^AiARLocation", "", str(x))
                    t = _re2.sub(r"(?<=[a-z])(?=[A-Z0-9])", " ", t)   # Barrack56 -> Barrack 56
                    out.append(t.replace("_", " ").strip().title() if "_" in t or t.islower() else t)
            return out if isinstance(val, list) else out[0]
        pers = {k: _pretty_pers(k, v) for k, v in (grouped["personalization"] or {}).items()}

        surveys = {
            "demographic": grouped["demographic"],
            "personalization": pers,
            "background": grouped.get("background") or {},
            "feedback": grouped.get("feedback") or {},
            "other": grouped["other"],
            "completed_demographic": bool(grouped["demographic"]),
            "completed_personalization": bool(grouped["personalization"]),
            "answer_count": len(answers),
        }

        # demographics: canonicalized (language variants folded, junk -> no_answer/
        # dropped) so the card never shows raw Dutch strings or zero-width blanks
        raw_demo = dict(sig.demographics or {}) or extract_demographics(answers)
        demographics = {}
        for k, v in raw_demo.items():
            cv = canon_demo_value(k, v)
            if cv:
                demographics[k] = cv

        # Dynamic model: content the visitor liked, with that content's tags.
        positives = sig.positives or {}
        liked_ids = sorted(positives, key=lambda k: positives[k], reverse=True)[:15]
        contents = c.content_store.get(liked_ids) if liked_ids else {}
        liked_content = []
        for cid in liked_ids:
            ct = contents.get(cid)
            liked_content.append({
                "id": cid,
                "title": getattr(ct, "title", None) if ct else None,
                "weight": round(float(positives.get(cid, 0.0)), 3),
                "tags": [t.key for t in getattr(ct, "tags", [])] if ct else [],
            })

        def _top(d, n=12):
            d = d or {}
            ranked = sorted(d.items(), key=lambda kv: abs(kv[1]), reverse=True)[:n]
            return [{"key": k, "weight": round(float(v), 3)} for k, v in ranked]

        model = {
            "tag_affinity": _top(sig.tag_affinity),
            "tag_aversion": _top(sig.tag_aversion),
            "behavior": dict(sig.behavior or {}),
            "liked_content": liked_content,
            "counts": {
                "positives": len(sig.positives or {}),
                "negatives": len(sig.negatives or {}),
                "viewed": len(sig.viewed or []),
            },
            "is_cold": bool(getattr(sig, "is_cold", False) or not (sig.viewed)),
        }

        return {"result": {"demographics": demographics, "surveys": surveys, "model": model}}

    @usermodel_routes.get("/usermodel/history")
    def usermodel_history(
        user_id: str = Query(..., examples=["u1"]),
        limit: int = Query(default=200, ge=1, le=2000),
    ) -> dict:
        """Raw visitor activity behind the model: the event timeline (views, dwell,
        end-reason, ts) + a per-content aggregate (visits/dwell/outcome). PII -> guarded."""
        from .signals.signal_builder import aggregate_views
        events = sorted(c.event_buffer.fetch_events(user_id), key=lambda e: e.ts)  # type: ignore[attr-defined]
        sig = c.model_store.get_signals(user_id)
        pos = set(sig.positives) if sig else set()
        neg = set(sig.negatives) if sig else set()
        aggs = aggregate_views(events)
        titles = {k: v.title for k, v in c.content_store.get(list(aggs.keys())).items()} if aggs else {}
        outcome = lambda cid: "positive" if cid in pos else ("negative" if cid in neg else "neutral")

        agg_rows = [{
            "content_id": cid, "title": titles.get(cid, ""), "visits": a.visits,
            "dwell_seconds": a.dwell_seconds,
            "end_reason": a.end_reason.value if a.end_reason else None,
            "last_ts": a.last_ts.isoformat() if a.last_ts else None,
            "outcome": outcome(cid),
        } for cid, a in aggs.items()]
        agg_rows.sort(key=lambda r: r["last_ts"] or "", reverse=True)

        def ev_meta(e) -> dict:
            """Compact disambiguation payload for the timeline: everything in the raw
            event's details/context (panel ids, survey ids, state names …) that the
            fixed columns don't already show. Lets the operator tell WHICH survey was
            presented or WHICH panel started without leaving the dashboard."""
            raw = e.raw or {}
            props = raw.get("properties") or {}
            det = props.get("details") or {}
            ctx = props.get("context") or {}
            out: dict = {}
            # keep the raw close reason when it is NOT one of the standard enum
            # values (apply_button, menu_button, ... are panel outcomes worth showing)
            drop = {"request_id", "session_id"} | ({"reason"} if e.end_reason else set())
            for k, v in {**det, **{k2: v2 for k2, v2 in ctx.items() if k2 != "candidates"}}.items():
                if v not in (None, "", []) and k not in drop:
                    out[k] = v
            if ctx.get("candidates"):
                out["candidates"] = len(ctx["candidates"])
            for k, v in props.items():   # props outside the known containers
                if k in ("content", "details", "context", "answers"):
                    continue
                if isinstance(v, (str, int, float, bool)) and v != "":
                    out[k] = v
                elif isinstance(v, dict):   # e.g. properties.panel.{panel_id,panel_name,panel_type}
                    for k2, v2 in v.items():
                        if isinstance(v2, (str, int, float, bool)) and v2 != "":
                            out[k2] = v2
            if e.survey_answers:
                out["answers"] = e.survey_answers
            return out

        ev_rows = [{
            "ts": e.ts.isoformat() if e.ts else None, "event": e.event,
            "content_id": e.content_id, "title": titles.get(e.content_id, ""),
            "dwell_seconds": e.dwell_seconds,
            "end_reason": e.end_reason.value if e.end_reason else None,
            "request_id": e.request_id, "session_id": e.session_id,
            "meta": ev_meta(e),
        } for e in sorted(events, key=lambda e: e.ts, reverse=True)[:limit]]

        # ----- per-session summary: duration, content viewed, survey timings -----
        by_session: dict[str, list] = {}
        for e in events:                                   # events sorted asc
            by_session.setdefault(e.session_id or "", []).append(e)
        if list(by_session.keys()) == [""]:
            # the app sends no session id — derive sessions from SESSION_STARTED
            # markers and >30 min silence gaps instead of one giant bucket
            segs: list[list] = []
            cur: list = []
            prev_ts = None
            for e in events:
                gap = prev_ts is not None and e.ts is not None and (e.ts - prev_ts).total_seconds() > 1800
                if cur and (e.event == "SESSION_STARTED" or gap):
                    segs.append(cur)
                    cur = []
                cur.append(e)
                prev_ts = e.ts or prev_ts
            if cur:
                segs.append(cur)
            by_session = {f"session {i + 1}": evs2 for i, evs2 in enumerate(segs)}
        session_rows = []
        for sid, evs in by_session.items():
            tss = [e.ts for e in evs if e.ts]
            surveys = []
            presented = None
            for e in evs:
                if e.event == "SURVEY_PRESENTED":
                    presented = e
                elif presented is not None and e.event in ("SURVEY_SUBMITTED", "SURVEY_DISMISSED"):
                    surveys.append({
                        "presented_ts": presented.ts.isoformat() if presented.ts else None,
                        "duration_seconds": round((e.ts - presented.ts).total_seconds(), 1)
                                            if (e.ts and presented.ts) else None,
                        "ended": e.event,
                        "survey": (ev_meta(presented).get("survey_id")
                                   or ev_meta(presented).get("survey")
                                   or ev_meta(presented).get("survey_name")),
                    })
                    presented = None
            session_rows.append({
                "session_id": sid or None,
                "start": min(tss).isoformat() if tss else None,
                "end": max(tss).isoformat() if tss else None,
                "duration_seconds": round((max(tss) - min(tss)).total_seconds(), 1) if len(tss) > 1 else 0,
                "n_events": len(evs),
                "n_views": sum(1 for e in evs if e.event == "CONTENT_VIEW_STARTED"),
                "surveys": surveys,
            })
        session_rows.sort(key=lambda r: r["start"] or "", reverse=True)

        return {"result": {"user_id": user_id, "event_count": len(events),
                           "sessions": session_rows,
                           "aggregates": agg_rows, "events": ev_rows}}

    @ops.get("/clusters")
    def clusters() -> dict:
        """The explainable visitor segments (offline-trained). Each cluster is described
        by its top taxonomy tags + a Falk breadth hint. CLUSTER_MODEL_PATH must be set."""
        model = _load_cluster_model(getattr(c, "cluster_model_path", None))
        if not model:
            return {"result": None, "detail": "CLUSTER_MODEL_PATH not set / file missing"}
        return {"result": {"method": model.get("method", "kmeans"), "profiles": model.get("profiles", [])}}

    @ops.get("/users")
    def users_list(limit: int = Query(default=500, ge=1, le=5000)) -> dict:
        """Per-visitor summary for the cohort table: interaction count, last interaction,
        liked / seen totals, cold flag, and segment. Visitors are enumerated from the
        MODEL STORE (every ingested visitor has signals; replay the event log to restore
        them after a store wipe) — not by scanning the whole parquet event log per
        request, which is what made this endpoint take seconds in production. The served
        log contributes serve counts and visitors who were served but never ingested;
        that scan is cached for 5 min, the assembled rows for 60s (dropped on ingest)."""
        def build_served():
            import glob
            n_served: dict[str, int] = {}
            served_last: dict[str, str] = {}
            base = _log_base(c)
            if base:
                try:
                    import pyarrow.parquet as pq
                except ImportError:
                    pq = None
                if pq is not None:
                    # per-file failures skip that file only, never the rest of the scan
                    for f in glob.glob(os.path.join(base, "served", "**", "*.parquet"), recursive=True):
                        try:
                            srows = pq.read_table(f, columns=["user_id", "ts"]).to_pylist()
                        except Exception:
                            try:
                                srows = pq.read_table(f).to_pylist()
                            except Exception:
                                continue
                        for r in srows:
                            u, ts = r.get("user_id"), r.get("ts")
                            if not u:
                                continue
                            u = str(u)
                            n_served[u] = n_served.get(u, 0) + 1
                            ts = str(ts) if ts else ""
                            if ts and ts > served_last.get(u, ""):
                                served_last[u] = ts
            return n_served, served_last

        def build_rows():
            from .signals.signal_builder import aggregate_views
            seg: dict[str, str] = {}
            model = _load_cluster_model(getattr(c, "cluster_model_path", None))
            if model:
                for p in model.get("profiles", []):
                    label = f"Cluster {p.get('cluster')}" + (f" · {p['falk_hint']}" if p.get("falk_hint") else "")
                    for u in (p.get("members") or []):
                        seg[u] = label
            n_served, served_last = _dash_cached(c, "served_scan", 300.0, build_served)
            sigs = {s.user_id: s for s in
                    (c.model_store.iter_signals() if hasattr(c.model_store, "iter_signals") else [])}
            ids = set(sigs) | set(seg) | set(n_served)
            rows = []
            for uid in ids:
                sig = sigs.get(uid)
                try:
                    evs = c.event_buffer.fetch_events(uid)
                except Exception:
                    evs = []
                last = max((e.ts for e in evs if e.ts), default=None)
                try:      # views = CONTENT views (paired start/end via aggregate_views),
                          # NOT raw event count — sessions/panels/surveys are events too
                    n_views = sum((a.visits or 0) for a in aggregate_views(evs).values())
                except Exception:
                    n_views = 0
                fb_scores = {}
                try:
                    from .survey import split_survey_answers
                    merged_ans: dict = {}
                    for e in evs:
                        for k2, v2 in (getattr(e, "survey_answers", None) or {}).items():
                            if v2 not in (None, ""):
                                merged_ans[k2] = v2
                    if merged_ans:
                        for k2, v2 in (split_survey_answers(merged_ans).get("feedback") or {}).items():
                            if isinstance(v2, dict) and v2.get("score") is not None:
                                fb_scores[k2] = v2["score"]
                except Exception:
                    pass
                rows.append({
                    "user_id": uid,
                    "feedback": fb_scores,
                    # anonymous = never answered any survey (no identify/survey event)
                    "anonymous": not any(getattr(e, "survey_answers", None) for e in evs),
                    "n_views": n_views,
                    "n_interactions": len(evs),
                    "n_served": n_served.get(uid, 0),
                    "last_interaction": last.isoformat() if last else served_last.get(uid),
                    "n_liked": len(sig.positives) if sig else 0,
                    "n_seen": len(getattr(sig, "viewed", []) or []) if sig else 0,
                    "cold": bool(sig.is_cold) if sig else True,
                    "segment": seg.get(uid),
                    "demographics": ({k: (canon_demo_value(k, v) or "")
                                      for k, v in (getattr(sig, "demographics", None) or {}).items()}
                                     if sig else {}),
                })
            rows.sort(key=lambda r: r["last_interaction"] or "", reverse=True)
            return rows

        rows = _dash_cached(c, "users_rows", 60.0, build_rows)
        return {"result": rows[:limit]}

    @ops.get("/cohort/stats")
    def cohort_stats(
        age: Optional[str] = Query(default=None),
        gender: Optional[str] = Query(default=None),
        nationality: Optional[str] = Query(default=None),
        province: Optional[str] = Query(default=None),
        personal_connection: Optional[str] = Query(default=None),
        email_shared: Optional[str] = Query(default=None),
        feedback: Optional[str] = Query(default=None, description="post-visit score ranges, e.g. boring:4-5,interest:1-2"),
    ) -> dict:
        """Cohort-wide aggregates for the Cohort tab: demographic distributions +
        behaviour statistics (volumes, depth, how views end), optionally FILTERED to a
        demographic slice. Each filter param takes one value or several comma-separated
        values (OR within a field, AND across fields). Faceted counting: each demographic
        field's distribution is computed against the OTHER active filters (so the selected
        values can be switched), while the behaviour numbers respect ALL filters.
        PII-guarded (counts only)."""
        pairs = _cohort_view_aggs(c)          # the shared cohort demand scan (D2)
        sigs = [s for s, _ in pairs]
        filters = {k: set(v.split(",")) for k, v in (
            ("age", age), ("gender", gender), ("nationality", nationality),
            ("province", province), ("personal_connection", personal_connection),
            ("email_shared", email_shared)) if v}
        fb_ranges: dict[str, tuple] = {}
        if feedback:
            for part in feedback.split(","):
                try:
                    q, rng = part.split(":", 1)
                    lo, hi = rng.split("-", 1)
                    fb_ranges[q.strip()] = (int(lo), int(hi))
                except Exception:
                    continue
        fb_map = _fb_scores_map() if fb_ranges else {}

        def matches(s, skip: Optional[str] = None) -> bool:
            demo = getattr(s, "demographics", None) or {}
            if not all((canon_demo_value(f, demo.get(f, "")) or "") in vals
                       for f, vals in filters.items() if f != skip):
                return False
            if fb_ranges:
                scores = fb_map.get(s.user_id) or {}
                for q, (lo, hi) in fb_ranges.items():
                    sc = scores.get(q)
                    if sc is None or sc < lo or sc > hi:
                        return False
            return True

        demographics: dict[str, dict[str, int]] = {}
        for s in sigs:
            for k, v in (getattr(s, "demographics", None) or {}).items():
                cv = canon_demo_value(str(k), v)   # folds language variants, drops junk
                if cv is None or not matches(s, skip=str(k)):
                    continue
                field = demographics.setdefault(str(k), {})
                field[cv] = field.get(cv, 0) + 1

        cohort = [(s, aggs) for s, aggs in pairs if matches(s)]
        total_events = total_views = positive = negative = warm = 0
        dwell_sum = 0.0
        dwell_views = 0
        end_reasons: dict[str, int] = {}
        for s, aggs in cohort:
            try:
                total_events += len(c.event_buffer.fetch_events(s.user_id))
            except Exception:
                pass
            for a in aggs.values():
                total_views += a.visits or 0
                if a.dwell_seconds:
                    dwell_sum += a.dwell_seconds
                    dwell_views += a.visits or 1
                if a.end_reason:
                    r = a.end_reason.value
                    end_reasons[r] = end_reasons.get(r, 0) + 1
            positive += len(s.positives or {})
            negative += len(s.negatives or {})
            if not getattr(s, "is_cold", True):
                warm += 1

        n = len(cohort)
        return {"result": {
            "visitors": n, "total_visitors": len(sigs), "warm": warm, "cold": n - warm,
            "events": total_events, "views": total_views,
            "views_per_visitor": round(total_views / n, 1) if n else 0,
            "avg_dwell_seconds": round(dwell_sum / dwell_views, 1) if dwell_views else 0,
            "positive": positive, "negative": negative,
            "end_reasons": end_reasons,
            "demographics": demographics,
            "labels": {f: {v: demo_label(f, v) for v in dist} for f, dist in demographics.items()},
            "filters": {k: sorted(v) for k, v in filters.items()},
        }}

    def _fb_scores_map() -> dict:
        """{user_id: {question: score}} over the whole model store, cached 60s.
        Powers post-visit feedback filtering of the cohort statistics."""
        def build():
            from .survey import split_survey_answers
            out: dict[str, dict] = {}
            sigs = list(c.model_store.iter_signals() if hasattr(c.model_store, "iter_signals") else [])
            for s2 in sigs:
                try:
                    evs = c.event_buffer.fetch_events(s2.user_id)
                except Exception:
                    continue
                answers: dict = {}
                for e in evs:
                    for k, v in (getattr(e, "survey_answers", None) or {}).items():
                        if v not in (None, ""):
                            answers[k] = v
                if not answers:
                    continue
                fb = split_survey_answers(answers).get("feedback") or {}
                scores = {k: v["score"] for k, v in fb.items()
                          if isinstance(v, dict) and v.get("score") is not None}
                if scores:
                    out[s2.user_id] = scores
            return out
        return _dash_cached(c, "fb_scores", 60.0, build)

    @ops.get("/cohort/feedback")
    def cohort_feedback(
        age: Optional[str] = Query(default=None),
        gender: Optional[str] = Query(default=None),
        nationality: Optional[str] = Query(default=None),
        province: Optional[str] = Query(default=None),
        personal_connection: Optional[str] = Query(default=None),
        email_shared: Optional[str] = Query(default=None),
        feedback: Optional[str] = Query(default=None),
    ) -> dict:
        """Post-visit rating aggregates (mean, count, score distribution) for the
        FILTERED cohort. Faceted like the demographic panels: each question's numbers
        respect every active filter EXCEPT its own score range, so its selection stays
        adjustable while everything reflects the selected group."""
        filters = {k: set(v.split(",")) for k, v in (
            ("age", age), ("gender", gender), ("nationality", nationality),
            ("province", province), ("personal_connection", personal_connection),
            ("email_shared", email_shared)) if v}
        fb_ranges: dict[str, tuple] = {}
        if feedback:
            for part in feedback.split(","):
                try:
                    q, rng = part.split(":", 1)
                    lo, hi = rng.split("-", 1)
                    fb_ranges[q.strip()] = (int(lo), int(hi))
                except Exception:
                    continue
        fb_map = _fb_scores_map()

        def build():
            sigs = list(c.model_store.iter_signals() if hasattr(c.model_store, "iter_signals") else [])
            def demo_ok(s2):
                demo = getattr(s2, "demographics", None) or {}
                return all((canon_demo_value(f, demo.get(f, "")) or "") in vals
                           for f, vals in filters.items())
            questions = sorted({q for scores in fb_map.values() for q in scores})
            out = []
            for q in questions:
                agg = {"sum": 0, "n": 0, "dist": {}}
                for s2 in sigs:
                    scores = fb_map.get(s2.user_id)
                    if not scores or q not in scores:
                        continue
                    if not demo_ok(s2):
                        continue
                    skip = False        # other questions' ranges apply; own range doesn't
                    for oq, (lo, hi) in fb_ranges.items():
                        if oq == q:
                            continue
                        sc2 = scores.get(oq)
                        if sc2 is None or sc2 < lo or sc2 > hi:
                            skip = True
                            break
                    if skip:
                        continue
                    sc = scores[q]
                    agg["sum"] += sc
                    agg["n"] += 1
                    agg["dist"][str(sc)] = agg["dist"].get(str(sc), 0) + 1
                if agg["n"]:
                    out.append({"question": q, "avg": round(agg["sum"] / agg["n"], 2),
                                "n": agg["n"], "scale": 5, "dist": agg["dist"]})
            return out
        if not filters and not fb_ranges:
            return {"result": _dash_cached(c, "feedback_agg", 300.0, build)}
        return {"result": build()}

    @ops.get("/cohort/timeline")
    def cohort_timeline() -> dict:
        """Distinct visitors (and events) per day over the full durable log, for the
        activity chart. One columnar pass per day partition, cached 5 min."""
        def build():
            import glob as _glob
            try:
                import pyarrow.parquet as pq
            except ImportError:
                raise HTTPException(status_code=503, detail="pyarrow not installed")
            base = _log_base(c)
            day_users: dict[str, set] = {}
            day_events: dict[str, int] = {}
            responded: set = set()          # visitors who EVER answered a survey
            if base:
                for d in sorted(_glob.glob(os.path.join(base, "date=*"))):
                    day = os.path.basename(d).split("=", 1)[1]
                    users = day_users.setdefault(day, set())
                    for f in _glob.glob(os.path.join(d, "*.parquet")):
                        try:
                            t = pq.read_table(f, columns=["user_id", "survey_answers"])
                        except Exception:
                            continue
                        day_events[day] = day_events.get(day, 0) + t.num_rows
                        for u, sa in zip(t["user_id"].to_pylist(), t["survey_answers"].to_pylist()):
                            if not u:
                                continue
                            users.add(u)
                            if sa and sa not in ("{}", "null", ""):
                                responded.add(u)
            return [{"date": day,
                     "visitors": len(users),
                     "responded": len(users & responded),
                     "anonymous": len(users - responded),
                     "events": day_events.get(day, 0)}
                    for day, users in sorted(day_users.items())]
        return {"result": _dash_cached(c, "timeline", 300.0, build)}

    @ops.get("/content/stats")
    def content_stats(limit: int = Query(default=30, ge=1, le=200)) -> dict:
        """Cohort-wide content engagement: which items are seen / liked / abandoned across
        all visitors, popular themes, and each cluster's content preferences. PII-guarded."""
        pairs = _cohort_view_aggs(c)          # the shared cohort demand scan (D2)
        sigs = [s for s, _ in pairs]
        visits: dict[str, int] = {}          # total view events (times viewed)
        dwell: dict[str, float] = {}         # summed dwell seconds
        exposed: dict[str, set] = {}         # unique users who saw it
        positive: dict[str, int] = {}        # users with a positive outcome
        theme: dict[str, float] = {}
        for s, aggs in pairs:
            for cid, a in aggs.items():
                visits[cid] = visits.get(cid, 0) + (a.visits or 0)
                dwell[cid] = dwell.get(cid, 0.0) + (a.dwell_seconds or 0.0)
                exposed.setdefault(cid, set()).add(s.user_id)
            for cid in s.viewed:             # served/seen even without a dwell event
                exposed.setdefault(cid, set()).add(s.user_id)
            for cid in s.positives:
                positive[cid] = positive.get(cid, 0) + 1
            for k, w in s.tag_affinity.items():
                if w > 0:                      # themes = HOW MANY visitors lean toward the
                    theme[k] = theme.get(k, 0) + 1   # tag, not an opaque summed weight

        all_cids = list(set(visits) | set(exposed) | set(positive))
        titles = {k: v.title for k, v in c.content_store.get(all_cids).items()} if all_cids else {}
        lbl = lambda k: k.split(":", 1)[1] if ":" in k else k

        content = []
        for cid in all_cids:
            v_ = visits.get(cid, 0)
            users_n = len(exposed.get(cid, ()))
            pos = positive.get(cid, 0)
            content.append({
                "content_id": cid, "title": titles.get(cid, ""),
                "views": v_,                                          # times viewed (events)
                "users": users_n,                                    # unique users exposed
                "avg_dwell": round(dwell.get(cid, 0.0) / v_, 1) if v_ else 0.0,
                "positive": pos,
                "positive_rate": round(pos / users_n, 3) if users_n else 0.0,   # positive vs others
            })
        content.sort(key=lambda r: (r["views"], r["users"]), reverse=True)

        # popular themes: only content-describing taxonomy facets. Demographic-seeded
        # affinities (person_who.*) describe who the visitor is, not what content is
        # popular, so they are excluded. Keys stay FULL so the UI can group by facet.
        _theme_facets = ("theme_what", "theme_how", "place_where", "location")
        themes = sorted(([k, round(w, 3)] for k, w in theme.items()
                         if k.split(":", 1)[0].split(".")[0] in _theme_facets),
                        key=lambda kv: kv[1], reverse=True)[:40]

        clusters = []
        model = _load_cluster_model(getattr(c, "cluster_model_path", None))
        if model:
            by_user = {s.user_id: s for s in sigs}
            for p in model.get("profiles", []):
                cl: dict[str, int] = {}
                clt: dict[str, float] = {}
                for u in p.get("members", []):
                    s = by_user.get(u)
                    if not s:
                        continue
                    for cid in s.positives:
                        cl[cid] = cl.get(cid, 0) + 1
                    for k, w in s.tag_affinity.items():
                        clt[k] = clt.get(k, 0.0) + w
                clusters.append({
                    "cluster": p["cluster"], "size": p.get("size"),
                    "top_content": [{"content_id": cid, "title": titles.get(cid, ""), "likes": n}
                                    for cid, n in sorted(cl.items(), key=lambda kv: kv[1], reverse=True)[:5]],
                    "top_themes": [{"label": lbl(k), "weight": round(w, 3)}
                                   for k, w in sorted(clt.items(), key=lambda kv: kv[1], reverse=True)[:6]],
                })

        return {"result": {"users": len(sigs), "content": content[:limit],
                           "themes": themes, "clusters": clusters}}

    # ----- SNAPSHOTS: stored, listable, downloadable point-in-time exports -----
    # A snapshot is a gzipped flat CSV of the full durable event log (one row per
    # event, complex fields as JSON strings) plus a .json sidecar with headline
    # stats, written to <EVENT_LOG_DIR>/<tenant>/snapshots/. The heavy log scan
    # happens ONLY when one is taken — listing is a directory read, so the tab
    # loads instantly (the previous scan-on-view 502'd behind the proxy timeout).
    import re as _re_snap
    _SNAP_NAME = _re_snap.compile(r"^[A-Za-z0-9_-]+-\d{8}-\d{6}\.csv\.gz$")

    def _snap_dir() -> str:
        base = _log_base(c)
        if not base:
            raise HTTPException(status_code=503, detail="EVENT_LOG_DIR not set")
        d = os.path.join(base, "snapshots")
        os.makedirs(d, exist_ok=True)
        return d

    def _snap_meta_path(p: str) -> str:
        return p[:-len(".csv.gz")] + ".json"

    @ops.post("/export/compact")
    def compact_log() -> dict:
        """Merge each day's many one-event part files into a single parquet per day.
        The append-only log grows one file per ingest; with weeks of traffic every
        full scan (snapshot, timeline, replay) pays tens of thousands of file opens.
        Idempotent; new events keep appending as parts until the next compaction."""
        import glob as _glob
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            raise HTTPException(status_code=503, detail="pyarrow not installed")
        base = _log_base(c)
        if not base:
            raise HTTPException(status_code=503, detail="EVENT_LOG_DIR not set")
        days = files_before = files_after = 0
        for d in sorted(_glob.glob(os.path.join(base, "date=*"))):
            parts = sorted(_glob.glob(os.path.join(d, "*.parquet")))
            if len(parts) <= 1:
                continue
            tables = []
            for f in parts:
                try:
                    tables.append(pq.read_table(f))
                except Exception:
                    continue
            if not tables:
                continue
            merged = pa.concat_tables(tables, promote_options="permissive")
            out = os.path.join(d, f"part-compacted-{uuid4().hex}.parquet")
            pq.write_table(merged, out)          # write new file FIRST, then drop parts
            for f in parts:
                try:
                    os.remove(f)
                except OSError:
                    pass
            days += 1
            files_before += len(parts)
            files_after += 1
        getattr(c, "_dash_cache", {}).clear()
        return {"result": {"days_compacted": days, "files_before": files_before,
                           "files_after": files_after}}

    @ops.get("/export/snapshots")
    def list_snapshots() -> dict:
        import json as _json
        d = _snap_dir()
        out = []
        for fn in sorted(os.listdir(d), reverse=True):
            if not _SNAP_NAME.match(fn):
                continue
            p = os.path.join(d, fn)
            meta = {}
            try:
                with open(_snap_meta_path(p), encoding="utf-8") as fh:
                    meta = _json.load(fh)
            except Exception:
                pass
            out.append({"name": fn, "size_bytes": os.path.getsize(p),
                        "created": meta.get("created")
                                   or datetime.fromtimestamp(os.path.getmtime(p), timezone.utc).isoformat(),
                        "stats": meta.get("stats") or {}})
        return {"result": out}

    @ops.post("/export/snapshot")
    def take_snapshot() -> dict:
        """Freeze the full parquet log into one gzipped CSV. Vectorized: a single
        pyarrow dataset read over all part files, arrow-native CSV encode into a
        gzip stream, and columnar stats — the earlier per-file python row loop
        took tens of seconds on production's thousands of small parts."""
        import glob as _glob
        import json as _json
        try:
            import pyarrow as pa
            import pyarrow.compute as pc
            import pyarrow.csv as pacsv
            import pyarrow.parquet as pq
        except ImportError:
            raise HTTPException(status_code=503, detail="pyarrow not installed")
        from .tenancy import current_tenant
        base = _log_base(c)
        files = sorted(_glob.glob(os.path.join(base, "date=*", "*.parquet")))
        now = datetime.now(timezone.utc)
        tenant = current_tenant.get() or "default"
        name = f"{tenant}-{now.strftime('%Y%m%d-%H%M%S')}.csv.gz"
        path = os.path.join(_snap_dir(), name)
        cols = ["ts", "user_id", "event", "content_id", "session_id", "request_id",
                "dwell_seconds", "end_reason", "query_text", "clicked_id",
                "impressions", "survey_answers", "raw"]
        # the log is many tiny one-event part files: opening them dominates, so
        # read in parallel (I/O bound) instead of sequentially
        from concurrent.futures import ThreadPoolExecutor
        def _read(f):
            try:
                return pq.read_table(f)
            except Exception:
                return None
        with ThreadPoolExecutor(max_workers=8) as ex:
            tables = [t for t in ex.map(_read, files) if t is not None]
        if tables:
            # permissive: a column that is all-null in one part file (null type)
            # promotes to the concrete type other parts have, instead of erroring
            table = pa.concat_tables(tables, promote_options="permissive")
            table = table.select([col for col in cols if col in table.column_names])
            for i, fld in enumerate(table.schema):   # all-null everywhere -> string
                if pa.types.is_null(fld.type):
                    table = table.set_column(i, fld.name, pc.cast(table.column(i), pa.string()))
            try:
                table = table.sort_by("ts")
            except Exception:
                pass
        else:
            table = pa.table({col: pa.array([], type=pa.string()) for col in cols})
        with pa.CompressedOutputStream(path, "gzip") as out:
            pacsv.write_csv(table, out)

        ev = table["event"] if "event" in table.column_names else pa.array([], type=pa.string())
        uid = table["user_id"] if "user_id" in table.column_names else pa.array([], type=pa.string())
        sa = table["survey_answers"] if "survey_answers" in table.column_names else None
        view_events = int(pc.sum(pc.starts_with(pc.cast(ev, pa.string()), "CONTENT_VIEW")).as_py() or 0) if len(ev) else 0
        content_views = int(pc.sum(pc.equal(ev, "CONTENT_VIEW_STARTED")).as_py() or 0) if len(ev) else 0
        users = set(u for u in uid.to_pylist() if u) if len(uid) else set()
        responded: set = set()
        if sa is not None and len(sa):
            for u, v in zip(uid.to_pylist(), sa.to_pylist()):
                if u and v and v not in ("{}", "null", ""):
                    responded.add(u)
        stats = {"total_events": table.num_rows, "total_visitors": len(users),
                 "survey_responded_visitors": len(responded),
                 "anonymous_visitors": len(users) - len(responded),
                 "content_views": content_views, "content_view_events": view_events}
        meta = {"created": now.isoformat(), "stats": stats}
        with open(_snap_meta_path(path), "w", encoding="utf-8") as fh:
            _json.dump(meta, fh)
        return {"result": {"name": name, "size_bytes": os.path.getsize(path), **meta}}

    @ops.get("/export/snapshot/{name}")
    def download_snapshot(name: str):
        if not _SNAP_NAME.match(name):
            raise HTTPException(status_code=400, detail="bad snapshot name")
        p = os.path.join(_snap_dir(), name)
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail="snapshot not found")
        from fastapi.responses import FileResponse
        return FileResponse(p, media_type="application/gzip", filename=name)

    @ops.delete("/export/snapshot/{name}")
    def delete_snapshot(name: str) -> dict:
        if not _SNAP_NAME.match(name):
            raise HTTPException(status_code=400, detail="bad snapshot name")
        p = os.path.join(_snap_dir(), name)
        if not os.path.exists(p):
            raise HTTPException(status_code=404, detail="snapshot not found")
        os.remove(p)
        try:
            os.remove(_snap_meta_path(p))
        except OSError:
            pass
        return {"result": {"deleted": name}}

    @ops.get("/export/events")
    def export_events():
        """Download the tenant's durable parquet log (ingested events + served
        impressions + registry) as one tar.gz: the complete raw record, for
        backups without cluster access. Guarded like every ops route."""
        import tarfile
        import tempfile
        from fastapi.responses import FileResponse
        from starlette.background import BackgroundTask
        base = _log_base(c)
        if not base or not os.path.isdir(base):
            raise HTTPException(status_code=404, detail="no event log directory")
        tmp = tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False)
        try:
            with tarfile.open(tmp.name, mode="w:gz") as tar:
                tar.add(base, arcname=os.path.basename(base.rstrip("/")) or "event-log")
        except Exception:
            os.unlink(tmp.name)
            raise
        fname = f"event-log-{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.tar.gz"
        return FileResponse(tmp.name, media_type="application/gzip", filename=fname,
                            background=BackgroundTask(os.unlink, tmp.name))

    @ops.get("/content/spread")
    def content_spread() -> dict:
        """Supply vs demand across SPACE and TIME, for the bias-aware map and year view.

        Coordinates and creation years come off the normalized `Content` (the payload
        normalizer is the only place raw payloads are parsed); demand comes from the
        shared cohort view scan. Reports BOTH catalogue density (items: supply) and
        engagement (views: demand), so 'popular' reads against how much content
        exists there/then."""
        views: dict[str, int] = {}
        for _s, aggs in _cohort_view_aggs(c):
            for cid, a in aggs.items():
                views[cid] = views.get(cid, 0) + (a.visits or 0)

        out_items = []
        n_items = n_geo = n_dated = 0
        for cont in _iter_catalogue(c):
            n_items += 1
            has_geo = cont.lat is not None and cont.lon is not None
            if has_geo:
                n_geo += 1
            if cont.years:
                n_dated += 1
            if has_geo or cont.years:
                out_items.append({"id": cont.id, "title": cont.title or cont.id,
                                  "lat": cont.lat, "lon": cont.lon,
                                  "years": cont.years, "views": views.get(cont.id, 0)})
        return {"result": {"items": out_items,
                           "coverage": {"items": n_items, "with_geo": n_geo, "with_dates": n_dated}}}

    @ops.get("/content/{content_id}")
    def content_detail(content_id: str) -> dict:
        """One item: the parsed Content + its raw Qdrant payload (for the detail card +
        metadata/JSON view)."""
        cont = c.content_store.get([content_id]).get(content_id)
        try:
            payload = c.content_store.raw_payloads([content_id]).get(content_id)
        except Exception:
            payload = None
        if cont is None and not payload:
            raise HTTPException(status_code=404, detail="content not found")
        return {"result": {"content_id": content_id,
                           "content": cont.model_dump() if cont else None,
                           "payload": payload}}

    @ops.get("/policy")
    def policy() -> dict:
        """The ranking policy: mode + bandit theta vs its prior + TRAINING HEALTH
        (per-weight confidence from the posterior, which weights still lack data, verdict)."""
        import math
        from .ranking.bandit import LinearBandit, FEATURE_ORDER
        cfg = c.cfg
        prior_w = {n: getattr(cfg.fusion, n, 0.0) for n in FEATURE_ORDER}
        prior = [prior_w[n] for n in FEATURE_ORDER]
        pol = getattr(c.recommender, "policy", None)
        if pol is None:                                   # static mode: still load a trained file to inspect
            path = getattr(c, "bandit_state_path", None) or os.getenv("BANDIT_STATE_PATH")
            if path and os.path.exists(path):
                import json
                try:
                    with open(path, encoding="utf-8") as fh:
                        pol = LinearBandit.from_dict(json.load(fh))
                except Exception:
                    pol = None
        theta = pol.theta() if pol else prior
        out = {
            "mode": cfg.ranking_mode, "feature_order": list(FEATURE_ORDER),
            "prior": prior, "theta": theta, "trained": bool(pol and pol.n_updates > 0),
            "alpha": cfg.bandit_alpha, "ridge": cfg.bandit_ridge, "explore": cfg.bandit_explore,
        }
        if pol:
            h = pol.health()
            ridge = h["ridge"] or 1.0
            prior_std = 1.0 / math.sqrt(ridge) if ridge > 0 else 1.0
            conf = [max(0.0, min(1.0, 1.0 - s / prior_std)) for s in h["std"]]   # 0 at prior -> 1 confident
            active = [i for i, d in enumerate(h["data"]) if d > 1e-6]
            no_data = [h["feature_order"][i] for i, d in enumerate(h["data"]) if d <= 1e-6]
            mean_conf = sum(conf[i] for i in active) / len(active) if active else 0.0
            n = h["n_updates"]
            verdict = "cold" if n < 20 else ("learning" if mean_conf < 0.5 else "converged")
            out["health"] = {
                "n_updates": n, "confidence": conf, "data": h["data"], "no_data": no_data,
                "mean_confidence": round(mean_conf, 3), "verdict": verdict,
            }
        return {"result": out}

    @ops.get("/metrics")
    def metrics_endpoint() -> dict:
        """Serving counters. Prefers the DURABLE per-tenant log (survives restarts);
        the in-process counters (since this process started) ride along under `session`."""
        base = _log_base(c)
        durable = _durable_metrics(base) if base else None
        sess_recs = metrics["recommends"] or 1
        session = {**metrics,
                   "cold_rate": round(metrics["cold"] / sess_recs, 4),
                   "avg_pool": round(metrics["pool_total"] / sess_recs, 2),
                   "distractor_rate": round(metrics["distractor_placed"] / (metrics["distractor_requested"] or 1), 4)}
        if not durable:
            return {"result": {**session, "source": "session"}}
        recs = durable["recommends"] or 1
        return {"result": {
            "ingests": durable["ingests"], "recommends": durable["recommends"],
            "cold": durable["cold"], "warm": durable["warm"], "distractor_placed": durable["distractor_placed"],
            "cold_rate": round(durable["cold"] / recs, 4),
            "avg_pool": round(durable["items_total"] / recs, 2),
            "distractor_rate": round(durable["distractor_placed"] / recs, 4),
            "source": "durable", "session": session,
        }}

    @ops.get("/served/recent")
    def served_recent(n: int = Query(default=50, ge=1, le=500),
                      user_id: Optional[str] = Query(default=None, description="only this visitor's impressions")) -> dict:
        """Tail of the durable served-impression log (PII -> guarded). Per-tenant log dir."""
        base = _log_base(c)
        if not base:
            return {"result": [], "detail": "EVENT_LOG_DIR not set"}
        try:
            import pyarrow.parquet  # noqa: F401  (presence check; helper imports it)
        except ImportError:
            return {"result": [], "detail": "pyarrow not installed"}
        return {"result": _tail_served(base, n, user_id=user_id)}

    @ops.get("/served/explain")
    def served_explain(request_id: str = Query(..., description="served impression request_id")) -> dict:
        """Explain a past recommendation: the served items, the visitor's current tag
        model, and, per item, the overlap between the visitor's tag affinities and the
        content's tags, plus the stored per-scorer breakdown (the other reasons it was
        ranked). Impressions logged before the breakdown was persisted ('legacy') still
        return live-computed tag overlap; has_breakdown flags that case for the UI."""
        import glob
        import json
        base = _log_base(c)
        if not base:
            return {"result": None, "detail": "EVENT_LOG_DIR not set"}
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return {"result": None, "detail": "pyarrow not installed"}
        rec = None
        for f in glob.glob(os.path.join(base, "served", "**", "*.parquet"), recursive=True):
            try:
                for r in pq.read_table(f).to_pylist():
                    if r.get("request_id") == request_id:
                        rec = r
                        break
            except Exception:
                continue
            if rec is not None:
                break
        if rec is None:
            return {"result": None, "detail": "request_id not found"}

        items = rec.get("items")
        if isinstance(items, str):
            try:
                items = json.loads(items)
            except Exception:
                items = []
        items = items or []

        user_id = rec.get("user_id")
        sig = c.model_store.get_signals(user_id) if user_id else None
        affinity = (getattr(sig, "tag_affinity", None) or {}) if sig else {}
        # The tag scorer matches user affinity to content tags on NORMALIZED keys
        # (casing/whitespace/synonyms), so the overlap shown here must normalize the
        # same way — otherwise "theme_what:Family" (raw tag) would miss
        # "theme_what:family" (affinity). Single source of truth: taxonomy.normalize_key.
        from .taxonomy import normalize_key
        aff_norm = {}
        for k, val in affinity.items():
            aff_norm[normalize_key(k)] = float(val)

        ids = [it.get("id") for it in items if it.get("id")]
        contents = c.content_store.get(ids) if ids else {}
        has_breakdown = any(it.get("breakdown") for it in items)

        out_items = []
        for it in items:
            cid = it.get("id")
            ct = contents.get(cid)
            content_tags = [t.key for t in getattr(ct, "tags", [])] if ct else []
            overlap = [
                {"key": k, "user_weight": round(aff_norm[normalize_key(k)], 3)}
                for k in content_tags if normalize_key(k) in aff_norm
            ]
            overlap.sort(key=lambda o: o["user_weight"], reverse=True)
            out_items.append({
                "id": cid,
                "title": getattr(ct, "title", None) if ct else None,
                "rank": it.get("rank"),
                "role": it.get("role"),
                "relevance_score": it.get("relevance_score"),
                "breakdown": it.get("breakdown") or {},
                "content_tags": content_tags,
                "tag_overlap": overlap,
                "image_url": getattr(ct, "image_url", None) if ct else None,
                "public_url": getattr(ct, "public_url", None) if ct else None,
            })

        # Persona-level view of the same user model (so the UI can speak the same language).
        user_model = None
        if sig is not None:
            from .explain.persona import explain_user
            contents_seen = c.content_store.get(sig.viewed or [])
            user_model = explain_user(sig, contents_seen).model_dump()

        user_tag_affinity = [
            {"key": k, "weight": round(float(v), 3)}
            for k, v in sorted(affinity.items(), key=lambda kv: kv[1], reverse=True)[:15]
        ]

        return {"result": {
            "request_id": request_id,
            "user_id": user_id,
            "ts": rec.get("ts"),
            "strategy": rec.get("strategy"),
            "filter": rec.get("filter"),
            "cold_start": rec.get("cold_start"),
            "ranking": rec.get("ranking"),
            "distractor_id": rec.get("distractor_id"),
            "has_breakdown": bool(has_breakdown),
            "user_model": user_model,
            "user_tag_affinity": user_tag_affinity,
            "items": out_items,
        }}

    @write.post("/recommend/preview")
    def recommend_preview(
        spec: PreviewSpec,
        filter: Optional[str] = Query(default=None),
        near_lat: Optional[float] = Query(default=None),
        near_lon: Optional[float] = Query(default=None),
        geo_radius_m: Optional[float] = Query(default=None, ge=0),
        include_content: bool = Query(default=True, description="false = compact (ids/scores only)"),
    ) -> dict:
        """Recommend from a hand-authored user model (no events). For manual /
        programmatic testing + LLM evaluation."""
        signals = build_preview_signals(spec, c.content_store)
        near = (near_lat, near_lon) if near_lat is not None and near_lon is not None else None
        rec = c.recommender.recommend_for_signals(signals, filter=filter, near=near, geo_radius_m=geo_radius_m)
        items = rec.items[:spec.limit] if spec.limit else rec.items
        out = rec.model_dump()
        out["filter"] = filter
        out["items"] = _dump_items(items, include_content)
        out["user_model"] = signals.model_dump()
        request_id = uuid4().hex
        out["request_id"] = request_id
        c.event_log.log_served(_served_record(request_id, signals.user_id, out["items"], out, filter))
        return {"result": out}

    # ----- runtime config (settings page) ---------------------------------- #

    def _apply_cfg(new_cfg) -> None:
        """Mutate the live cfg in place so the recommender/updater (which hold the
        same object) see the change, then rebuild the bandit policy."""
        from .contracts.config import RecConfig
        from .composition import _build_policy
        for name in RecConfig.model_fields:
            setattr(c.cfg, name, getattr(new_cfg, name))
        c.recommender.policy = _build_policy(c.cfg)

    @admin.post("/replay-events")
    def replay_events() -> dict:
        """Rebuild THIS tenant's Redis state (event buffers + visitor models) from its
        durable parquet event log — disaster recovery after a redis wipe, without
        cluster access. Re-feeds every logged event through the same path /api/ingest
        uses. Idempotent: the buffer dedupes identical events, so re-running is safe."""
        import json as _json
        base = _log_base(c)
        if not base or not os.path.isdir(base):
            raise HTTPException(status_code=404, detail="no event log directory")
        try:
            import pyarrow.parquet as pq
        except ImportError:
            raise HTTPException(status_code=503, detail="pyarrow not installed")
        from .contracts.enums import EndReason
        import glob as _glob

        files = sorted(_glob.glob(os.path.join(base, "date=*", "*.parquet")))
        events, skipped = [], 0
        for f in files:
            try:
                rows = pq.read_table(f).to_pylist()
            except Exception:
                skipped += 1
                continue
            for r in rows:
                try:
                    events.append(InteractionEvent(
                        user_id=r["user_id"], event=r["event"],
                        ts=datetime.fromisoformat(r["ts"]) if r.get("ts") else None,
                        session_id=r.get("session_id"), request_id=r.get("request_id"),
                        content_id=r.get("content_id"), dwell_seconds=r.get("dwell_seconds"),
                        end_reason=EndReason(r["end_reason"]) if r.get("end_reason") else None,
                        query_text=r.get("query_text"), clicked_id=r.get("clicked_id"),
                        impressions=_json.loads(r.get("impressions") or "[]"),
                        survey_answers=_json.loads(r.get("survey_answers") or "{}"),
                        raw=_json.loads(r.get("raw") or "{}"),
                    ))
                except Exception:
                    skipped += 1
        events.sort(key=lambda e: e.ts or datetime(1970, 1, 1, tzinfo=timezone.utc))
        users: set[str] = set()
        for e in events:
            c.event_buffer.append(e)  # type: ignore[attr-defined]
            users.add(e.user_id)
        now = datetime.now(timezone.utc)
        for uid in users:
            c.updater.refresh(uid, c.event_buffer, now=now,
                              demographics=c.demographics.get_demographics(uid))
        getattr(c, "_dash_cache", {}).clear()   # everything just changed
        return {"result": {"files": len(files), "events": len(events),
                           "visitors": len(users), "skipped": skipped}}

    @admin.get("/config")
    def get_config() -> dict:
        """Current effective RecConfig (baseline + any runtime override)."""
        return {"result": c.cfg.model_dump()}

    @admin.put("/config")
    def put_config(patch: dict = Body(...)) -> dict:
        """Apply a (partial) RecConfig patch: validate -> apply live -> persist to
        the override store. Survives until the override store is cleared/flushed."""
        from .contracts.config import RecConfig
        from .adapters.config_store import deep_merge
        merged = deep_merge(c.cfg.model_dump(), patch)
        try:
            new_cfg = RecConfig.model_validate(merged)   # type/range validation
        except Exception as e:  # noqa: BLE001
            raise HTTPException(status_code=422, detail=f"invalid config: {e}")
        _apply_cfg(new_cfg)
        c.config_store.set(c.cfg.model_dump())
        return {"result": c.cfg.model_dump(), "status": "applied"}

    @admin.post("/config/reset")
    def reset_config() -> dict:
        """Drop the runtime override -> revert to the env/default baseline."""
        from .composition import _build_config
        c.config_store.clear()
        _apply_cfg(_build_config())
        return {"result": c.cfg.model_dump(), "status": "reset_to_baseline"}

    # ----- evaluation: synthetic visitors -> recommendation behaviour ------ #

    @evalr.get("/personas")
    def eval_personas() -> dict:
        """Built-in synthetic personas, each grounded in this collection's real tag vocab."""
        from .evaluation import BUILTIN_PERSONAS, persona_to_spec
        vocab = _safe_vocab(c)
        out = [{**p, "spec": persona_to_spec(p, vocab)} for p in BUILTIN_PERSONAS]
        return {"result": out}

    @evalr.get("/vocab")
    def eval_vocab() -> dict:
        """The tenant collection's tag vocabulary (facets -> labels) for building personas."""
        return {"result": _safe_vocab(c)}

    @evalr.post("/generate")
    def eval_generate(body: dict = Body(...)) -> dict:
        """Prompt -> persona, matched against the real tag vocabulary (deterministic)."""
        from .evaluation import generate_persona
        return {"result": generate_persona(str(body.get("prompt", "")), _safe_vocab(c))}

    @evalr.post("/run")
    def eval_run(body: EvalRun) -> dict:
        """Score a synthetic persona across scenarios and report behaviour metrics."""
        from .evaluation import list_metrics
        if body.user_id:                     # probe with a REAL visitor's live model
            signals = c.model_store.get_signals(body.user_id)
            if signals is None:
                raise HTTPException(status_code=404, detail=f"no model for visitor {body.user_id}")
        else:
            signals = build_preview_signals(body.spec, c.content_store)
        scenarios = body.scenarios or [{"name": "open"}]
        if body.cold:
            scenarios = scenarios + [{"name": "cold-start", "_cold": True}]
        results = []
        for sc in scenarios:
            sig = UserSignals(user_id="eval") if sc.get("_cold") else signals
            near = ((sc["near_lat"], sc["near_lon"])
                    if sc.get("near_lat") is not None and sc.get("near_lon") is not None else None)
            rec = c.recommender.recommend_for_signals(
                sig, filter=sc.get("filter"), near=near, geo_radius_m=sc.get("geo_radius_m"))
            items = _dump_items(rec.items, include_content=True)
            ids = [it["id"] for it in items]
            vectors = c.content_store.get_vectors(ids)
            try:
                payloads = c.content_store.raw_payloads(ids)     # media/meta for item detail cards
            except Exception:
                payloads = {}
            for it in items:
                it["payload"] = payloads.get(it["id"])
            results.append({"name": sc.get("name", "scenario"), "filter": sc.get("filter"),
                            "strategy": rec.strategy, "items": items,
                            "metrics": list_metrics(items, vectors, rec.strategy),
                            "diagnostics": rec.diagnostics})
        return {"result": {"user_model": signals.model_dump(), "scenarios": results}}

    for sub in (serving, write, usermodel_routes, ops, admin, evalr):
        router.include_router(sub)
    return router


def _safe_vocab(c) -> dict:
    try:
        return c.content_store.vocab()
    except Exception:
        return {"facets": {}, "tags": [], "counts": {}}


class TenantIn(BaseModel):
    tenant_id: str
    collection: Optional[str] = None
    redis_prefix: Optional[str] = None
    bandit_state_path: Optional[str] = None
    cluster_model_path: Optional[str] = None
    config_overrides: dict = Field(default_factory=dict)
    api_keys: list[str] = Field(default_factory=list)        # operator-supplied plaintext keys (hashed before storage)
    generate_api_key: bool = False                           # server mints a key, returns it ONCE, stores only its hash
    replace_keys: bool = False                               # rotate/clear: drop stored keys, keep only what this call sets


def make_tenant_admin_router(manager) -> APIRouter:
    """Control-plane: runtime tenant management (list / create / delete) WITHOUT a redeploy.
    A separate router so it stays out of the app-facing serving surface. INGEST_API_KEY-guarded.
    Note: this registers the tenant SLICE; its content must still be ingested into the tenant's
    Qdrant collection (content-engine) separately."""
    router = APIRouter(prefix="/api/tenants", tags=["tenants"], dependencies=[Depends(_require_api_key)])

    @router.get("")
    def list_tenants() -> dict:
        items = manager.list_tenants()
        qc = getattr(manager, "qdrant_client", None)
        if qc is not None:
            for t in items:                                  # best-effort: show content count
                col = t.get("collection")
                try:
                    t["content_count"] = qc.count(col).count if col else None
                except Exception:
                    t["content_count"] = None
        return {"result": items}

    @router.post("")
    def upsert_tenant(t: TenantIn) -> dict:
        import secrets
        spec = t.model_dump(exclude={"generate_api_key"})
        generated = None
        if t.generate_api_key:
            generated = secrets.token_urlsafe(32)
            spec.setdefault("api_keys", []).append(generated)
        manager.upsert_tenant(spec)                           # hashes keys; never persists plaintext
        # never echo stored keys back; surface a freshly minted key ONCE
        safe = {k: v for k, v in spec.items() if k not in ("api_keys", "api_key_hashes")}
        resp = {"result": safe, "status": "saved"}
        if generated:
            resp["api_key"] = generated
            resp["note"] = "store this key now: only its hash is kept, it cannot be retrieved later"
        return resp

    @router.delete("/{tenant_id}")
    def delete_tenant(tenant_id: str) -> dict:
        manager.delete_tenant(tenant_id)
        return {"status": "deleted", "tenant_id": tenant_id}

    return router


def create_app(components: Optional[Components] = None) -> FastAPI:
    """Build the FastAPI app. Pass a fixed `Components` for single-tenant/tests;
    omit it for the multi-tenant server (a `ComponentManager` + tenant middleware
    resolve a per-tenant `Components` from the `X-Tenant-Id` header). Mounts the
    `/api` router, the search router, health, and the inspector dashboard."""
    from fastapi.middleware.cors import CORSMiddleware

    # Swagger docs left on in prod (operator request). Note: /docs + /openapi.json
    # expose the full API map publicly until the ingress is fronted with auth.
    app = FastAPI(title="AI-Engine Recsys")
    # browser test UIs (ui4testing) call /api/* directly; allow cross-origin in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("AI_ENGINE_CORS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Multi-tenancy: a fixed Components (tests) serves a single tenant; otherwise build a
    # ComponentManager and resolve the tenant per request from the X-Tenant-Id header.
    from .tenancy import TenantProxy, TenantASGIMiddleware
    manager = None
    if components is not None:
        c_or_proxy = components                          # tests / single fixed tenant
    else:
        from .composition import ComponentManager
        manager = ComponentManager()
        c_or_proxy = TenantProxy(manager)
        # key_resolver -> derive tenant from a per-tenant key (trust boundary, not the raw header)
        app.add_middleware(TenantASGIMiddleware, key_resolver=manager.tenant_for_key)

    app.include_router(make_router(c_or_proxy))
    if manager is not None:
        app.include_router(make_tenant_admin_router(manager))

    # Qdrant search surface (ported from legacy ai-engine-api service.py).
    # Guarded so the recsys API still boots if the search stack (embedding model
    # download / Qdrant connectivity) fails at startup.
    try:
        from ai_engine.search.api import make_search_router
        app.include_router(make_search_router())
        logger.info("Search router mounted (/api/search*)")
    except Exception as e:  # noqa: BLE001
        logger.warning(f"Search router NOT mounted: {e}")

    @app.get("/health", tags=["Health"])
    def health() -> dict:
        # APP_VERSION is set from the deployed image tag via the Flux imagepolicy
        # setter (see deployment.yaml), so this reports the running release.
        return {"status": "ok", "version": os.getenv("APP_VERSION", "unknown")}

    @app.get("/dashboard", tags=["Control panel"])
    def dashboard():
        # the holistic control panel (inspect + admin: tenants, config). Served at /dashboard.
        # Inert HTML shell, no secrets: the operator pastes the API key in the page; its JS
        # sends it as X-API-Key on the guarded calls, so no data is reachable without the key.
        from fastapi.responses import HTMLResponse
        path = os.path.join(os.path.dirname(__file__), "static", "dashboard.html")
        if not os.path.exists(path):
            return HTMLResponse("<h1>dashboard.html missing</h1>", status_code=404)
        with open(path, encoding="utf-8") as fh:
            return HTMLResponse(fh.read())

    # static assets for the dashboard (favicons / icons). Inert public files.
    try:
        from fastapi.staticfiles import StaticFiles
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        app.mount("/static", StaticFiles(directory=static_dir), name="static")

        @app.get("/favicon.ico", include_in_schema=False)
        def favicon():
            from fastapi.responses import FileResponse, Response
            ico = os.path.join(static_dir, "favicon", "favicon.ico")
            return FileResponse(ico) if os.path.exists(ico) else Response(status_code=404)
    except Exception as e:  # noqa: BLE001
        logger.warning(f"static mount failed: {e}")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.recsys.api:app", host="0.0.0.0", port=8001, reload=True)
