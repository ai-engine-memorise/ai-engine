"""FastAPI surface for the recommendation engine.

- POST /api/ingest   : the ingest WEBHOOK. RudderStack POSTs user events here
                          (single object or list). Normalize -> buffer -> rebuild
                          the user model.
- GET  /api/recommend: serve recommendations for a user (reads the user model).
- GET  /api/usermodel: debug — inspect the current user model.

Mount `router` into the main service, or run `app` standalone. With no REDIS_URL /
QDRANT_API_URL set it runs fully in-memory on dev fixtures.
"""
from __future__ import annotations
import logging
import os
import threading
from datetime import datetime, timezone
from typing import Any, Optional, Union
from uuid import uuid4

from fastapi import APIRouter, FastAPI, Body, Query, Header, HTTPException, Depends
from pydantic import BaseModel, Field

from .composition import Components, build_components
from .adapters.rudderstack import normalize_events
from .contracts.models import UserSignals
from .survey import survey_affinity


class PreviewSpec(BaseModel):
    """A hand-authored user model for testing recs without going through events."""
    tag_affinity: dict[str, float] = Field(default_factory=dict)   # {"theme_what:forced labor": 1.0}
    like_items: list[str] = Field(default_factory=list)            # -> taste vector (centroid) + excluded as seen
    demographics: dict = Field(default_factory=dict)               # {"age_group":"25_34","gender":"female",...}
    limit: Optional[int] = None


class EvalRun(BaseModel):
    """Run a synthetic persona across scenarios (module-level so FastAPI reads it as a body)."""
    spec: PreviewSpec = Field(default_factory=PreviewSpec)
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
                   "features": it.get("features", [])} for it in items],
    }


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
        return {"result": {"tenant": current_tenant.get() or "default", "scope": scope}}

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

        ev_rows = [{
            "ts": e.ts.isoformat() if e.ts else None, "event": e.event,
            "content_id": e.content_id, "title": titles.get(e.content_id, ""),
            "dwell_seconds": e.dwell_seconds,
            "end_reason": e.end_reason.value if e.end_reason else None,
            "request_id": e.request_id, "session_id": e.session_id,
        } for e in sorted(events, key=lambda e: e.ts, reverse=True)[:limit]]

        return {"result": {"user_id": user_id, "event_count": len(events),
                           "aggregates": agg_rows, "events": ev_rows}}

    @ops.get("/clusters")
    def clusters() -> dict:
        """The explainable visitor segments (offline-trained). Each cluster is described
        by its top taxonomy tags + a Falk breadth hint. CLUSTER_MODEL_PATH must be set."""
        model = _load_cluster_model(getattr(c, "cluster_model_path", None))
        if not model:
            return {"result": None, "detail": "CLUSTER_MODEL_PATH not set / file missing"}
        return {"result": {"method": model.get("method", "kmeans"), "profiles": model.get("profiles", [])}}

    @ops.get("/content/stats")
    def content_stats(limit: int = Query(default=30, ge=1, le=200)) -> dict:
        """Cohort-wide content engagement: which items are seen / liked / abandoned across
        all visitors, popular themes, and each cluster's content preferences. PII-guarded."""
        sigs = c.model_store.iter_signals() if hasattr(c.model_store, "iter_signals") else []
        views: dict[str, int] = {}
        likes: dict[str, int] = {}
        dislikes: dict[str, int] = {}
        theme: dict[str, float] = {}
        for s in sigs:
            for cid in s.viewed:
                views[cid] = views.get(cid, 0) + 1
            for cid in s.positives:
                likes[cid] = likes.get(cid, 0) + 1
            for cid in s.negatives:
                dislikes[cid] = dislikes.get(cid, 0) + 1
            for k, w in s.tag_affinity.items():
                theme[k] = theme.get(k, 0.0) + w

        all_cids = list(set(views) | set(likes) | set(dislikes))
        titles = {k: v.title for k, v in c.content_store.get(all_cids).items()} if all_cids else {}
        lbl = lambda k: k.split(":", 1)[1] if ":" in k else k

        content = [{
            "content_id": cid, "title": titles.get(cid, ""),
            "views": views.get(cid, 0), "likes": likes.get(cid, 0), "dislikes": dislikes.get(cid, 0),
            "like_rate": round(likes.get(cid, 0) / views[cid], 3) if views.get(cid) else 0.0,
        } for cid in all_cids]
        content.sort(key=lambda r: (r["views"], r["likes"]), reverse=True)

        themes = sorted(([lbl(k), round(w, 3)] for k, w in theme.items()),
                        key=lambda kv: kv[1], reverse=True)[:15]

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
    def served_recent(n: int = Query(default=50, ge=1, le=500)) -> dict:
        """Tail of the durable served-impression log (PII -> guarded). Per-tenant log dir."""
        import glob
        import json
        base = _log_base(c)
        if not base:
            return {"result": [], "detail": "EVENT_LOG_DIR not set"}
        try:
            import pyarrow.parquet as pq
        except ImportError:
            return {"result": [], "detail": "pyarrow not installed"}
        rows: list = []
        for f in glob.glob(os.path.join(base, "served", "**", "*.parquet"), recursive=True):
            try:
                rows.extend(pq.read_table(f).to_pylist())
            except Exception:
                continue
        rows.sort(key=lambda r: r.get("ts") or "", reverse=True)
        out = []
        for r in rows[:n]:
            items = r.get("items")
            if isinstance(items, str):
                try:
                    items = json.loads(items)
                except Exception:
                    items = []
            out.append({**r, "items": items})
        return {"result": out}

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
            resp["note"] = "store this key now — only its hash is kept, it cannot be retrieved later"
        return resp

    @router.delete("/{tenant_id}")
    def delete_tenant(tenant_id: str) -> dict:
        manager.delete_tenant(tenant_id)
        return {"status": "deleted", "tenant_id": tenant_id}

    return router


def create_app(components: Optional[Components] = None) -> FastAPI:
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
        return {"status": "ok"}

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

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.recsys.api:app", host="0.0.0.0", port=8001, reload=True)
