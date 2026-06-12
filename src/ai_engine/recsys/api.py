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


def _require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
    """Guard write endpoints with a shared secret (env INGEST_API_KEY).

    If INGEST_API_KEY is unset, requests are allowed (dev) but a warning is logged.
    """
    expected = os.getenv("INGEST_API_KEY")
    if not expected:
        logger.warning("INGEST_API_KEY unset: /api/ingest is UNAUTHENTICATED")
        return
    if x_api_key != expected:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def _dump_items(items, include_content: bool) -> list:
    """Output contract: id / rank / relevance_score / role (+ breakdown, features, optional content).
    The distractor is OUTSIDE the rank ordering -> rank=null; only targets are ranked."""
    out = []
    rank = 0
    for i in items:
        d = i.model_dump()
        is_distractor = d.get("kind") == "distractor"
        if not is_distractor:
            rank += 1
        item = {
            "id": d["content_id"],
            "rank": None if is_distractor else rank,
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


def _load_cluster_model() -> Optional[dict]:
    """Lazily load the offline-trained cluster model from CLUSTER_MODEL_PATH (or None)."""
    import json
    path = os.getenv("CLUSTER_MODEL_PATH")
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
    for e in events:
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
        updated += 1
    if updated:
        path = os.getenv("BANDIT_STATE_PATH")
        if path:
            import json
            tmp = path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(policy.to_dict(), fh)
            os.replace(tmp, path)
    return updated


def make_router(components: Components) -> APIRouter:
    router = APIRouter(prefix="/api", tags=["Recsys"])
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

    @router.post("/ingest", dependencies=[Depends(_require_api_key)])
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

    @router.get("/recommend")
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

    @router.get("/usermodel", dependencies=[Depends(_require_api_key)])
    def usermodel(user_id: str = Query(..., examples=["u1"])) -> dict:
        # debug/inspection endpoint — exposes demographics (PII), so it is guarded
        # by the same INGEST_API_KEY. The serving /recommend stays open for the app.
        sig = c.model_store.get_signals(user_id)
        return {"result": sig.model_dump() if sig else None}

    @router.get("/usermodel/explain", dependencies=[Depends(_require_api_key)])
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
        model = _load_cluster_model()
        if model:                                  # place the visitor in a learned segment
            from .explain.clusters import assign, assign_fuzzy
            out["cluster"] = (assign_fuzzy(sig, model) if model.get("method") == "fcm"
                              else assign(sig, model))
        return {"result": out}

    @router.get("/usermodel/history", dependencies=[Depends(_require_api_key)])
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

    @router.get("/clusters", dependencies=[Depends(_require_api_key)])
    def clusters() -> dict:
        """The explainable visitor segments (offline-trained). Each cluster is described
        by its top taxonomy tags + a Falk breadth hint. CLUSTER_MODEL_PATH must be set."""
        model = _load_cluster_model()
        if not model:
            return {"result": None, "detail": "CLUSTER_MODEL_PATH not set / file missing"}
        return {"result": {"method": model.get("method", "kmeans"), "profiles": model.get("profiles", [])}}

    @router.get("/content/stats", dependencies=[Depends(_require_api_key)])
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
        model = _load_cluster_model()
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

    @router.get("/policy")
    def policy() -> dict:
        """The ranking policy: mode + bandit θ vs its prior (the static fusion weights)."""
        from .ranking.bandit import LinearBandit, FEATURE_ORDER
        cfg = c.cfg
        prior_w = {n: getattr(cfg.fusion, n, 0.0) for n in FEATURE_ORDER}
        prior = [prior_w[n] for n in FEATURE_ORDER]
        theta, trained = prior, False
        path = os.getenv("BANDIT_STATE_PATH")
        if path and os.path.exists(path):
            import json
            try:
                with open(path, encoding="utf-8") as fh:
                    theta = LinearBandit.from_dict(json.load(fh)).theta()
                trained = True
            except Exception:
                pass
        return {"result": {
            "mode": cfg.ranking_mode, "feature_order": list(FEATURE_ORDER),
            "prior": prior, "theta": theta, "trained": trained,
            "alpha": cfg.bandit_alpha, "ridge": cfg.bandit_ridge, "explore": cfg.bandit_explore,
        }}

    @router.get("/metrics")
    def metrics_endpoint() -> dict:
        """In-process serving counters for the inspector (not Prometheus)."""
        recs = metrics["recommends"] or 1
        return {"result": {
            **metrics,
            "cold_rate": round(metrics["cold"] / recs, 4),
            "avg_pool": round(metrics["pool_total"] / recs, 2),
            "distractor_rate": round(metrics["distractor_placed"] / (metrics["distractor_requested"] or 1), 4),
        }}

    @router.get("/served/recent", dependencies=[Depends(_require_api_key)])
    def served_recent(n: int = Query(default=50, ge=1, le=500)) -> dict:
        """Tail of the durable served-impression log (PII -> guarded). Needs EVENT_LOG_DIR."""
        import glob
        import json
        base = os.getenv("EVENT_LOG_DIR")
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

    @router.post("/recommend/preview")
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

    return router


def create_app(components: Optional[Components] = None) -> FastAPI:
    from fastapi.middleware.cors import CORSMiddleware

    app = FastAPI(title="AI-Engine Recsys")
    # browser test UIs (ui4testing) call /api/* directly; allow cross-origin in dev
    app.add_middleware(
        CORSMiddleware,
        allow_origins=os.getenv("AI_ENGINE_CORS", "*").split(","),
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(make_router(components or build_components()))

    @app.get("/health", tags=["Health"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/inspector", tags=["Inspector"])
    def inspector():
        from fastapi.responses import HTMLResponse
        path = os.path.join(os.path.dirname(__file__), "static", "inspector.html")
        if not os.path.exists(path):
            return HTMLResponse("<h1>inspector.html missing</h1>", status_code=404)
        with open(path, encoding="utf-8") as fh:
            return HTMLResponse(fh.read())

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("ai_engine.recsys.api:app", host="0.0.0.0", port=8001, reload=True)
