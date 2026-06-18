"""Composition root: assemble the recsys components from environment.

If REDIS_URL / QDRANT_API_URL are set, use the real adapters; otherwise fall back
to in-memory fakes (with dev fixtures) so the service runs locally with no infra.
This is the ONE place IO backends are chosen — everything else takes ports.
"""
from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional

from .contracts.config import RecConfig
from .contracts.ports import ContentStore, EventSource, UserModelStore, DemographicsProvider
from .recommender import Recommender
from .updater import UserModelUpdater


@dataclass
class Components:
    cfg: RecConfig
    content_store: ContentStore
    event_buffer: EventSource          # also supports .append() (Redis buffer / fake)
    model_store: UserModelStore
    updater: UserModelUpdater
    recommender: Recommender
    demographics: DemographicsProvider
    event_log: object   # durable append-only log (Parquet) — .append(events)
    impressions: object   # short-lived served-feature store (request_id -> features) for online bandit
    config_store: object   # runtime RecConfig override store (Redis / in-memory)
    cluster_model_path: Optional[str] = None   # per-tenant explainable-cluster model
    bandit_state_path: Optional[str] = None    # per-tenant bandit state (online persistence)


def _build_content_store() -> ContentStore:
    url = os.getenv("QDRANT_API_URL")
    if url:
        from qdrant_client import QdrantClient
        from .adapters.qdrant_store import QdrantContentStore
        client = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        return QdrantContentStore(client, os.getenv("COLLECTION_NAME", "omeka-items"))
    # dev fallback: the hand-built fixture world
    from .testing.fakes import FakeContentStore
    from .testing.fixtures import make_contents_and_vectors
    contents, vectors = make_contents_and_vectors()
    return FakeContentStore(contents, vectors)


def _build_stores():
    url = os.getenv("REDIS_URL")
    if url:
        import redis
        from .adapters.redis_store import RedisEventBuffer, RedisUserModelStore, RedisImpressionStore
        client = redis.from_url(url, decode_responses=True)
        return RedisEventBuffer(client), RedisUserModelStore(client), RedisImpressionStore(client)
    from .testing.fakes import FakeEventSource, InMemoryUserModelStore, InMemoryImpressionStore
    return FakeEventSource(), InMemoryUserModelStore(), InMemoryImpressionStore()


def _build_demographics() -> DemographicsProvider:
    if os.getenv("DB_NAME"):
        from .adapters.demographics import PostgresDemographicsProvider
        return PostgresDemographicsProvider()
    from .adapters.demographics import NullDemographicsProvider
    return NullDemographicsProvider()


def _build_event_log():
    d = os.getenv("EVENT_LOG_DIR")
    if d:
        from .adapters.event_log import ParquetEventLog
        return ParquetEventLog(d)
    from .adapters.event_log import NullEventLog
    return NullEventLog()


def _build_config_store():
    url = os.getenv("REDIS_URL")
    if url:
        import redis
        from .adapters.config_store import RedisConfigStore
        return RedisConfigStore(redis.from_url(url, decode_responses=True))
    from .adapters.config_store import InMemoryConfigStore
    return InMemoryConfigStore()


def _build_config() -> RecConfig:
    """RecConfig from defaults, overridden by env so weights/limits are tunable
    WITHOUT a redeploy (a stop-gap until a learned ranker owns the weights).

        RECSYS_W_{SEMANTIC,AFFINITY,TAG,RECENCY,AVERSION}   fusion weights (float)
        RECSYS_MMR_LAMBDA                                    relevance<->diversity
        RECSYS_FINAL_LIMIT                                   items returned (int)
        RECSYS_DISTRACTOR_PROBABILITY                        0..1
    """
    cfg = RecConfig()

    def _f(name: str, cur: float) -> float:
        v = os.getenv(name)
        try:
            return float(v) if v is not None else cur
        except ValueError:
            return cur

    f = cfg.fusion
    f.semantic = _f("RECSYS_W_SEMANTIC", f.semantic)
    f.affinity = _f("RECSYS_W_AFFINITY", f.affinity)
    f.tag = _f("RECSYS_W_TAG", f.tag)
    f.recency = _f("RECSYS_W_RECENCY", f.recency)
    f.aversion = _f("RECSYS_W_AVERSION", f.aversion)
    cfg.mmr_lambda = _f("RECSYS_MMR_LAMBDA", cfg.mmr_lambda)
    cfg.distractor_probability = _f("RECSYS_DISTRACTOR_PROBABILITY", cfg.distractor_probability)
    ds = os.getenv("RECSYS_DISTRACTOR_SLOTS")        # e.g. "2,3" -> distractor at rank 2 or 3
    if ds:
        try:
            cfg.distractor_slots = [int(x) for x in ds.split(",") if x.strip()]
        except ValueError:
            pass
    cfg.ranking_mode = os.getenv("RECSYS_RANKING_MODE", cfg.ranking_mode)   # "static" | "bandit"
    cfg.bandit_alpha = _f("RECSYS_BANDIT_ALPHA", cfg.bandit_alpha)
    cfg.bandit_ridge = _f("RECSYS_BANDIT_RIDGE", cfg.bandit_ridge)
    cfg.bandit_online = os.getenv("RECSYS_BANDIT_ONLINE", "").lower() in {"1", "true", "yes"} or cfg.bandit_online
    fl = os.getenv("RECSYS_FINAL_LIMIT")
    if fl is not None:
        try:
            cfg.final_limit = int(fl)
        except ValueError:
            pass
    return cfg


def _build_policy(cfg: RecConfig):
    """The learned ranking policy (bandit). Only built when ranking_mode == 'bandit'.
    Loads a trained state from BANDIT_STATE_PATH if present, else starts at the prior
    (θ0 = the static fusion weights) so day-one behavior matches weighted fusion."""
    if cfg.ranking_mode != "bandit":
        return None
    from .ranking.bandit import LinearBandit, FEATURE_ORDER
    path = os.getenv("BANDIT_STATE_PATH")
    if path and os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as fh:
            return LinearBandit.from_dict(json.load(fh))
    weights = {name: getattr(cfg.fusion, name, 0.0) for name in FEATURE_ORDER}
    return LinearBandit.with_prior(weights, ridge=cfg.bandit_ridge, alpha=cfg.bandit_alpha)


def _build_policy_for(cfg: RecConfig, path: Optional[str]):
    if cfg.ranking_mode != "bandit":
        return None
    from .ranking.bandit import LinearBandit, FEATURE_ORDER
    if path and os.path.exists(path):
        import json
        with open(path, encoding="utf-8") as fh:
            return LinearBandit.from_dict(json.load(fh))
    weights = {name: getattr(cfg.fusion, name, 0.0) for name in FEATURE_ORDER}
    return LinearBandit.with_prior(weights, ridge=cfg.bandit_ridge, alpha=cfg.bandit_alpha)


def build_components_for(spec, mgr) -> Components:
    """Build a tenant-scoped Components: its own Qdrant collection + Redis key-prefix +
    config / bandit / cluster / event-log, sharing the manager's Redis & Qdrant clients."""
    from .adapters.config_store import deep_merge
    cfg = _build_config()
    if spec.config_overrides:
        try:
            cfg = RecConfig.model_validate(deep_merge(cfg.model_dump(), spec.config_overrides))
        except Exception:
            pass

    qc = mgr.qdrant_client
    if qc is not None:
        from .adapters.qdrant_store import QdrantContentStore
        content_store = QdrantContentStore(qc, spec.collection or os.getenv("COLLECTION_NAME", "omeka-items"))
    else:
        from .testing.fakes import FakeContentStore
        from .testing.fixtures import make_contents_and_vectors
        contents, vectors = make_contents_and_vectors()
        content_store = FakeContentStore(contents, vectors)

    rc = mgr.redis_client
    p = spec.prefix
    if rc is not None:
        from .adapters.redis_store import RedisEventBuffer, RedisUserModelStore, RedisImpressionStore
        from .adapters.config_store import RedisConfigStore
        event_buffer = RedisEventBuffer(rc, key_prefix=f"{p}:evt")
        model_store = RedisUserModelStore(rc, key_prefix=f"{p}:umodel")
        impressions = RedisImpressionStore(rc, key_prefix=f"{p}:imp")
        config_store = RedisConfigStore(rc, key=f"{p}:recsys:config")
    else:
        from .testing.fakes import FakeEventSource, InMemoryUserModelStore, InMemoryImpressionStore
        from .adapters.config_store import InMemoryConfigStore
        event_buffer, model_store, impressions = FakeEventSource(), InMemoryUserModelStore(), InMemoryImpressionStore()
        config_store = InMemoryConfigStore()

    override = config_store.get()
    if override:
        try:
            cfg = RecConfig.model_validate(deep_merge(cfg.model_dump(), override))
        except Exception:
            pass

    base = os.getenv("EVENT_LOG_DIR")
    if base:
        from .adapters.event_log import ParquetEventLog
        event_log = ParquetEventLog(os.path.join(base, spec.tenant_id))   # per-tenant partition
    else:
        from .adapters.event_log import NullEventLog
        event_log = NullEventLog()

    return Components(
        cfg=cfg, content_store=content_store, event_buffer=event_buffer, model_store=model_store,
        updater=UserModelUpdater(content_store, model_store, cfg),
        recommender=Recommender(content_store, model_store, cfg,
                                policy=_build_policy_for(cfg, spec.bandit_state_path)),
        demographics=_build_demographics(), event_log=event_log, impressions=impressions,
        config_store=config_store, cluster_model_path=spec.cluster_model_path,
        bandit_state_path=spec.bandit_state_path,
    )


class ComponentManager:
    """Builds + caches one Components per tenant, over shared Redis & Qdrant clients."""

    def __init__(self, registry=None):
        from .tenancy import TenantRegistry
        self.registry = registry or TenantRegistry.from_env()
        self._cache: dict = {}
        self._redis = None
        self._redis_init = False
        self._qdrant = None
        self._qdrant_init = False
        self._tenant_store = None
        self._ts_init = False

    @property
    def tenant_store(self):
        """Runtime tenant registry (Redis), merged over the TENANTS_PATH baseline."""
        if not self._ts_init:
            self._ts_init = True
            rc = self.redis_client
            if rc is not None:
                from .adapters.tenant_store import RedisTenantStore
                self._tenant_store = RedisTenantStore(rc)
            else:
                from .adapters.tenant_store import InMemoryTenantStore
                self._tenant_store = InMemoryTenantStore()
        return self._tenant_store

    @property
    def redis_client(self):
        if not self._redis_init:
            self._redis_init = True
            url = os.getenv("REDIS_URL")
            if url:
                import redis
                self._redis = redis.from_url(url, decode_responses=True)
        return self._redis

    @property
    def qdrant_client(self):
        if not self._qdrant_init:
            self._qdrant_init = True
            url = os.getenv("QDRANT_API_URL")
            if url:
                from qdrant_client import QdrantClient
                self._qdrant = QdrantClient(url=url, api_key=os.getenv("QDRANT_API_KEY"))
        return self._qdrant

    def _spec(self, tenant_id):
        from .tenancy import TenantSpec
        d = self.tenant_store.get(tenant_id) if tenant_id else None   # runtime store wins
        if d:
            fields = TenantSpec.__dataclass_fields__
            return TenantSpec(**{k: v for k, v in d.items() if k in fields})
        return self.registry.get(tenant_id)

    def get(self, tenant_id) -> Components:
        spec = self._spec(tenant_id)
        if spec.tenant_id not in self._cache:
            self._cache[spec.tenant_id] = build_components_for(spec, self)
        return self._cache[spec.tenant_id]

    def list_tenants(self) -> list[dict]:
        """All tenants: config baseline + runtime-created, runtime wins. For the admin UI."""
        out: dict[str, dict] = {}
        for tid in self.registry.ids():
            s = self.registry.get(tid)
            out[tid] = {"tenant_id": tid, "collection": s.collection, "source": "config",
                        "bandit_state_path": s.bandit_state_path,
                        "cluster_model_path": s.cluster_model_path}
        for d in self.tenant_store.all():
            out[d["tenant_id"]] = {**d, "source": "runtime"}
        return sorted(out.values(), key=lambda t: t["tenant_id"])

    def upsert_tenant(self, spec: dict) -> None:
        self.tenant_store.set(spec)
        self._cache.pop(spec["tenant_id"], None)      # rebuilt with the new spec next request

    def delete_tenant(self, tenant_id: str) -> None:
        self.tenant_store.delete(tenant_id)
        self._cache.pop(tenant_id, None)


def build_components(cfg: Optional[RecConfig] = None) -> Components:
    config_store = _build_config_store()
    cfg = cfg or _build_config()
    # overlay the runtime override (if any) on top of the env/default baseline
    if cfg is not None:
        override = config_store.get()
        if override:
            from .adapters.config_store import deep_merge
            try:
                cfg = RecConfig.model_validate(deep_merge(cfg.model_dump(), override))
            except Exception:  # noqa: BLE001 — a bad override must never block startup
                pass
    content_store = _build_content_store()
    event_buffer, model_store, impressions = _build_stores()
    return Components(
        cfg=cfg,
        content_store=content_store,
        event_buffer=event_buffer,
        model_store=model_store,
        updater=UserModelUpdater(content_store, model_store, cfg),
        recommender=Recommender(content_store, model_store, cfg, policy=_build_policy(cfg)),
        demographics=_build_demographics(),
        event_log=_build_event_log(),
        impressions=impressions,
        config_store=config_store,
    )
