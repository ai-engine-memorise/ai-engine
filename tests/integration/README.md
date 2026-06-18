# Integration tests (layer 4)

Replays RudderStack payloads through the **real** adapters — Redis (event buffer +
user-model store) and Qdrant (tagged content) — to prove the whole serving chain:

```
ingest webhook → Redis → user model → Qdrant tag/semantic match → recommendation
```

RudderStack itself is not needed: its webhook destination forwards exactly these
track payloads, so replaying them hits the same contract.

## Run

```bash
# 1. start infra
docker compose -f tests/integration/docker-compose.yml up -d

# 2. (optional) seed by hand to inspect
PYTHONPATH=src python tests/integration/seed_qdrant.py

# 3. run (the test seeds + flushes itself)
RUN_INTEGRATION=1 PYTHONPATH=src python -m pytest tests/integration -q

# 4. tear down
docker compose -f tests/integration/docker-compose.yml down -v
```

Without `RUN_INTEGRATION=1` (or if Qdrant:6333 / Redis:6379 are unreachable) the
tests skip, so the default `pytest tests/` stays infra-free.

## Full smoke (manual, with real RudderStack/Jitsu + app)

The above covers everything except the CDP hop. To smoke that once:
1. Run the API with the recsys router mounted (`ai-engine-api`, port 8000).
2. In RudderStack/Jitsu add a **Webhook** destination → `POST http://<host>:8000/api/ingest`.
3. Fire events from the app (`emit.js` → `rudderanalytics.track(...)`).
4. `GET /api/usermodel` then `/api/recommend` to confirm they landed.
