"""Seed a locally running recsys API with synthetic visitors for UI review.

Generates a demographically varied cohort against the built-in fixture world
(content ids 101-103, 201-203, 301-302 — the dev fallback when QDRANT_API_URL
is unset), so every dashboard page has data:

  - identify events -> demographics (Cohort statistics: age/gender/nationality/...)
  - CONTENT_VIEW_STARTED/ENDED pairs -> user models, content stats, engagement outcomes
  - /api/recommend calls with request_id echoed on a follow-up view -> served log (Traffic)

Usage:
    python scripts/seed_fake_data.py [--host http://localhost:8002] [--visitors 60]

The target instance should run with AI_ENGINE_DEV=1 (no API key) or pass --key.
"""
from __future__ import annotations

import argparse
import json
import random
import urllib.request
from datetime import datetime, timedelta, timezone

CONTENT = {
    # cluster -> content ids (fixture world semantic clusters)
    "labor": ["101", "102", "103"],
    "family": ["201", "202", "203"],
    "liberation": ["301", "302"],
}
AGES = ["under_16", "16_18", "18_24", "25_34", "35_44", "45_54", "55_64", "65_74", "75_84"]
GENDERS = ["female", "male", "non_binary", "no_answer"]
NATIONALITIES = ["dutch", "german", "belgian", "british", "french", "other"]
PROVINCES = ["Noord-Holland", "Zuid-Holland", "Utrecht", "Gelderland", "Drenthe", "Groningen"]
CONNECTIONS = ["yes", "no", "unknown"]
END_REASONS = ["next_button", "link", "close_button", "abandon"]


def post(host: str, path: str, payload, key: str | None) -> dict:
    req = urllib.request.Request(
        host + path,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **({"X-API-Key": key} if key else {})},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def get(host: str, path: str, key: str | None) -> dict:
    req = urllib.request.Request(
        host + path, headers={"X-API-Key": key} if key else {}
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read())


def identify(uid: str, ts: datetime, rng: random.Random) -> dict:
    traits = {
        "q:age": rng.choice(AGES),
        "q:gender": rng.choices(GENDERS, weights=[45, 45, 4, 6])[0],
        "q:nationality": rng.choices(NATIONALITIES, weights=[50, 15, 10, 8, 7, 10])[0],
        "q:ww2_connection": rng.choices(CONNECTIONS, weights=[25, 55, 20])[0],
    }
    if traits["q:nationality"] == "dutch":
        traits["q:province"] = rng.choice(PROVINCES)
    return {"type": "identify", "userId": uid,
            "timestamp": ts.isoformat(), "traits": traits}


def view_pair(uid: str, cid: str, ts: datetime, dwell: float, reason: str,
              session: str, request_id: str | None = None) -> list[dict]:
    details = {"reason": reason}
    if request_id:
        details["request_id"] = request_id
    return [
        {"type": "track", "event": "CONTENT_VIEW_STARTED", "userId": uid,
         "timestamp": ts.isoformat(),
         "properties": {"content": {"content_id": cid},
                        "context": {"session_id": session}}},
        {"type": "track", "event": "CONTENT_VIEW_ENDED", "userId": uid,
         "timestamp": (ts + timedelta(seconds=dwell)).isoformat(),
         "properties": {"content": {"content_id": cid},
                        "details": details,
                        "context": {"session_id": session}}},
    ]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:8002")
    ap.add_argument("--visitors", type=int, default=60)
    ap.add_argument("--key", default=None, help="X-API-Key if the instance is guarded")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    now = datetime.now(timezone.utc)

    total_events = 0
    for i in range(args.visitors):
        uid = f"vis_{i:03d}"
        # visits spread over the last 14 days
        t0 = now - timedelta(days=rng.uniform(0, 14), hours=rng.uniform(0, 8))
        batch: list[dict] = [identify(uid, t0, rng)]

        # each visitor leans toward one cluster; cold visitors view little or nothing
        fav = rng.choice(list(CONTENT))
        n_views = rng.choices([0, 1, 3, 6, 10], weights=[15, 20, 30, 25, 10])[0]
        t = t0 + timedelta(minutes=2)
        session = f"s_{i:03d}"
        for _ in range(n_views):
            in_fav = rng.random() < 0.7
            cluster = fav if in_fav else rng.choice(list(CONTENT))
            cid = rng.choice(CONTENT[cluster])
            # engaged on the favourite cluster, weak elsewhere
            if in_fav:
                dwell = rng.uniform(25, 120)
                reason = rng.choices(END_REASONS, weights=[55, 25, 15, 5])[0]
            else:
                dwell = rng.uniform(2, 20)
                reason = rng.choices(END_REASONS, weights=[10, 10, 40, 40])[0]
            batch.extend(view_pair(uid, cid, t, dwell, reason, session))
            t += timedelta(minutes=rng.uniform(1, 6))

        r = post(args.host, "/api/ingest", batch, args.key)
        total_events += r.get("ingested", 0)

        # warm visitors also hit /api/recommend (fills the served log / Traffic page)
        # and echo the request_id on a follow-up view (joins reward to impression)
        if n_views >= 3 and rng.random() < 0.8:
            rec = get(args.host, f"/api/recommend?user_id={uid}&limit=5", args.key)["result"]
            items = rec.get("items") or []
            if items:
                cid = str(items[0]["id"])
                follow = view_pair(uid, cid, t, rng.uniform(20, 90), "next_button",
                                   session, request_id=rec.get("request_id"))
                r = post(args.host, "/api/ingest", follow, args.key)
                total_events += r.get("ingested", 0)

        print(f"{uid}: {n_views} views, cluster={fav}")

    print(f"\nSeeded {args.visitors} visitors, {total_events} events -> {args.host}")


if __name__ == "__main__":
    main()
