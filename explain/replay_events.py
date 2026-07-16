"""Rebuild Redis state from the durable parquet event log (disaster recovery).

The event buffer and user models live in Redis; the parquet log under
EVENT_LOG_DIR is the durable record of every ingested event. This replays those
events back into the buffer and rebuilds each visitor's model, exactly like
/api/ingest would (minus bandit updates, which need live impressions).

Idempotent: the buffer is a sorted set keyed by the event's JSON, so replaying
an event that is already there is a no-op. Events older than the buffer's
30-day window are trimmed on append, so only the recent window lands in Redis;
older history stays parquet-only (models only read the window anyway).

Run as a one-off Job with the serving image (same env as the API):

    kubectl -n ai-engine create job --from=cronjob/cluster-train replay-tmp \
      --dry-run=client -o yaml | \
      sed 's/cluster_train.py.*/replay_events.py"]/' | kubectl apply -f -

or locally:

    EVENT_LOG_DIR=/app/logs REDIS_URL=redis://recsys-redis:6379/0 \
        python explain/replay_events.py
"""
from __future__ import annotations
import glob
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import pyarrow.parquet as pq  # noqa: E402

from ai_engine.recsys.composition import build_components  # noqa: E402
from ai_engine.recsys.contracts.enums import EndReason  # noqa: E402
from ai_engine.recsys.contracts.models import InteractionEvent  # noqa: E402

_EPOCH = datetime(1970, 1, 1, tzinfo=timezone.utc)


def _event(r: dict) -> InteractionEvent:
    return InteractionEvent(
        user_id=r["user_id"],
        event=r["event"],
        ts=datetime.fromisoformat(r["ts"]) if r.get("ts") else None,
        session_id=r.get("session_id"),
        request_id=r.get("request_id"),
        content_id=r.get("content_id"),
        dwell_seconds=r.get("dwell_seconds"),
        end_reason=EndReason(r["end_reason"]) if r.get("end_reason") else None,
        query_text=r.get("query_text"),
        clicked_id=r.get("clicked_id"),
        impressions=json.loads(r.get("impressions") or "[]"),
        survey_answers=json.loads(r.get("survey_answers") or "{}"),
        raw=json.loads(r.get("raw") or "{}"),
    )


def _replay_dir(c, tenant_dir: str, label: str) -> None:
    files = sorted(glob.glob(os.path.join(tenant_dir, "date=*", "*.parquet")))
    events: list[InteractionEvent] = []
    skipped = 0
    for f in files:
        try:
            rows = pq.read_table(f).to_pylist()
        except Exception as exc:
            print(f"[{label}] unreadable {f}: {exc}")
            continue
        for r in rows:
            try:
                events.append(_event(r))
            except Exception as exc:
                skipped += 1
                print(f"[{label}] skip row ({exc})")

    events.sort(key=lambda e: e.ts or _EPOCH)
    users: dict[str, int] = {}
    for e in events:
        c.event_buffer.append(e)  # type: ignore[attr-defined]
        users[e.user_id] = users.get(e.user_id, 0) + 1

    now = datetime.now(timezone.utc)
    for uid in users:
        demographics = c.demographics.get_demographics(uid)
        c.updater.refresh(uid, c.event_buffer, now=now, demographics=demographics)

    print(f"[{label}] replayed {len(events)} events for {len(users)} visitors "
          f"from {len(files)} files ({skipped} rows skipped)")


def main() -> None:
    base = os.getenv("EVENT_LOG_DIR")
    if not base:
        raise SystemExit("EVENT_LOG_DIR is required (the parquet log root)")

    # The log is partitioned per tenant (<base>/<tenant>/date=*), and each tenant
    # owns its Redis key prefix, so replay through tenant-scoped components.
    tenants = [d for d in sorted(os.listdir(base))
               if os.path.isdir(os.path.join(base, d)) and d != "served"]
    if tenants:
        from ai_engine.recsys.composition import ComponentManager
        mgr = ComponentManager()
        for t in tenants:
            try:
                comp = mgr.get(t)
            except Exception as exc:
                print(f"[{t}] no tenant components ({exc}), skipped")
                continue
            _replay_dir(comp, os.path.join(base, t), t)
    else:
        # single-tenant layout: date=* directly under the base
        _replay_dir(build_components(), base, "default")


if __name__ == "__main__":
    main()
