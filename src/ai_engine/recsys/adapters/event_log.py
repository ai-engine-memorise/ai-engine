"""Durable, append-only event log (Parquet) — the training/eval record.

Redis is the ephemeral serving store; this is the permanent log. Every ingested
InteractionEvent is appended to date-partitioned Parquet files:

    <base>/date=YYYY-MM-DD/part-<uuid>.parquet      (Hive-partitioned, append-only)

Read for training with DuckDB / pandas / Spark:
    duckdb.sql("SELECT * FROM read_parquet('<base>/**/*.parquet', hive_partitioning=1)")
"""
from __future__ import annotations
import json
from pathlib import Path
from typing import Sequence
from uuid import uuid4

from ..contracts.models import InteractionEvent


class NullEventLog:
    """No-op (when EVENT_LOG_DIR is unset)."""
    def append(self, events: Sequence[InteractionEvent]) -> None:
        pass


def _row(e: InteractionEvent) -> dict:
    return {
        "user_id": e.user_id,
        "event": e.event,
        "ts": e.ts.isoformat() if e.ts else None,
        "session_id": e.session_id,
        "content_id": e.content_id,
        "dwell_seconds": e.dwell_seconds,
        "end_reason": e.end_reason.value if e.end_reason else None,
        "query_text": e.query_text,
        "clicked_id": e.clicked_id,
        "impressions": json.dumps(e.impressions),       # complex fields -> JSON strings
        "survey_answers": json.dumps(e.survey_answers),
        "raw": json.dumps(e.raw),
    }


class ParquetEventLog:
    def __init__(self, base_dir: str):
        self.base = Path(base_dir)

    def append(self, events: Sequence[InteractionEvent]) -> None:
        if not events:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        by_day: dict[str, list[InteractionEvent]] = {}
        for e in events:
            day = e.ts.date().isoformat() if e.ts else "unknown"
            by_day.setdefault(day, []).append(e)

        for day, evs in by_day.items():
            d = self.base / f"date={day}"
            d.mkdir(parents=True, exist_ok=True)
            table = pa.Table.from_pylist([_row(e) for e in evs])
            pq.write_table(table, d / f"part-{uuid4().hex}.parquet")   # immutable -> append-only
