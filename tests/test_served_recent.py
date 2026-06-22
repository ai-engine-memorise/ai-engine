"""_tail_served: returns the latest n served rows newest-first, bounded to the
newest day partition(s) instead of reading the whole history."""
import json

import pyarrow as pa
import pyarrow.parquet as pq

from ai_engine.recsys.api import _tail_served


def _write(base, day, ts, rid):
    d = base / "served" / f"date={day}"
    d.mkdir(parents=True, exist_ok=True)
    row = {"request_id": rid, "ts": ts, "items": json.dumps([{"id": rid}])}
    pq.write_table(pa.Table.from_pylist([row]), d / f"part-{rid}.parquet")


def test_tail_newest_first_and_items_parsed(tmp_path):
    # one row per file, mirroring log_served; within-day files are NOT ts-ordered.
    _write(tmp_path, "2026-06-21", "2026-06-21T10:00:00", "old")
    _write(tmp_path, "2026-06-22", "2026-06-22T09:00:00", "a")
    _write(tmp_path, "2026-06-22", "2026-06-22T11:00:00", "c")
    _write(tmp_path, "2026-06-22", "2026-06-22T10:00:00", "b")

    out = _tail_served(str(tmp_path), n=2)

    # newest two, correctly ordered — catches a premature mid-day break that would
    # miss "c" (read after "a"/"b" within the same day).
    assert [r["request_id"] for r in out] == ["c", "b"]
    assert out[0]["items"] == [{"id": "c"}]  # JSON-stringified items decoded


def test_tail_crosses_day_when_needed(tmp_path):
    _write(tmp_path, "2026-06-21", "2026-06-21T10:00:00", "old")
    _write(tmp_path, "2026-06-22", "2026-06-22T09:00:00", "new")
    out = _tail_served(str(tmp_path), n=5)
    assert [r["request_id"] for r in out] == ["new", "old"]


def test_tail_empty(tmp_path):
    assert _tail_served(str(tmp_path), n=10) == []
