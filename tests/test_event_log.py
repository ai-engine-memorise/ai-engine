from datetime import datetime, timezone

import pytest

from ai_engine.recsys.contracts.models import InteractionEvent
from ai_engine.recsys.adapters.event_log import NullEventLog, ParquetEventLog


def test_null_event_log_is_noop():
    NullEventLog().append([InteractionEvent(user_id="u", event="X", ts=datetime.now(timezone.utc))])
    NullEventLog().log_served({"request_id": "r", "user_id": "u"})


def test_parquet_log_served_writes_impression(tmp_path):
    pytest.importorskip("pyarrow")
    import json
    import pyarrow.parquet as pq

    rec = {
        "request_id": "req1", "user_id": "u1", "ts": "2026-06-12T10:00:00+00:00",
        "strategy": "warm", "filter": None, "cold_start": False, "distractor_id": "303",
        "items": [{"id": "101", "rank": 1, "role": "target"},
                  {"id": "303", "rank": None, "role": "distractor"}],
    }
    ParquetEventLog(str(tmp_path)).log_served(rec)

    files = list(tmp_path.glob("served/date=2026-06-12/*.parquet"))
    assert files                                       # separate served/ dataset, date-partitioned
    row = pq.read_table(files[0]).to_pylist()[0]
    assert row["request_id"] == "req1" and row["distractor_id"] == "303"
    items = json.loads(row["items"])
    assert items[0]["id"] == "101" and items[1]["role"] == "distractor"


def test_parquet_event_log_writes_partitioned(tmp_path):
    pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq

    ev = InteractionEvent(
        user_id="u1", event="CONTENT_VIEW_ENDED", ts=datetime(2026, 6, 12, 10, tzinfo=timezone.utc),
        content_id="101", dwell_seconds=120, end_reason="next_button",
        survey_answers={"q:age": "25_34"},
    )
    ParquetEventLog(str(tmp_path)).append([ev])

    files = list(tmp_path.glob("date=2026-06-12/*.parquet"))
    assert files                                       # date-partitioned, append-only
    row = pq.read_table(files[0]).to_pylist()[0]
    assert row["user_id"] == "u1" and row["content_id"] == "101"
    assert row["event"] == "CONTENT_VIEW_ENDED" and row["end_reason"] == "next_button"
    import json
    assert json.loads(row["survey_answers"]) == {"q:age": "25_34"}
