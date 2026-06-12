from datetime import datetime, timezone

import pytest

from ai_engine.recsys.contracts.models import InteractionEvent
from ai_engine.recsys.adapters.event_log import NullEventLog, ParquetEventLog


def test_null_event_log_is_noop():
    NullEventLog().append([InteractionEvent(user_id="u", event="X", ts=datetime.now(timezone.utc))])


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
