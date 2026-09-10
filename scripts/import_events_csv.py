"""Rebuild a local Parquet event log from an events CSV export.

The CSV is the flat row shape `ParquetEventLog` writes (the columns of
`adapters.event_log._row`: user_id, event, ts, session_id, request_id, content_id,
dwell_seconds, end_reason, query_text, clicked_id, impressions, survey_answers, raw),
which is also what the Westerbork export delivers. Rows land in `date=YYYY-MM-DD/`
partitions under the target directory, so a dev instance started with
`EVENT_LOG_DIR=<dir>` reads them exactly like production, and POST
/api/admin/replay-events rebuilds visitor models from them.

Usage:
    python scripts/import_events_csv.py <events.csv[.gz]> <event_log_dir> [--tenant default]
"""
from __future__ import annotations
import argparse
import csv
import gzip
import os
import sys
from uuid import uuid4

COLS = ["user_id", "event", "ts", "session_id", "request_id", "content_id",
        "dwell_seconds", "end_reason", "query_text", "clicked_id",
        "impressions", "survey_answers", "raw"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv")
    ap.add_argument("out_dir")
    ap.add_argument("--tenant", default=None,
                    help="write under <out_dir>/<tenant> (multi-tenant layout); omit for single-tenant")
    args = ap.parse_args()
    import pyarrow as pa
    import pyarrow.parquet as pq

    csv.field_size_limit(sys.maxsize)
    opener = gzip.open if args.csv.endswith(".gz") else open
    base = os.path.join(args.out_dir, args.tenant) if args.tenant else args.out_dir
    by_day: dict[str, list[dict]] = {}
    with opener(args.csv, "rt", encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k in COLS:
                v = r.get(k)
                if v == "":
                    v = None
                row[k] = v
            if row["dwell_seconds"] is not None:
                try:
                    row["dwell_seconds"] = float(row["dwell_seconds"])
                except ValueError:
                    row["dwell_seconds"] = None
            row["impressions"] = row["impressions"] or "[]"
            row["survey_answers"] = row["survey_answers"] or "{}"
            row["raw"] = row["raw"] or "{}"
            day = (row["ts"] or "")[:10] or "unknown"
            by_day.setdefault(day, []).append(row)

    schema = pa.schema([(k, pa.float64() if k == "dwell_seconds" else pa.string()) for k in COLS])
    n = 0
    for day, rows in sorted(by_day.items()):
        d = os.path.join(base, f"date={day}")
        os.makedirs(d, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=schema), os.path.join(d, f"part-{uuid4().hex}.parquet"))
        n += len(rows)
    print(f"wrote {n} events into {len(by_day)} day partitions under {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
