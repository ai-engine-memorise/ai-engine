"""Offline trainer for the linear contextual bandit.

Joins the two durable Parquet logs the engine writes:

    served/date=*/  -> impressions: (request_id, items[{id, features}])      = CONTEXT/ACTION
    date=*/         -> events:      (request_id, content_id, dwell, reason)   = REWARD

For every served item it forms a sample (x = feature vector, r = realized engagement
strength of the view it produced; 0 if the item was shown but never opened). It fits
a LinearBandit starting from the PRIOR (θ0 = the static fusion weights) and applies one
LinUCB update per sample, then writes the trained state to JSON (BANDIT_STATE_PATH).

    python bandit/train.py --log ./data/eventlog --out ./data/bandit_state.json
    # then serve with:  RECSYS_RANKING_MODE=bandit  BANDIT_STATE_PATH=./data/bandit_state.json

Reward is a proxy: it reuses engagement_strength with a nominal reading-time estimate
(the logs don't carry word_count). Relative ordering of rewards drives the learning.
"""
from __future__ import annotations
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.contracts.enums import EndReason
from ai_engine.recsys.ranking.bandit import LinearBandit, FEATURE_ORDER
from ai_engine.recsys.signals.engagement import estimate_reading_time, engagement_strength

NOMINAL_WORDS = 150   # logs lack word_count; assume a typical short story for the est


def _read_parquet_dir(pattern: str) -> list[dict]:
    import pyarrow.parquet as pq
    rows: list[dict] = []
    for f in glob.glob(pattern, recursive=True):
        rows.extend(pq.read_table(f).to_pylist())
    return rows


def _rewards_by_impression(events: list[dict], cfg: RecConfig) -> dict[tuple, float]:
    """(request_id, content_id) -> engagement strength, from CONTENT_VIEW_* events."""
    est = estimate_reading_time(NOMINAL_WORDS, False, cfg)
    grouped: dict[tuple, dict] = {}
    for e in events:
        rid, cid = e.get("request_id"), e.get("content_id")
        if not rid or not cid:
            continue
        g = grouped.setdefault((rid, cid), {"dwell": None, "reason": None, "starts": 0})
        if e.get("event") == "CONTENT_VIEW_STARTED":
            g["starts"] += 1
        d = e.get("dwell_seconds")
        if d is not None:
            g["dwell"] = max(g["dwell"] or 0.0, float(d))
        if e.get("end_reason"):
            g["reason"] = e["end_reason"]

    out: dict[tuple, float] = {}
    for key, g in grouped.items():
        try:
            reason = EndReason(g["reason"]) if g["reason"] else None
        except ValueError:
            reason = None
        out[key] = engagement_strength(
            dwell_seconds=g["dwell"], est_reading_time=est, end_reason=reason,
            visits=max(g["starts"], 1), survey_rating=None, cfg=cfg,
        )
    return out


def _samples(served: list[dict], rewards: dict[tuple, float]) -> list[tuple]:
    samples: list[tuple] = []
    for row in served:
        rid = row.get("request_id")
        items = row.get("items")
        if isinstance(items, str):
            items = json.loads(items)
        for it in items or []:
            x = it.get("features")
            if not x:
                continue                       # e.g. a distractor with no captured features
            r = rewards.get((rid, it["id"]), 0.0)   # shown-but-not-opened -> 0
            samples.append((x, r))
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default=os.getenv("EVENT_LOG_DIR", "./data/eventlog"),
                    help="EVENT_LOG_DIR base (holds date=*/ events and served/date=*/ impressions)")
    ap.add_argument("--out", default=os.getenv("BANDIT_STATE_PATH", "./data/bandit_state.json"))
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args()

    cfg = RecConfig()
    served = _read_parquet_dir(os.path.join(args.log, "served", "**", "*.parquet"))
    events = _read_parquet_dir(os.path.join(args.log, "date=*", "*.parquet"))
    if not served:
        sys.exit(f"no served impressions under {args.log}/served — serve some traffic first")

    rewards = _rewards_by_impression(events, cfg)
    samples = _samples(served, rewards)

    weights = {name: getattr(cfg.fusion, name, 0.0) for name in FEATURE_ORDER}
    bandit = LinearBandit.with_prior(weights, ridge=args.ridge, alpha=args.alpha)
    theta0 = bandit.theta()
    for x, r in samples:
        bandit.update(x, r)
    theta1 = bandit.theta()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(bandit.to_dict(), fh, indent=2)

    pos = sum(1 for _, r in samples if r > 0)
    print(f"impressions={len(served)}  samples={len(samples)}  rewarded(+)={pos}")
    print("feature      prior_theta  trained_theta")
    for name, a, b in zip(FEATURE_ORDER, theta0, theta1):
        print(f"  {name:<10} {a:+.3f}       {b:+.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
