"""Bootstrap the bandit from ANOTHER study's interaction sequences.

Different content, same domain: the learned θ is over ABSTRACT per-scorer features
(tag-match vs semantic vs recency), not items — so weights transfer even though the
content differs, as long as the other study's items can be tagged + embedded.

Temporal replay (no leakage): for each view in a user's session, the feature vector is
computed from the model built on EARLIER views only, and the reward is that view's
engagement strength — exactly the (serve -> reward) pairing the live trainer reconstructs.

Inputs (JSONL):
  --content  one item per line: {"id","title","text","word_count","has_image",
                                 "tags":[{"facet","label","weight"}], "vector":[...]}
  --sessions one user per line:  {"user_id", "views":[{"content_id","dwell_seconds",
                                 "end_reason","visits","ts"}]}     (views in time order)

    python bandit/replay.py --content their_items.jsonl --sessions their_sessions.jsonl \
        --out ./data/bandit_state.json --weight 0.5     # down-weight foreign data

Then serve with RECSYS_RANKING_MODE=bandit BANDIT_STATE_PATH=./data/bandit_state.json, or
pass --out to bandit/train.py's --init to keep learning from live memorial traffic.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.contracts.enums import EndReason
from ai_engine.recsys.contracts.models import Content, InteractionEvent, Tag
from ai_engine.recsys.ranking.bandit import LinearBandit, FEATURE_ORDER, feature_vector
from ai_engine.recsys.recommender import score_features
from ai_engine.recsys.signals.signal_builder import build_user_signals
from ai_engine.recsys.signals.engagement import estimate_reading_time, engagement_strength
from ai_engine.recsys.testing.fakes import FakeContentStore


def _load_content(path: str) -> FakeContentStore:
    contents, vectors = {}, {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            cid = str(d["id"])
            contents[cid] = Content(
                id=cid, title=d.get("title", ""), text=d.get("text", ""),
                word_count=int(d.get("word_count", 0)), has_image=bool(d.get("has_image", False)),
                tags=[Tag(**t) for t in d.get("tags", [])],
                lat=d.get("lat"), lon=d.get("lon"),
            )
            vectors[cid] = [float(x) for x in d.get("vector", [])]
    return FakeContentStore(contents, vectors)


def _ts(v) -> datetime:
    if not v:
        return datetime.now(timezone.utc)
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)


def _view_to_events(user_id: str, v: dict) -> list[InteractionEvent]:
    cid = str(v["content_id"])
    t = _ts(v.get("ts"))
    reason = None
    try:
        reason = EndReason(v["end_reason"]) if v.get("end_reason") else None
    except ValueError:
        reason = None
    return [
        InteractionEvent(user_id=user_id, event="CONTENT_VIEW_STARTED", content_id=cid, ts=t),
        InteractionEvent(user_id=user_id, event="CONTENT_VIEW_ENDED", content_id=cid, ts=t,
                         dwell_seconds=v.get("dwell_seconds"), end_reason=reason),
    ]


def _samples_from_session(user_id: str, views: list[dict], store: FakeContentStore,
                          cfg: RecConfig) -> list[tuple]:
    history: list[InteractionEvent] = []
    samples: list[tuple] = []
    for v in sorted(views, key=lambda x: str(x.get("ts", ""))):
        cid = str(v["content_id"])
        content = store.get([cid]).get(cid)
        vec = store.get_vectors([cid]).get(cid)
        if content is None:
            history += _view_to_events(user_id, v)
            continue
        # model BEFORE this view (no leakage)
        ids = list({e.content_id for e in history if e.content_id})
        signals = build_user_signals(
            user_id=user_id, events=history, now=v and _ts(v.get("ts")),
            contents=store.get(ids), vectors=store.get_vectors(ids), cfg=cfg)
        liked = []
        if signals.positives:
            lv = store.get_vectors(list(signals.positives))
            mx = max(signals.positives.values()) or 1.0
            liked = [(signals.positives[c] / mx, lv[c]) for c in signals.positives if c in lv]
        per = score_features(signals, content, vec, liked, cfg)
        x = feature_vector(per)
        est = estimate_reading_time(content.word_count, content.has_image, cfg)
        reason = None
        try:
            reason = EndReason(v["end_reason"]) if v.get("end_reason") else None
        except ValueError:
            pass
        reward = engagement_strength(dwell_seconds=v.get("dwell_seconds"), est_reading_time=est,
                                     end_reason=reason, visits=int(v.get("visits", 1)),
                                     survey_rating=None, cfg=cfg)
        samples.append((x, reward))
        history += _view_to_events(user_id, v)
    return samples


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--out", default="./data/bandit_state.json")
    ap.add_argument("--init", help="continue from an existing bandit state JSON (else start at the prior)")
    ap.add_argument("--weight", type=float, default=1.0, help="per-sample weight (<1 down-weights foreign data)")
    ap.add_argument("--ridge", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=0.3)
    args = ap.parse_args()

    cfg = RecConfig()
    store = _load_content(args.content)

    if args.init and os.path.exists(args.init):
        with open(args.init, encoding="utf-8") as fh:
            bandit = LinearBandit.from_dict(json.load(fh))
    else:
        weights = {n: getattr(cfg.fusion, n, 0.0) for n in FEATURE_ORDER}
        bandit = LinearBandit.with_prior(weights, ridge=args.ridge, alpha=args.alpha)
    theta0 = bandit.theta()

    n_samples = n_sessions = 0
    with open(args.sessions, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            sess = json.loads(line)
            samples = _samples_from_session(str(sess["user_id"]), sess.get("views", []), store, cfg)
            for x, r in samples:
                bandit.update(x, r, weight=args.weight)
            n_samples += len(samples)
            n_sessions += 1
    theta1 = bandit.theta()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(bandit.to_dict(), fh, indent=2)

    print(f"sessions={n_sessions}  samples={n_samples}  weight={args.weight}")
    print("feature      prior_theta  bootstrapped_theta")
    for name, a, b in zip(FEATURE_ORDER, theta0, theta1):
        print(f"  {name:<10} {a:+.3f}       {b:+.3f}")
    print(f"wrote {args.out}  (serve: RECSYS_RANKING_MODE=bandit BANDIT_STATE_PATH={args.out})")


if __name__ == "__main__":
    main()
