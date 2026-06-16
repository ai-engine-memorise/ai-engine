"""Held-out evaluation: is the (legacy) data actually good for the bandit?

Splits interaction sessions into train / holdout, trains theta on train (temporal
replay), then on the HELD-OUT sessions measures whether theta.x predicts engagement
better than the static prior. If trained beats prior -> the data carries transferable
signal; if not -> keep the prior (too little data, or it doesn't transfer).

    python bandit/eval.py --content legacy_items.jsonl --sessions legacy_sessions.jsonl

Inputs are the SAME JSONL shapes as bandit/replay.py.
"""
from __future__ import annotations
import argparse
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)
import replay  # reuse the content loader + temporal-replay sample generator

from ai_engine.recsys.contracts.config import RecConfig
from ai_engine.recsys.ranking.bandit import LinearBandit, FEATURE_ORDER, feature_vector


def _dot(theta, x):
    return sum(t * xi for t, xi in zip(theta, x))


def _auc(scores, labels):
    """Probability a positive (engaged) scores above a negative. 0.5 = no better than chance."""
    pos = [s for s, l in zip(scores, labels) if l]
    neg = [s for s, l in zip(scores, labels) if not l]
    if not pos or not neg:
        return float("nan")
    wins = sum((1.0 if p > n else 0.5 if p == n else 0.0) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    vx = sum((a - mx) ** 2 for a in xs) ** 0.5
    vy = sum((b - my) ** 2 for b in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


def _samples(sessions, store, cfg):
    out = []
    for s in sessions:
        out += replay._samples_from_session(str(s["user_id"]), s.get("views", []), store, cfg)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--content", required=True)
    ap.add_argument("--sessions", required=True)
    ap.add_argument("--holdout-every", type=int, default=5, help="every Nth session -> holdout")
    ap.add_argument("--ridge", type=float, default=1.0)
    args = ap.parse_args()

    cfg = RecConfig()
    store = replay._load_content(args.content)
    sessions = [json.loads(l) for l in open(args.sessions, encoding="utf-8") if l.strip()]
    train = [s for i, s in enumerate(sessions) if i % args.holdout_every != 0]
    hold = [s for i, s in enumerate(sessions) if i % args.holdout_every == 0]
    if not train or not hold:
        sys.exit("need enough sessions to split into train + holdout")

    train_s = _samples(train, store, cfg)
    hold_s = _samples(hold, store, cfg)
    if not hold_s:
        sys.exit("no holdout samples (content ids may not match the sessions)")

    weights = {n: getattr(cfg.fusion, n, 0.0) for n in FEATURE_ORDER}
    prior = LinearBandit.with_prior(weights, ridge=args.ridge)
    trained = LinearBandit.with_prior(weights, ridge=args.ridge)
    for x, r in train_s:
        trained.update(x, r)

    labels = [r > 0 for _, r in hold_s]
    rewards = [r for _, r in hold_s]
    th_p, th_t = prior.theta(), trained.theta()
    sc_p = [_dot(th_p, x) for x, _ in hold_s]
    sc_t = [_dot(th_t, x) for x, _ in hold_s]

    auc_p, auc_t = _auc(sc_p, labels), _auc(sc_t, labels)
    cor_p, cor_t = _pearson(sc_p, rewards), _pearson(sc_t, rewards)
    pos = sum(labels)

    print(f"train sessions {len(train)} ({len(train_s)} samples) / holdout {len(hold)} ({len(hold_s)} samples, {pos} engaged)")
    print(f"{'metric':<14}{'prior':>10}{'trained':>10}")
    print(f"{'AUC':<14}{auc_p:>10.3f}{auc_t:>10.3f}")
    print(f"{'corr(score,r)':<14}{cor_p:>10.3f}{cor_t:>10.3f}")
    if auc_t != auc_t or auc_p != auc_p:   # NaN
        verdict = "INCONCLUSIVE - holdout has only one class (need more / more varied data)"
    elif auc_t > auc_p + 0.02:
        verdict = f"USE IT - the data transfers (AUC {auc_p:.3f} -> {auc_t:.3f})"
    elif auc_t < auc_p - 0.02:
        verdict = "KEEP PRIOR - training HURTS on holdout (distribution shift)"
    else:
        verdict = "KEEP PRIOR - no better than the prior (too little signal / no transfer)"
    print("verdict:", verdict)


if __name__ == "__main__":
    main()
