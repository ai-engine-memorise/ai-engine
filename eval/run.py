"""LLM evaluation of recommendations.

For each persona: POST a hand-authored user model to the recsys preview endpoint,
get top-k recommendations, then have Claude judge each recommendation's relevance to
that persona (1-5 + reason). Aggregates mean relevance + precision@k.

This is the Phase-0 eval (expert/LLM judgment) used before real click data exists.

Usage:
    pip install anthropic requests
    export ANTHROPIC_API_KEY=...
    python eval/run.py --api http://localhost:8010 --k 8
    python eval/run.py --model claude-haiku-4-5-20251001        # cheaper judge
"""
from __future__ import annotations
import argparse
import json
import os
import re
import sys

import requests

JUDGE_PROMPT = """You are evaluating a heritage-story recommender for a Holocaust \
memorial (Camp Westerbork). The recommender was given this user persona:

PERSONA: {persona}

It recommended the following {k} stories. For EACH, rate how relevant it is to this \
persona on a 1-5 scale (5 = highly relevant, 1 = irrelevant) and give a one-line reason.
Judge on thematic fit to the persona's stated interest, not on writing quality.

STORIES:
{stories}

Return ONLY a JSON array, one object per story, in order:
[{{"n": 1, "score": 4, "reason": "..."}}, ...]"""


def get_recs(api: str, model_spec: dict, k: int) -> list[dict]:
    body = {**model_spec, "limit": k}
    r = requests.post(f"{api}/api/recommend/preview", json=body, timeout=30)
    r.raise_for_status()
    return r.json()["result"]["items"]


def judge(client, judge_model: str, persona: str, recs: list[dict], k: int) -> list[dict]:
    stories = "\n".join(
        f"{i+1}) {(it.get('content') or {}).get('title','(untitled)')} — "
        f"{((it.get('content') or {}).get('text','') or '')[:500]}"
        for i, it in enumerate(recs)
    )
    prompt = JUDGE_PROMPT.format(persona=persona, k=len(recs), stories=stories)
    msg = client.messages.create(
        model=judge_model, max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    text = msg.content[0].text
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    return json.loads(text)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default=os.getenv("RECSYS_API", "http://localhost:8010"))
    ap.add_argument("--personas", default=os.path.join(os.path.dirname(__file__), "personas.json"))
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--model", default=os.getenv("JUDGE_MODEL", "claude-sonnet-4-6"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "report.json"))
    args = ap.parse_args()

    if not os.getenv("ANTHROPIC_API_KEY"):
        sys.exit("set ANTHROPIC_API_KEY")
    from anthropic import Anthropic
    client = Anthropic()

    personas = json.load(open(args.personas, encoding="utf-8"))
    report, all_scores = [], []

    for p in personas:
        recs = get_recs(args.api, p["model"], args.k)
        if not recs:
            print(f"[{p['name']}] no recs (empty pool)"); continue
        verdicts = judge(client, args.model, p["description"], recs, args.k)
        scores = [v["score"] for v in verdicts]
        mean = sum(scores) / len(scores)
        prec = sum(1 for s in scores if s >= 4) / len(scores)
        all_scores += scores
        report.append({
            "persona": p["name"], "n": len(recs),
            "mean_relevance": round(mean, 2), "precision_at_k": round(prec, 2),
            "items": [
                {"content_id": recs[i]["content_id"],
                 "title": (recs[i].get("content") or {}).get("title", ""),
                 "final_score": round(recs[i]["final_score"], 3),
                 **{kk: verdicts[i][kk] for kk in ("score", "reason")}}
                for i in range(len(verdicts))
            ],
        })
        print(f"[{p['name']:<42}] mean={mean:.2f}  P@{args.k}={prec:.2f}")

    overall = sum(all_scores) / len(all_scores) if all_scores else 0
    print(f"\nOVERALL mean relevance: {overall:.2f}  (judge={args.model})")
    json.dump({"overall_mean_relevance": round(overall, 2), "personas": report},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
