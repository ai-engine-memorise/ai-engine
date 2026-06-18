"""Evaluation tool: synthetic visitors -> recommendation behaviour.

Lets a domain expert see HOW the engine recommends, without real traffic:
  - a built-in library of museum-grounded personas (Falk 2009 visit identities), and
  - on-command personas generated from a free-text prompt,
each turned into a `PreviewSpec` (tag affinity + demographics + liked items) that the
existing `/api/recommend/preview` path scores. We then run a persona across SCENARIOS
(open / location filter / geo / cold) and compute behaviour METRICS so the curators can
read what the ranker is doing.

Personas are grounded in the tenant's REAL tag vocabulary (`ContentStore.vocab()`), so they
never reference tags that don't exist in that collection.
"""
from __future__ import annotations
import re
from typing import Optional

from .ranking.scorers import cosine

# Built-in synthetic visitors. Each is defined by free-text SEEDS (matched against the
# collection's real tags at run time) + demographics, so they work on any collection.
BUILTIN_PERSONAS = [
    {"key": "explorer", "name": "Explorer", "type": "Explorer",
     "description": "Curiosity-led generalist — broad interests, samples widely.",
     "seeds": ["resistance", "liberation", "everyday life", "biography", "children"],
     "demographics": {"age_group": "25_34"}},
    {"key": "facilitator", "name": "Facilitator", "type": "Facilitator",
     "description": "Visiting for someone else (family / education) — gentle narrative entry points.",
     "seeds": ["children", "family", "education", "everyday life", "biography"],
     "demographics": {"age_group": "35_44"}},
    {"key": "experience_seeker", "name": "Experience-Seeker", "type": "Experience-Seeker",
     "description": "Place- and icon-driven — the key sites and landmark stories.",
     "seeds": ["barrack", "watchtower", "entrance", "transport", "deportation"],
     "demographics": {"age_group": "18_24"}},
    {"key": "professional", "name": "Hobbyist / Professional", "type": "Hobbyist",
     "description": "Deep, focused interest in a theme — wants substance and detail.",
     "seeds": ["forced labor", "anti-jewish measures", "persecution", "administration", "resistance"],
     "demographics": {"age_group": "45_54"}},
    {"key": "recharger", "name": "Recharger", "type": "Recharger",
     "description": "Reflective, contemplative — memorial and commemoration themes.",
     "seeds": ["memorial", "commemoration", "remembrance", "liberation", "loss"],
     "demographics": {"age_group": "65_plus"}},
]

_WORD = re.compile(r"[a-z0-9]+")


def _tokens(s: str) -> list[str]:
    return [w for w in _WORD.findall((s or "").lower()) if len(w) > 2]


def match_tags(text: str, vocab: dict, *, top: int = 12) -> dict[str, float]:
    """Map free text -> {tag_key: weight in 0..1} using the collection's REAL tag vocabulary.
    A tag scores on token overlap with its label (substring + whole-word), so a persona can
    only ever reference tags that exist in this collection."""
    toks = _tokens(text)
    if not toks:
        return {}
    scored: dict[str, float] = {}
    for key in vocab.get("tags", []):
        label = key.split(":", 1)[-1].lower()
        ltoks = set(_tokens(label))
        substr = sum(1 for t in toks if t in label)
        word = sum(1 for t in toks if t in ltoks)
        s = substr + word
        if s > 0:
            scored[key] = float(s)
    items = sorted(scored.items(), key=lambda kv: kv[1], reverse=True)[:top]
    if not items:
        return {}
    mx = items[0][1]
    return {k.lower(): round(v / mx, 3) for k, v in items}


def persona_to_spec(persona: dict, vocab: dict) -> dict:
    """Turn a persona (built-in or generated) into a PreviewSpec dict against this vocab."""
    spec = persona.get("spec")
    if spec is not None:
        return spec
    return {"tag_affinity": match_tags(" ".join(persona.get("seeds", [])), vocab),
            "like_items": [], "demographics": persona.get("demographics", {})}


def _infer_demographics(prompt: str) -> dict:
    p = (prompt or "").lower()
    d: dict = {}
    if "female" in p or "woman" in p or "women" in p:
        d["gender"] = "female"
    elif "male" in p or re.search(r"\bman\b", p):
        d["gender"] = "male"
    m = re.search(r"\b(\d{2})\b", p)
    if any(w in p for w in ["child", "teen", "young", "student", "school"]):
        d["age_group"] = "18_24"
    elif any(w in p for w in ["elderly", "older", "senior", "survivor", "grand"]):
        d["age_group"] = "65_plus"
    elif m:
        age = int(m.group(1))
        d["age_group"] = ("18_24" if age < 25 else "25_34" if age < 35 else "35_44"
                          if age < 45 else "45_54" if age < 55 else "55_64" if age < 65 else "65_plus")
    return d


def _short_name(prompt: str) -> str:
    words = (prompt or "").strip().split()
    return " ".join(words[:6]) + ("…" if len(words) > 6 else "") or "Custom persona"


def generate_persona(prompt: str, vocab: dict) -> dict:
    """Heuristic prompt -> persona: match the prompt against the real tag vocabulary and
    infer coarse demographics. Deterministic (no LLM dependency); an LLM mapper can replace
    `match_tags`/`_infer_demographics` later behind the same shape."""
    aff = match_tags(prompt, vocab)
    return {
        "key": "prompt", "name": _short_name(prompt), "type": "Custom",
        "description": (prompt or "").strip()[:240],
        "spec": {"tag_affinity": aff, "like_items": [], "demographics": _infer_demographics(prompt)},
        "matched_tags": list(aff.keys()),
    }


def intra_list_diversity(vectors: list[Optional[list]]) -> Optional[float]:
    """Mean pairwise (1 - cosine) over the served items' vectors — higher = more varied."""
    vs = [v for v in vectors if v]
    if len(vs) < 2:
        return None
    tot = cnt = 0.0
    for i in range(len(vs)):
        for j in range(i + 1, len(vs)):
            tot += 1.0 - cosine(vs[i], vs[j])
            cnt += 1
    return round(tot / cnt, 3) if cnt else None


def list_metrics(items: list[dict], vectors: dict, strategy: str) -> dict:
    """Behaviour metrics for one served list (the 'what is the ranker doing' view)."""
    ids = [it["id"] for it in items]
    targets = [it for it in items if it.get("role") != "distractor"]
    scores = [it.get("relevance_score") or 0.0 for it in targets]
    facet_counts: dict[str, int] = {}
    label_set: set[str] = set()
    for it in items:
        for t in ((it.get("content") or {}).get("tags") or []):
            f = t.get("facet", "unknown")
            facet_counts[f] = facet_counts.get(f, 0) + 1
            label_set.add(f"{f}:{t.get('label','')}")
    return {
        "n_items": len(items),
        "n_targets": len(targets),
        "avg_target_score": round(sum(scores) / len(scores), 3) if scores else None,
        "intra_list_diversity": intra_list_diversity([vectors.get(i) for i in ids]),
        "distinct_tags": len(label_set),
        "facet_spread": dict(sorted(facet_counts.items(), key=lambda kv: -kv[1])),
        "distractor_present": any(it.get("role") == "distractor" for it in items),
        "strategy": strategy,
    }
