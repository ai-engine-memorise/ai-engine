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

# Built-in synthetic visitors for MEMORISE. Encoded with the EXACT survey fields the engine
# consumes (see survey.py / test_survey.py):
#   pre-survey (demographic onboarding): age_group, gender, nationality, province, personal_connection
#   personalization survey:              q:personalization_theme    -> theme_what
#                                        q:personalization_interest -> theme_how.type_of_stores
# `survey_affinity` turns these into tag-affinity exactly as a live survey would. `seeds` add
# breadth, matched against the collection's REAL tags (so nothing is invented).
#
# Visitors complete DIFFERENT parts of the survey — the set deliberately spans completion
# states (`completion`) so the tool shows how the recommender behaves on partial data:
#   full = both surveys · personalization = themes only · demographics = onboarding only · none = walk-in
BUILTIN_PERSONAS = [
    {"key": "school_student_nl", "name": "Dutch school student", "type": "Student", "completion": "full",
     "description": "Class visit, local (Drenthe). Did both surveys: everyday-life and children's stories.",
     "seeds": ["children", "daily life", "school", "biography"],
     "demographics": {"age_group": "16_18", "province": "Drenthe",
                      "q:personalization_theme": ["Daily Life"]}},
    {"key": "university_student", "name": "University student: resistance", "type": "Student",
     "completion": "personalization",
     "description": "Skipped the demographic onboarding; only picked themes: resistance and personal stories.",
     "seeds": ["resistance", "rescue", "escape", "aid and protection"],
     "demographics": {"q:personalization_theme": ["Resistance"], "q:personalization_interest": ["Personal stories"]}},
    {"key": "researcher_forced_labour", "name": "Researcher: forced labour", "type": "Researcher",
     "completion": "full",
     "description": "Deep, focused interest in forced labour and deportation; completed both surveys.",
     "seeds": ["forced labor", "administration", "transit camp", "deportation", "registration"],
     "demographics": {"age_group": "25_34", "q:personalization_theme": ["Forced Labor", "Deportation"]}},
    {"key": "historian_persecution", "name": "Historian: persecution & policy", "type": "Researcher",
     "completion": "personalization",
     "description": "Theme picks only (deportation, family); no demographics given.",
     "seeds": ["anti-jewish measures", "persecution", "administration", "deportation"],
     "demographics": {"q:personalization_theme": ["Deportation", "Family"]}},
    {"key": "intl_tourist", "name": "International tourist", "type": "Tourist", "completion": "demographics",
     "description": "Did only the demographic onboarding (35-44, from Germany); no theme preferences; place-driven.",
     "seeds": ["barrack", "watchtower", "entrance", "memorial", "transport"],
     "demographics": {"age_group": "35_44", "nationality": "germany"}},
    {"key": "regional_tourist", "name": "Regional day-tripper", "type": "Tourist", "completion": "demographics",
     "description": "Onboarding only: Dutch, 55-64, from Gelderland; no theme picks.",
     "seeds": ["liberation", "memorial", "remembrance", "postwar"],
     "demographics": {"age_group": "55_64", "province": "Gelderland"}},
    {"key": "descendant", "name": "Descendant / personal tie", "type": "Descendant", "completion": "full",
     "description": "Relative of someone held here; personal connection, personal stories and family themes.",
     "seeds": ["biography", "family", "personal stories", "remembrance", "commemoration"],
     "demographics": {"age_group": "65_74", "personal_connection": "yes",
                      "q:personalization_theme": ["Family"], "q:personalization_interest": ["Personal stories"]}},
    {"key": "walk_in", "name": "Walk-in visitor (no survey)", "type": "Tourist", "completion": "none",
     "description": "Took no survey at all: pure cold-start; tests the diverse fallback the engine serves.",
     "seeds": [],
     "demographics": {}},
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
    """Mean pairwise (1 - cosine) over the served items' vectors, higher = more varied."""
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
