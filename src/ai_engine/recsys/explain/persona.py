"""Persona explanation — a pure, evidence-backed reading of the user model.

Grounded in museum-visitor theory so the output is interpretable, not ad-hoc:

- **Falk (2009), *Identity and the Museum Visitor Experience*** — visit-identity types
  (Explorer / Hobbyist / Recharger / Experience-Seeker / Facilitator). Derived from the
  BREADTH of tag affinity (entropy) × engagement DEPTH (dwell/completion) × pace.
- **Pekarik, Doering & Karns (1999), "Satisfying Experiences in Museums"** — experience
  preference (object / cognitive / introspective / social). Derived from which taxonomy
  FACET FAMILY dominates the affinity (medium vs theme vs personal-story/person).
- **Tintarev & Masthoff** — the aim here is SCRUTABILITY: every claim carries its evidence
  (the content ids / signals that produced it) so a visitor could inspect and correct it.

Pure: UserSignals (+ optional content tags for evidence/trajectory) -> PersonaExplanation.
No IO, no LLM. Verbalization is a separate, optional step (explain/verbalize.py).
"""
from __future__ import annotations
import math
from typing import Optional

from ..contracts.models import Content, Interest, PersonaExplanation, UserSignals, VisitorType

# Pekarik experience preference <- taxonomy facet family
_OBJECT_FACETS = ("medium_what",)
_COGNITIVE_FACETS = ("theme_what", "time_when", "place_where")
_INTROSPECTIVE_HINTS = ("personal stories", "personal story", "testimony")


def _split(key: str) -> tuple[str, str]:
    facet, _, label = key.partition(":")          # label may itself contain ':'
    return facet, label


def _norm_entropy(weights: list[float]) -> float:
    """Shannon entropy of a weight distribution, normalized to [0,1]. 0 = one theme
    dominates (narrow), 1 = spread evenly (broad)."""
    ws = [w for w in weights if w > 0]
    s = sum(ws)
    if s <= 0 or len(ws) <= 1:
        return 0.0
    ps = [w / s for w in ws]
    h = -sum(p * math.log(p) for p in ps)
    return h / math.log(len(ps))


def _evidence_ids(facet: str, label: str, ids: list[str], contents: dict[str, Content]) -> list[str]:
    key = f"{facet}:{label}".lower()
    out = []
    for cid in ids:
        c = contents.get(cid)
        if c and any(t.key.lower() == key for t in c.tags):
            out.append(cid)
    return out[:5]


def _interests(affinity: dict[str, float], evidence_ids: list[str],
               contents: dict[str, Content], *, top: int = 5) -> list[Interest]:
    out = []
    for key, w in sorted(affinity.items(), key=lambda kv: kv[1], reverse=True)[:top]:
        facet, label = _split(key)
        out.append(Interest(facet=facet, label=label, weight=round(w, 4),
                            evidence=_evidence_ids(facet, label, evidence_ids, contents)))
    return out


def _engagement_style(b: dict, breadth: float) -> str:
    dwell = b.get("avg_dwell_ratio", 0.0)
    completion = b.get("completion_rate", 0.0)
    n = b.get("n_views", 0)
    if n == 0:
        return "unknown"
    if dwell >= 0.6 and completion >= 0.5:
        return "deep_reader"            # lingers AND finishes
    if dwell >= 0.7 and n <= 3 and breadth < 0.5:
        return "contemplative"          # lingers on a few, narrow, without necessarily finishing
    if completion >= 0.7 and n >= 4:
        return "completionist"
    if dwell < 0.35 and n >= 4:
        return "skimmer"
    return "sampler"


def _experience_preference(affinity: dict[str, float], demographics: dict) -> str:
    fam = {"object": 0.0, "cognitive": 0.0, "introspective": 0.0, "social": 0.0}
    for key, w in affinity.items():
        facet, label = _split(key)
        ll = label.lower()
        if facet.startswith("theme_how") and any(h in ll for h in _INTROSPECTIVE_HINTS):
            fam["introspective"] += w
        elif facet.startswith("person_who"):
            fam["introspective"] += 0.5 * w
        elif facet.startswith(_OBJECT_FACETS):
            fam["object"] += w
        elif facet.startswith(_COGNITIVE_FACETS):
            fam["cognitive"] += w
    pc = str(demographics.get("personal_connection", "")).lower()
    if pc in {"yes", "family", "descendant", "survivor"}:
        fam["introspective"] += 0.5     # a personal WW2 connection -> reflective engagement
    if not any(fam.values()):
        return "unknown"
    return max(fam, key=fam.get)


def _visitor_type(breadth: float, b: dict, introspective_share: float,
                  cognitive_share: float, demographics: dict, *, warm: bool = False) -> VisitorType:
    dwell = b.get("avg_dwell_ratio", 0.0)
    completion = b.get("completion_rate", 0.0)
    n = b.get("n_views", 0)
    # engagement depth; a warm user with affinity but no dwell data still counts as engaged
    # (floor), so the type comes from breadth — NOT from the Facilitator fallback.
    depth = max(dwell, b.get("depth", 0.0), 0.35 if warm else 0.0)
    few = max(0.0, 1.0 - (max(n - 1, 0) / 6.0))         # 1 at a single view -> 0 by ~7 views
    many = min(n / 5.0, 1.0) if n else 0.4
    pc = str(demographics.get("personal_connection", "")).lower()
    social_hint = 0.6 if pc in {"family", "group", "with_family", "social"} else 0.0

    scores = {
        # Falk identity types (interpretable heuristics over breadth × depth × pace).
        # Facilitator needs an ACTUAL social signal — it is never the default winner.
        "Hobbyist": (1.0 - breadth) * depth * (0.4 + 0.6 * cognitive_share) * (0.4 + 0.6 * many),
        "Explorer": breadth * depth,
        "Experience-Seeker": breadth * (1.0 - min(dwell, completion)) * (0.4 + 0.6 * many),
        "Recharger": depth * (1.0 - breadth) * few * (0.4 + 0.6 * introspective_share),
        "Facilitator": social_hint,
    }
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    top, top_v = ranked[0]
    second_v = ranked[1][1] if len(ranked) > 1 else 0.0
    rationale = {
        "Hobbyist": "narrow, deep, repeat engagement with specific themes",
        "Explorer": "broad curiosity across themes with real engagement",
        "Experience-Seeker": "broad but light, skims many without finishing",
        "Recharger": "lingers on a few introspective stories, slow pace",
        "Facilitator": "signals of a social/accompanied visit",
    }[top]
    return VisitorType(type=top, confidence=round(max(top_v - second_v, 0.0), 4),
                       rationale=rationale, scores={k: round(v, 4) for k, v in scores.items()})


def _engaged_themes(signals: UserSignals, contents: dict[str, Content]) -> list[str]:
    """Dominant theme_what label of each engaged item — liked items if any, else all
    viewed (so a browser who never 'liked' still has a measurable thematic breadth)."""
    ids = list(signals.positives) or list(signals.viewed)
    themes = []
    for cid in ids:
        c = contents.get(cid)
        if not c:
            continue
        tw = [t for t in c.tags if t.facet == "theme_what"]
        if tw:
            themes.append(max(tw, key=lambda t: t.weight).label)
    return themes


def explain_user(signals: UserSignals, contents: Optional[dict[str, Content]] = None) -> PersonaExplanation:
    """Structured, evidence-backed persona from the user model. `contents` (tags for the
    user's touched items) enriches interest evidence + the thematic trajectory; optional."""
    contents = contents or {}
    aff = signals.tag_affinity
    # breadth = spread across distinct CONTENT themes (not affinity sub-labels, which make
    # one theme look broad). 1 theme -> 0.0, scaling up to ~1.0 by 4+ distinct themes.
    distinct = len(set(_engaged_themes(signals, contents)))
    if distinct:
        breadth = 0.0 if distinct <= 1 else min((distinct - 1) / 3.0, 1.0)
    else:   # no content tags available -> fall back to affinity entropy
        theme_w = [w for k, w in aff.items() if k.startswith("theme_what")]
        breadth = _norm_entropy(theme_w if theme_w else list(aff.values()))

    total = sum(aff.values()) or 1.0
    introspective_share = sum(
        w for k, w in aff.items() if k.startswith("person_who")) / total
    cognitive_share = sum(
        w for k, w in aff.items() if k.startswith("theme_what")) / total

    interests = _interests(aff, list(signals.positives), contents)
    aversions = _interests(signals.tag_aversion, list(signals.negatives), contents)

    # thematic trajectory: dominant theme_what label per recent view, most-recent first
    trajectory: list[str] = []
    for cid in signals.recent_views:
        c = contents.get(cid)
        if not c:
            continue
        themes = [t for t in c.tags if t.facet == "theme_what"]
        if themes:
            label = max(themes, key=lambda t: t.weight).label
            if not trajectory or trajectory[-1] != label:    # collapse consecutive repeats
                trajectory.append(label)

    return PersonaExplanation(
        user_id=signals.user_id,
        is_cold=signals.is_cold,
        interests=interests,
        aversions=aversions,
        engagement_style=_engagement_style(signals.behavior, breadth),
        experience_preference=_experience_preference(aff, signals.demographics),
        visitor_type=_visitor_type(breadth, signals.behavior, introspective_share,
                                   cognitive_share, signals.demographics, warm=bool(aff)),
        trajectory=trajectory[:6],
        demographics=signals.demographics,
    )
