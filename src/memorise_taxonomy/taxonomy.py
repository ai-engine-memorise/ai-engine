"""Single source of truth for MEMORISE tag matching.

The recommender matches a visitor's survey-derived tags against content tags by
string equality on the key ``"facet:label"``. The survey side (ai-engine) and the
content-ingest side (omeka-tools / content-engine) must produce identical keys for
the same concept or the match silently drops to zero. This package is that shared
logic, imported by all three repos so the two sides can never drift.

Two concerns:

* ``normalize_label`` / ``normalize_key`` -- the canonical match form (casing,
  whitespace, accents, separators, spelling, typo aliases).
* ``assign_facet`` / ``to_tag`` -- map a flat Omeka tag string to its facet, driven
  by the authoritative ``data/tags.json`` ("Final tags, adapted from VHA tags").
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

_TAGS_JSON = Path(__file__).with_name("data") / "tags.json"

# --- canonical match form --------------------------------------------------------

# substring spelling unifications (British/US + common in-collection typos), so they
# also fix compounds like "construction forced labour".
_SPELL: list[tuple[str, str]] = [
    ("labour", "labor"),
    ("theatre", "theater"),
    ("tranport", "transport"),
]

# confirmed divergences normalization alone can't reconcile: data-entry typos,
# Dutch<->English duplicates, wording variants. normalized -> normalized.
ALIASES: dict[str, str] = {
    "from: united kindom": "from: united kingdom",
    "barak": "barrack",
    "transitcamp": "transit camp",
    "commandant's house": "commander's house",
    "commanders house": "commander's house",
    "joods leven": "jewish life",
    "daders & omstanders": "perpetrators & bystanders",
    "verzet": "resistance",
    "volksdeutche": "volksdeutsche",
    "death and mortality": "deaths and mortality",
    "death": "deaths and mortality",
    "social dynamics": "social relations",
    "biography": "biographical text",
    "personal story": "personal stories",
    "photography": "photograph",
    "prisoners": "prisoner",
}

_WS = re.compile(r"[\s_\-]+")
_QUOTES = str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})


def _base_normalize(label: str) -> str:
    s = str(label).translate(_QUOTES)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.casefold().strip()
    for a, b in _SPELL:
        s = s.replace(a, b)
    return _WS.sub(" ", s).strip()


def normalize_label(label: str) -> str:
    """Raw label -> canonical match form (base normalization + alias resolution)."""
    s = _base_normalize(label)
    return ALIASES.get(s, s)


def normalize_key(key: str) -> str:
    """Normalize a ``"facet:label"`` key. Splits on the FIRST colon so labels that
    contain a colon (``"From: Spain"``) survive. The facet is a stable machine key --
    only casefolded, not separator-collapsed."""
    facet, sep, label = key.partition(":")
    if not sep:
        return normalize_label(key)
    return f"{facet.strip().casefold()}:{normalize_label(label)}"


# --- AiAR machine tags -----------------------------------------------------------

def _decode_aiar(tag: str) -> str:
    """``AiARLocationBarrack56`` -> ``Barrack 56`` (the app emits camelCase AR tags
    for the same camp areas curators tag by hand)."""
    if tag[:12].lower() == "aiarlocation":
        rest = tag[12:]
        rest = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", rest)
        rest = re.sub(r"(?<=[A-Za-z])(?=\d)", " ", rest)
        return rest.strip()
    return tag


# --- facet taxonomy, loaded from the authoritative tags.json ---------------------

# dimension+group -> facet key, aligned with the survey side (ai_engine.recsys.survey)
# so survey-derived keys and content keys agree.
_GROUP_FACET: dict[tuple[str, str], str] = {
    ("person_who", "Reason for imprisonment"): "person_who.reason_for_imprisonment",
    ("person_who", "Age group"): "person_who.age_group",
    ("person_who", "Gender and age"): "person_who.gender_and_age",
    ("person_who", "Mothertongue"): "person_who.mothertongue",
    ("person_who", "Province (Netherlands)"): "person_who.province_netherlands",
    ("person_who", "City/village & country"): "person_who.city_village_country",
    ("place_where", "Camp areas"): "place_where.camp_areas",
    ("place_where", "Barrack number"): "place_where.camp_areas",
    ("place_where", "Transit destinations (arrived from, deported to)"): "place_where.transit_destinations",
    ("time_when", "Time period"): "time_when.time_period",
    ("theme_how", "AI Engine themes Westerbork"): "theme_how.ai_engine_themes",
    ("theme_how", "Type of stores"): "theme_how.type_of_stores",
    ("theme_how", "Daily Life & Camp Experience"): "theme_how.type_of_stores",
    ("theme_how", "Historical Context & Overview"): "theme_how.type_of_stores",
    ("theme_how", "Reflections & Encounters"): "theme_how.type_of_stores",
}

_PREFIXES: list[tuple[str, str]] = [
    ("from:", "person_who.city_village_country"),
    ("born in:", "person_who.city_village_country"),
    ("lived in:", "person_who.city_village_country"),
    ("live in:", "person_who.city_village_country"),
    ("deported to:", "place_where.transit_destinations"),
    ("arrived from:", "place_where.transit_destinations"),
    ("transported to:", "place_where.transit_destinations"),
    ("transferred to:", "place_where.transit_destinations"),
    ("died in:", "place_where.transit_destinations"),
    ("interned in:", "place_where.transit_destinations"),
    ("imprisoned in:", "place_where.transit_destinations"),
    ("in:", "place_where.transit_destinations"),
]

# patterns on the normalized label (hyphens are already spaces post-normalize).
_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^age \d{2}( \d{2}|\+| \+)?$"), "person_who.age_group"),
    (re.compile(r"^\d{2}( \d{2})?(\+| \+)?( years)?$"), "person_who.age_group"),
    (re.compile(r"\(tone\)$"), "language_how.tone"),
    (re.compile(r"^barrack(s)? \d"), "place_where.camp_areas"),
    (re.compile(r"^camp "), "place_where.transit_destinations"),
]

# category / rollup tags that match by membership, not a literal subtag. The visitor
# side emits the matching rollup key for the complement set (e.g. any non-core
# nationality also emits "international"). See ai_engine.recsys.survey.CORE_COUNTRIES.
_ROLLUP: dict[str, str] = {
    "international": "person_who.city_village_country",
}

DEFAULT_FACET = "theme_what"


@lru_cache(maxsize=1)
def _load_index() -> tuple[dict[str, str], dict[str, str], str]:
    """Build (exact_label->facet, theme_subtag->main_tag, source) from tags.json."""
    tax = json.loads(_TAGS_JSON.read_text("utf-8"))
    exact: dict[str, str] = {}
    sub_to_main: dict[str, str] = {}

    def expand(subtag: str):
        if "/" in subtag:
            head, _, tail = subtag.partition(" ")
            if "/" in head:
                for opt in head.split("/"):
                    yield f"{opt} {tail}".strip()
            for opt in subtag.split("/"):
                yield opt.strip()
        yield subtag

    for dim in tax["dimensions"]:
        dkey = dim["key"]
        for m in dim.get("main_tags", []):
            main = m.get("tag")
            if not main:
                continue
            exact[normalize_label(main)] = "theme_what"
            for st in m.get("subtags", []):
                for variant in expand(st):
                    nv = normalize_label(variant)
                    exact.setdefault(nv, "theme_what")
                    sub_to_main[nv] = main
        for g in dim.get("groups", []):
            facet = _GROUP_FACET.get((dkey, g.get("label", "")))
            if facet is None:
                continue
            if not g.get("subtags") and dkey == "theme_how":
                exact[normalize_label(g["label"])] = facet
            for st in g.get("subtags", []):
                for variant in expand(st):
                    exact.setdefault(normalize_label(variant), facet)
        for st in dim.get("subtags", []):
            facet = {"medium_what": "medium_what"}.get(dkey, dkey)
            exact.setdefault(normalize_label(st), facet)

    return exact, sub_to_main, str(tax.get("source", ""))


@dataclass(frozen=True)
class FacetAssignment:
    facet: str
    label: str          # canonical label written to the content tag (the GRANULAR one)
    source: str         # pattern | prefix | exact | subtag | rollup | default
    rollup: str | None = None  # parent main-tag label to ALSO emit, for coarse matching


def assign_facet(tag: str) -> FacetAssignment:
    """Classify one flat Omeka tag string into ``{facet, label, source, rollup}``.

    For a theme subtag the granular label is kept as ``label`` and the parent main
    tag is returned in ``rollup`` (emit both via :func:`to_tags`) so richness is
    preserved while still matching the main-level theme the survey asks about."""
    exact, sub_to_main, _ = _load_index()
    norm = normalize_label(_decode_aiar(str(tag)))

    if norm in sub_to_main:
        main = normalize_label(sub_to_main[norm])
        # keep the granular subtag; main is an additive rollup (unless it IS the main)
        return FacetAssignment("theme_what", norm, "subtag", rollup=None if norm == main else main)
    if norm in _ROLLUP:
        return FacetAssignment(_ROLLUP[norm], norm, "rollup")
    if norm in exact:
        return FacetAssignment(exact[norm], norm, "exact")
    for pat, facet in _PATTERNS:
        if pat.search(norm):
            return FacetAssignment(facet, norm, "pattern")
    for prefix, facet in sorted(_PREFIXES, key=lambda p: -len(p[0])):
        if norm.startswith(prefix):
            return FacetAssignment(facet, norm, "prefix")
    return FacetAssignment(DEFAULT_FACET, norm, "default")


def to_tag(tag: str, *, weight: float = 1.0) -> dict:
    """Flat Omeka tag string -> the single PRIMARY (granular) structured tag dict.
    Use :func:`to_tags` to also get the main-tag rollup for theme subtags."""
    a = assign_facet(tag)
    return {"facet": a.facet, "label": a.label, "weight": weight}


def to_tags(tag: str, *, weight: float = 1.0, rollup_weight: float | None = None) -> list[dict]:
    """Flat Omeka tag string -> structured tag dict(s).

    Returns the granular tag plus, for a theme subtag, an additional rollup tag for
    its parent main theme so coarse survey answers still match without losing the
    fine-grained label. ``rollup_weight`` defaults to ``weight``."""
    a = assign_facet(tag)
    out = [{"facet": a.facet, "label": a.label, "weight": weight}]
    if a.rollup:
        out.append({"facet": "theme_what", "label": a.rollup,
                    "weight": weight if rollup_weight is None else rollup_weight})
    return out


def review_vocab(tags) -> list[FacetAssignment]:
    """Classify an iterable of flat tag strings; for auditing the mapping."""
    return [assign_facet(t) for t in tags]
