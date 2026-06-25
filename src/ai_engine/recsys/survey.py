"""Survey -> user-model mapping (the persona).

Survey answers become tag-affinity in the SAME taxonomy the content is tagged with,
so the recommender's score_tag does persona<->content tag similarity. Supports the
kwb research survey (survey:kwb:survey: q:age/q:gender/q:nationality + q:personalization_*)
and the demographic onboarding quiz (age_group/gender/nationality).

Personalization questions are explicit preferences: their answer VALUE must be the
canonical taxonomy label (e.g. "Forced Labor") so it matches content tags 1:1.
"""
from __future__ import annotations
from typing import Optional

# Origins with their own country-specific content tags. Any other nationality rolls
# up to the `International` tag (see survey_affinity). Compared casefolded.
CORE_COUNTRIES = {"netherlands", "germany", "poland"}

SURVEY_EVENTS = ("SURVEY_SUBMITTED", "SURVEY_ANSWERED")
# events whose answers/traits seed demographics + persona affinity. IDENTIFY is the
# RudderStack identify call the app sends AFTER the survey (traits = persona).
DEMOGRAPHIC_EVENTS = SURVEY_EVENTS + ("IDENTIFY",)

# age answer value -> person_who.age_group taxonomy label (kwb + quiz buckets)
_AGE = {
    "under_18": "child", "under_16": "child",
    "16_18": "age 16-18", "18_24": "age 18-24", "25_34": "age 25-34",
    "35_44": "age 35-44", "45_54": "age 45-54", "55_64": "age 55-64",
    "65_74": "age 65-74", "75_84": "age 75-84", "85_plus": "age 85+",
    "65_plus": "elderly",
}
_GENDER = {"female": "female", "male": "Male", "non_binary": "non-binary"}

# question_id / trait key -> demographics (covers survey qids AND identify traits)
_AGE_QIDS = ("q:age", "age_group", "age")
_GENDER_QIDS = ("q:gender", "gender")
_NAT_QIDS = ("q:nationality", "nationality", "country")
_PROVINCE_QIDS = ("q:province", "province", "region")   # NL province the visitor is from
_CONN_QIDS = ("q:ww2_connection", "personal_connection")

# personalization questions: answer value = canonical taxonomy label -> facet
_PERSONALIZATION = {
    "q:personalization_theme": "theme_what",
    "q:personalization_interest": "theme_how.type_of_stores",
    "q:personalization_area": "place_where.camp_areas",
}


def _qid_variants(qid: str) -> tuple[str, ...]:
    """Both spellings of a question id: the survey track event sends the `q:`-prefixed
    form (`q:personalization_area`), the post-survey identify traits send the bare form
    (`personalization_area`). Matching either keeps them in the same bucket/facet — the
    demographic qid tuples already list both variants; this does the same for personalization."""
    bare = qid[2:] if qid.startswith("q:") else qid
    return (f"q:{bare}", bare)

# Survey answer values must line up with content tag LABELS or the persona silently
# matches nothing. score_tag already compares case-insensitively, so casing is safe;
# this map fixes the remaining divergences: separator style (slug/underscore/hyphen)
# and spelling/synonyms (British vs American). Lookup key is the separator-normalized
# value lowercased; extend as the survey vocabulary is finalized against the taxonomy.
_LABEL_ALIASES = {
    "forced labour": "Forced Labor",
    "forced labor": "Forced Labor",
    "liberation": "Liberation",
    "deportation": "Deportation",
    "daily life": "Daily Life",
    "personal stories": "Personal stories",
    "resistance": "Resistance",
}


def _canonical_label(v) -> str:
    """Normalize a free survey answer to the content taxonomy label.

    Separators (`_`,`-`) -> spaces, whitespace collapsed, then an explicit alias
    lookup. Already-clean labels (e.g. 'Forced Labor') pass through unchanged so
    case is preserved for display; matching downstream is case-insensitive anyway.
    """
    s = " ".join(str(v).replace("_", " ").replace("-", " ").split())
    return _LABEL_ALIASES.get(s.lower(), s)


def _clean(v):
    """Survey may emit the answer entity id ('a:age:55_64') instead of the value.
    Strip the 'a:<question>:' prefix -> '55_64'. Leaves plain values untouched."""
    if isinstance(v, str) and v.startswith("a:") and v.count(":") >= 2:
        return v.split(":", 2)[2]
    return v


def _vals(answers: dict, *qids: str) -> list:
    """All answer values for the first matching question id (scalar or multi-list)."""
    for qid in qids:
        if qid in answers and answers[qid] is not None:
            v = answers[qid]
            vals = v if isinstance(v, list) else [v]
            return [_clean(x) for x in vals]
    return []


def extract_demographics(answers: dict) -> dict:
    """Raw survey answers -> demographics dict (for storage / inspection)."""
    out = {}
    for field, qids in (("age", _AGE_QIDS), ("gender", _GENDER_QIDS),
                        ("nationality", _NAT_QIDS), ("province", _PROVINCE_QIDS),
                        ("personal_connection", _CONN_QIDS)):
        vals = _vals(answers, *qids)
        if vals:
            out[field] = vals[0]
    return out


def split_survey_answers(answers: dict) -> dict:
    """Group raw survey answers by origin survey for the holistic visitor profile:
    demographic (presurvey) vs personalization vs anything else. Keys are the raw
    question ids; values are the raw answers."""
    demo_qids = set(_AGE_QIDS) | set(_GENDER_QIDS) | set(_NAT_QIDS) | set(_PROVINCE_QIDS) | set(_CONN_QIDS)
    pers_qids = {v for qid in _PERSONALIZATION for v in _qid_variants(qid)}
    demographic: dict = {}
    personalization: dict = {}
    other: dict = {}
    for k, v in (answers or {}).items():
        bucket = demographic if k in demo_qids else personalization if k in pers_qids else other
        bucket[k] = v
    return {"demographic": demographic, "personalization": personalization, "other": other}


def survey_affinity(answers: dict) -> dict[str, float]:
    """Survey answers -> {tag_key: weight} in the content taxonomy (the persona)."""
    out: dict[str, float] = {}

    for v in _vals(answers, *_AGE_QIDS):
        if v in _AGE:
            out[f"person_who.age_group:{_AGE[v]}"] = 0.5
    for v in _vals(answers, *_GENDER_QIDS):
        if v in _GENDER:
            out[f"person_who.gender_and_age:{_GENDER[v]}"] = 0.3
    for v in _vals(answers, *_NAT_QIDS):
        country = str(v).replace("_", " ").title()
        out[f"person_who.city_village_country:From: {country}"] = 0.4
        # rollup: visitors whose origin is not one of the prominently-tagged
        # countries also match content tagged `International` (the complement set).
        if country.strip().casefold() not in CORE_COUNTRIES:
            out["person_who.city_village_country:International"] = 0.3
    # visitor's NL province -> person_who.province_netherlands, the facet the content
    # province tags use per the authoritative taxonomy (tags.json). score_tag boosts
    # same-province stories. Keep the label as-is (hyphens matter: "Zuid-Holland").
    for v in _vals(answers, *_PROVINCE_QIDS):
        if v:
            out[f"person_who.province_netherlands:{str(v).strip()}"] = 0.5

    # explicit preference questions: value IS the taxonomy label -> strong weight.
    # canonicalize the value so slug/underscore/spelling variants still match content.
    for qid, facet in _PERSONALIZATION.items():
        for v in _vals(answers, *_qid_variants(qid)):
            if v:
                out[f"{facet}:{_canonical_label(v)}"] = 1.0

    return out
