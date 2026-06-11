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

SURVEY_EVENTS = ("SURVEY_SUBMITTED", "SURVEY_ANSWERED")

# age answer value -> person_who.age_group taxonomy label (kwb + quiz buckets)
_AGE = {
    "under_18": "child", "under_16": "child",
    "16_18": "age 16-18", "18_24": "age 18-24", "25_34": "age 25-34",
    "35_44": "age 35-44", "45_54": "age 45-54", "55_64": "age 55-64",
    "65_74": "age 65-74", "75_84": "age 75-84", "85_plus": "age 85+",
    "65_plus": "elderly",
}
_GENDER = {"female": "female", "male": "Male", "non_binary": "non-binary"}

# question_id -> (answer keys it may use)  for demographics
_AGE_QIDS = ("q:age", "age_group")
_GENDER_QIDS = ("q:gender", "gender")
_NAT_QIDS = ("q:nationality", "nationality")
_CONN_QIDS = ("q:ww2_connection", "personal_connection")

# personalization questions: answer value = canonical taxonomy label -> facet
_PERSONALIZATION = {
    "q:personalization_theme": "theme_what",
    "q:personalization_interest": "theme_how.type_of_stores",
    "q:personalization_area": "place_where.camp_areas",
}


def _vals(answers: dict, *qids: str) -> list:
    """All answer values for the first matching question id (scalar or multi-list)."""
    for qid in qids:
        if qid in answers and answers[qid] is not None:
            v = answers[qid]
            return v if isinstance(v, list) else [v]
    return []


def extract_demographics(answers: dict) -> dict:
    """Raw survey answers -> demographics dict (for storage / inspection)."""
    out = {}
    for field, qids in (("age", _AGE_QIDS), ("gender", _GENDER_QIDS),
                        ("nationality", _NAT_QIDS), ("personal_connection", _CONN_QIDS)):
        vals = _vals(answers, *qids)
        if vals:
            out[field] = vals[0]
    return out


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
        out[f"person_who.city_village_country:From: {str(v).replace('_', ' ').title()}"] = 0.4

    # explicit preference questions: value IS the taxonomy label -> strong weight
    for qid, facet in _PERSONALIZATION.items():
        for v in _vals(answers, qid):
            if v:
                out[f"{facet}:{v}"] = 1.0

    return out
