"""Survey → user-model mapping.

The cold-start survey (ai-engine-api quiz-steps.json: age_group, gender, nationality,
personal_connection) maps directly onto the person_who taxonomy facets. SURVEY_ANSWERED /
SURVEY_SUBMITTED events (event-catalog) carry these answers; here we turn them into
tag-affinity keys + a demographics dict that the user model stores and the recommender uses.
"""
from __future__ import annotations
from typing import Optional

SURVEY_EVENTS = ("SURVEY_SUBMITTED", "SURVEY_ANSWERED")

# quiz answer_value -> person_who.age_group taxonomy label
_AGE = {
    "under_18": "child",
    "18_24": "age 18-24",
    "25_34": "age 25-34",
    "35_44": "age 35-44",
    "45_54": "age 45-54",
    "55_64": "age 55-64",
    "65_plus": "elderly",
}
# quiz answer_value -> person_who.gender_and_age taxonomy label (prefer_not_say -> drop)
_GENDER = {"female": "female", "male": "Male", "non_binary": "non-binary"}


def extract_demographics(answers: dict) -> dict:
    """Raw survey answers -> demographics dict (for storage / inspection)."""
    out = {}
    for k in ("age_group", "gender", "nationality", "personal_connection"):
        if answers.get(k) is not None:
            out[k] = answers[k]
    return out


def survey_affinity(answers: dict) -> dict[str, float]:
    """Survey answers -> {tag_key: weight} on person_who facets (cold-start signal)."""
    out: dict[str, float] = {}
    if (ag := answers.get("age_group")) in _AGE:
        out[f"person_who.age_group:{_AGE[ag]}"] = 0.5
    if (g := answers.get("gender")) in _GENDER:
        out[f"person_who.gender_and_age:{_GENDER[g]}"] = 0.3
    if nat := answers.get("nationality"):
        country = str(nat).replace("_", " ").title()
        out[f"person_who.city_village_country:From: {country}"] = 0.4
    return out
