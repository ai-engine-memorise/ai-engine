"""Survey -> user-model mapping (the persona).

Survey answers become tag-affinity in the SAME taxonomy the content is tagged with,
so the recommender's score_tag does persona<->content tag similarity. Supports the
kwb research survey (survey:kwb:survey: q:age/q:gender/q:nationality + q:personalization_*)
and the demographic onboarding quiz (age_group/gender/nationality).

Personalization questions are explicit preferences: their answer VALUE must be the
canonical taxonomy label (e.g. "Forced Labor") so it matches content tags 1:1.
"""
from __future__ import annotations
import re as _re
from typing import Optional

# Origins with their own country-specific content tags. Any other nationality rolls
# up to the `International` tag (see survey_affinity). Compared casefolded.
CORE_COUNTRIES = {"netherlands", "germany", "poland"}
# demonym (canonical nationality answer) -> country noun used by content From: tags
_DEMONYM_COUNTRY = {"dutch": "Netherlands", "german": "Germany", "polish": "Poland",
                    "belgian": "Belgium", "french": "France", "british": "United Kingdom",
                    "american": "United States", "israeli": "Israel"}

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
_PROVINCE_QIDS = ("q:province", "province", "region",
                  "q:nationality_province", "nationality_province")   # NL province the visitor is from
_CONN_QIDS = ("q:ww2_connection", "personal_connection")

# personalization questions: answer value = canonical taxonomy label -> facet.
# Like the demographic qids above, each question also has a prefix-less alias:
# production clients send "personalization_theme", not "q:personalization_theme".
_PERSONALIZATION = {
    "q:personalization_theme": "theme_what",
    "personalization_theme": "theme_what",
    "q:personalization_interest": "theme_how.type_of_stores",
    "personalization_interest": "theme_how.type_of_stores",
    "q:personalization_area": "place_where.camp_areas",
    "personalization_area": "place_where.camp_areas",
}

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


# Aggregation-level canonicalization of raw demographic answers. Production clients
# answer in several languages and UI versions, so the same meaning arrives as
# "No.", "Nee.", "no": fold spelling/language variants onto one token per meaning
# and drop placeholder junk, WITHOUT touching the stored raw values.
_DEMO_CANON = {
    "gender": {"f": "female", "vrouw": "female", "m": "male", "man": "male",
               "nonbinary": "non_binary", "x": "non_binary",
               "prefer_not_to_say": "no_answer", "geen_antwoord": "no_answer"},
    "email_shared": {"ja": "yes", "true": "yes", "nee": "no", "false": "no"},
    "personal_connection": {"ja": "yes", "nee": "no",
                            "ik_weet_het_niet": "unknown", "weet_niet": "unknown",
                            "i_dont_know": "unknown", "idk": "unknown", "dont_know": "unknown",
                            # declined-to-answer, all languages/phrasings -> one bucket
                            "i_do_not_want_to_answer": "no_answer",
                            "i_dont_want_to_answer": "no_answer",
                            "ik_wil_geen_antwoord_geven": "no_answer",
                            "geen_antwoord": "no_answer",
                            "prefer_not_to_say": "no_answer",
                            "liever_niet": "no_answer"},
    "nationality": {"nederlandse": "dutch", "belgische": "belgian", "duitse": "german",
                    "franse": "french", "poolse": "polish", "britse": "british",
                    # country noun -> demonym, so "netherlands" and "dutch" fold together
                    "netherlands": "dutch", "belgium": "belgian", "germany": "german",
                    "france": "french", "poland": "polish", "uk": "british",
                    "united_kingdom": "british", "new_zealand": "new_zealander",
                    # Dutch-language demonyms -> the English canonical (the survey UI
                    # serves both languages, so both spellings arrive)
                    "nederlands": "dutch", "nederlander": "dutch",
                    "afghaan": "afghan", "afghaanse": "afghan",
                    "albanees": "albanian", "albanese": "albanian",
                    "amerikaan": "american", "amerikaanse": "american",
                    "australiër": "australian", "australische": "australian",
                    "braziliaan": "brazilian", "braziliaanse": "brazilian",
                    "canadees": "canadian", "canadese": "canadian",
                    "chinees": "chinese",
                    "deen": "danish", "deense": "danish",
                    "duitser": "german",
                    "engels": "british", "engelse": "british", "brit": "british",
                    "fin": "finnish", "finse": "finnish",
                    "fransman": "french", "française": "french",
                    "griek": "greek", "griekse": "greek",
                    "ier": "irish", "ierse": "irish",
                    "indiaas": "indian", "indiase": "indian",
                    "israëliër": "israeli", "israelisch": "israeli", "israëlische": "israeli",
                    "italiaan": "italian", "italiaanse": "italian",
                    "japanner": "japanese", "japans": "japanese", "japanse": "japanese",
                    "marokkaan": "moroccan", "marokkaanse": "moroccan",
                    "noor": "norwegian", "noorse": "norwegian",
                    "oostenrijker": "austrian", "oostenrijkse": "austrian",
                    "oekraïens": "ukrainian", "oekraïense": "ukrainian",
                    "pool": "polish",
                    "portugees": "portuguese", "portugese": "portuguese",
                    "rus": "russian", "russische": "russian",
                    "spanjaard": "spanish", "spaans": "spanish", "spaanse": "spanish",
                    "tsjech": "czech", "tsjechische": "czech",
                    "turk": "turkish", "turkse": "turkish",
                    "zweed": "swedish", "zweedse": "swedish",
                    "zwitser": "swiss", "zwitserse": "swiss"},
}
_DEMO_JUNK = {"select", "select___", "none", "null", "_", "n_a"}


# gender is a closed set: any free-typed value outside it ("Magnetron", jokes,
# typos) folds into 'other' instead of minting a first-class category in the
# cohort statistics. Every other field stays open (nationality etc. are
# legitimately unbounded).
_GENDER_CANONICAL = {"female", "male", "non_binary", "no_answer", "other"}


def canon_demo_value(field: str, value) -> Optional[str]:
    """One canonical token per answer meaning, or None for placeholder junk.
    Empty/junk gender collapses to 'no_answer' (a real survey outcome); junk in
    any other field is dropped from distributions entirely."""
    v = str(value or "").strip().lower().rstrip(".")
    v = v.replace("'", "").replace("’", "")
    v = _re.sub(r"[\s\-.]+", "_", v).strip("_") if v else ""
    # values with no letter/digit at all (zero-width chars, dashes, emoji …) render
    # as blank labels: treat them exactly like an empty answer
    if not v or v in _DEMO_JUNK or not _re.search(r"[a-z0-9]", v):
        return "no_answer" if field == "gender" else None
    out = _DEMO_CANON.get(field, {}).get(v, v)
    if field == "gender" and out not in _GENDER_CANONICAL:
        return "other"
    return out


def demo_label(field: str, value) -> str:
    """Human label for a canonical demographic value. The server owns both the
    semantics (canon_demo_value) and the wording, so the dashboard renders labels
    verbatim instead of re-deriving them (docs/debt-payload-scatter.md D3)."""
    v = str(value or "")
    if field == "age":
        if v.startswith("under_"):
            return "<" + "".join(ch for ch in v if ch.isdigit())
        return v.replace("_plus", "+").replace("_", "–")
    return v.replace("_", " ").strip().title()


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
    """Pull the demographic fields out of raw survey answers into a flat dict.

    Reads the five demographic fields (age, gender, nationality, province,
    personal_connection) from a survey or identify payload and returns them as plain
    `{field: value}`. Each field can arrive under several question ids or trait keys (for
    example age as `q:age`, `age_group`, or `age`); the first id that is present wins, and
    for a multi-select answer the first value is taken.

    The values are kept **raw** (for example `"55_64"`, not the taxonomy label
    `"age 55-64"`). This function is only for storing and inspecting who the visitor is
    (it populates `UserSignals.demographics`). Turning demographics into weighted taxonomy
    tags for matching is a separate step, `survey_affinity` / `_demographic_affinity`.

    Args:
        answers: `question_id -> answer` from a survey/identify event.

    Returns:
        `{field: value}` containing only the demographic fields that were answered
        (empty dict if none are present).

    Example:
        ```python
        extract_demographics({
            "q:age": "55_64",
            "q:gender": "female",
            "nationality": "france",       # from an identify trait
        })
        # {"age": "55_64", "gender": "female", "nationality": "france"}
        ```
    """
    out = {}
    for field, qids in (("age", _AGE_QIDS), ("gender", _GENDER_QIDS),
                        ("nationality", _NAT_QIDS), ("province", _PROVINCE_QIDS),
                        ("personal_connection", _CONN_QIDS)):
        vals = _vals(answers, *qids)
        if vals:
            out[field] = vals[0]
    # email: PII-guarded to a pure yes/no flag — did the visitor leave an address?
    # The address itself NEVER enters the stored demographics.
    for k, v in answers.items():
        if "email" in str(k).lower():
            sv = str(_clean(v) or "").strip().lower()
            if sv in ("", "no", "nee", "false", "none", "null"):
                out["email_shared"] = "no"
            elif "@" in sv or sv in ("yes", "ja", "true"):
                out["email_shared"] = "yes"
            else:
                out["email_shared"] = "no"
            break
    return out


# English labels for the canonical answer codes (the app sends both a Dutch
# display string and a canonical code per question; the dashboard shows English)
_LIKERT = {"1": "Strongly disagree", "2": "Disagree", "3": "Undecided",
           "4": "Agree", "5": "Strongly agree"}
_YN = {"yes": "Yes", "no": "No", "unknown": "Don't know"}
_ANSWER_LABELS = {
    "prior_visit": {"yes": "Yes", "no": "No", "once": "Yes, once",
                    "multiple": "Yes, several times", "several": "Yes, several times",
                    "unknown": "Don't know"},
    "other_camp_connection": _YN, "ww2_connection": _YN, "religious": _YN,
    "knowledge_level": _LIKERT, "learned_something": _LIKERT, "interest": _LIKERT,
    "weak_impression": _LIKERT, "innovative": _LIKERT, "boring": _LIKERT,
    "overwhelmed": _LIKERT, "made_me_think": _LIKERT, "share_learning": _LIKERT,
    "learn_more": _LIKERT,
    "political_identity": {"1": "Left", "2": "Left of center", "3": "Center",
                           "4": "Right of center", "5": "Right"},
    "activities_before": {"start_here": "Started the visit here",
                          "exhibition": "Visited the exhibition",
                          "museum": "Visited the museum",
                          "camp_terrain": "Visited the camp terrain"},
    "education": {"primary": "Primary school", "high_school": "High school",
                  "vocational": "Vocational (MBO)", "bachelor": "Bachelor (HBO/WO)",
                  "master": "Master or higher", "other": "Other"},
    "visit_context": {"alone": "Alone", "partner": "With partner", "family": "With family",
                      "teen_parent": "As a teen with parent(s)",
                      "child_parent": "As a child with parent(s)",
                      "friends": "With friends", "group": "With a group", "school": "School visit"},
    "visit_purpose": {"learn_history": "Learn about the history", "commemoration": "Commemoration",
                      "family_history": "Family history", "school": "School / study",
                      "passing_by": "Passing by", "other": "Other"},
    "relevance": {"past": "The past", "present": "The present", "future": "The future"},
}


def _answer_label(base: str, v):
    """Canonical code -> English label; free text / unknown codes pass through.
    Likert answers come back structured ({label, score, scale}) so the dashboard
    can draw them as a rating bar instead of text."""
    if isinstance(v, list):
        return [_answer_label(base, x) for x in v]
    m = _ANSWER_LABELS.get(base)
    sv = str(v).strip()
    if m is _LIKERT and sv.isdigit():
        return {"label": m.get(sv, sv), "score": int(sv), "scale": 5}
    if m:
        return m.get(sv) or m.get(sv.lower()) or v
    return v


# post-visit evaluation questions (the exit survey), split out from the visit
# background so the profile card reads in two sensible blocks
_FEEDBACK_BASE = {"learned_something", "what_learned", "interest", "weak_impression",
                  "innovative", "boring", "overwhelmed", "made_me_think", "memorable",
                  "share_learning", "learn_more", "meaning_today", "relevance"}


def split_survey_answers(answers: dict) -> dict:
    """Group raw survey answers for the holistic visitor profile.

    The app sends every question TWICE: a plain key with the human display text
    ("knowledge_level": "Oneens") and a q:-prefixed key with the canonical code
    ("q:knowledge_level": "2"). Both used to render as separate duplicate rows;
    now each base question appears once, preferring the human display value.
    Email answers are dropped entirely (PII — the email_shared flag covers them),
    and zero-width characters are stripped from values."""
    demo_qids = set(_AGE_QIDS) | set(_GENDER_QIDS) | set(_NAT_QIDS) | set(_PROVINCE_QIDS) | set(_CONN_QIDS)
    pers_qids = set(_PERSONALIZATION)

    def cleanv(v):
        if isinstance(v, list):
            return [cleanv(x) for x in v]
        return str(v).replace("​", "").strip()

    # dedup q:/plain pairs onto the base name. The q:-prefixed key carries the
    # localized display text ("q:prior_visit": "Nee."), the plain key the
    # canonical code ("prior_visit": "no"). The dashboard is English, so the
    # CANONICAL value wins and is mapped to an English label; localized free
    # text (open questions) has no code counterpart and passes through.
    merged: dict = {}
    for k, v in (answers or {}).items():
        if v is None or v == "":
            continue
        is_q = str(k).startswith("q:")
        base = k[2:] if is_q else str(k)
        if "email" in base.lower():
            continue
        if not is_q or base not in merged:
            cv = cleanv(v)
            if isinstance(cv, list):
                cv = [x for x in cv if x != ""]
            if cv == "" or cv == []:      # answer was only whitespace / zero-width junk
                continue
            merged[base] = cv

    demographic: dict = {}
    personalization: dict = {}
    background: dict = {}
    feedback: dict = {}
    for base, v in merged.items():
        qids = {base, "q:" + base}
        if qids & demo_qids:
            demographic[base] = v
        elif qids & pers_qids or base.startswith("personalization_"):
            personalization[base] = v
        elif base in _FEEDBACK_BASE:
            feedback[base] = _answer_label(base, v)
        else:
            background[base] = _answer_label(base, v)
    return {"demographic": demographic, "personalization": personalization,
            "background": background, "feedback": feedback,
            "other": {**background, **feedback}}   # legacy shape for old clients


def survey_affinity(answers: dict) -> dict[str, float]:
    """Turn raw survey answers into weighted taxonomy tags (the visitor's persona).

    Emits `{"facet:label": weight}` keyed in the SAME taxonomy the content is tagged
    with, so `score_tag` can match persona against content directly. Two kinds of answer
    contribute, at deliberately different weights:

    - **Demographics** (age, gender, nationality, NL province) map to `person_who.*`
      facets at modest weights: age 0.5, gender 0.3, nationality 0.4, province 0.5. A
      "core country" is an origin the collection tags content for specifically
      (`CORE_COUNTRIES` = Netherlands, Germany, Poland); a nationality outside that set
      has no country tag of its own, so it also emits the `International` rollup (0.3) and
      matches content tagged for that non-core complement.
    - **Personalization preferences** (theme / interest / area) are what the visitor
      explicitly picked, so they get the strongest weight (1.0). The answer value IS the
      taxonomy label, run through `_canonical_label` so spelling/separator variants
      (e.g. "forced labour") still line up with the content label ("Forced Labor").

    Multi-select answers emit one key per selected value. These land in the survey side
    of `tag_affinity`; `build_user_signals` blends them with engagement so they dominate
    on cold start (see `ai_engine.recsys.signals.signal_builder.build_user_signals`).

    Args:
        answers: `question_id -> answer` from a survey/identify event (values may be
            scalars or lists; entity-id values like `a:age:55_64` are cleaned first).

    Returns:
        `{"facet:label": weight}`. Empty if no recognized questions are present.

    Example:
        ```python
        survey_affinity({
            "q:age": "55_64",
            "q:gender": "female",
            "q:nationality": "france",              # not a core country
            "q:personalization_theme": "forced labour",
        })
        # {
        #     'person_who.age_group:age 55-64': 0.5,
        #     'person_who.gender_and_age:female': 0.3,
        #     'person_who.city_village_country:From: France': 0.4,
        #     'person_who.city_village_country:International': 0.3,
        #     'theme_what:Forced Labor': 1.0,
        # }
        ```
    """
    out: dict[str, float] = {}

    for v in _vals(answers, *_AGE_QIDS):
        if v in _AGE:
            out[f"person_who.age_group:{_AGE[v]}"] = 0.5
    for v in _vals(answers, *_GENDER_QIDS):
        if v in _GENDER:
            out[f"person_who.gender_and_age:{_GENDER[v]}"] = 0.3
    for v in _vals(answers, *_NAT_QIDS):
        raw = str(v).strip().casefold().replace(" ", "_")
        # answers arrive as demonyms ("dutch"); the core-country set and the
        # content's From: tags use country nouns - map before comparing, or a
        # Dutch visitor gets the International rollup (and misses NL content)
        country = _DEMONYM_COUNTRY.get(raw, str(v).replace("_", " ").title())
        out[f"person_who.city_village_country:From: {country}"] = 0.4
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
        if facet == "place_where.camp_areas":
            continue    # AR areas are a proximity FILTER, not a taste signal:
                        # every recommendation is already within the site
        for v in _vals(answers, qid):
            if not v:
                continue
            if str(v).strip().isdigit():
                continue    # bare content ids (interest picks) are not taxonomy labels
            out[f"{facet}:{_canonical_label(v)}"] = 1.0

    return out
