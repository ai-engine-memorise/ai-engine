from datetime import datetime

from ai_engine.recsys.contracts import RecConfig
from ai_engine.recsys.contracts.models import InteractionEvent
from ai_engine.recsys.survey import survey_affinity, extract_demographics
from ai_engine.recsys.adapters.rudderstack import normalize_event
from ai_engine.recsys.signals.signal_builder import build_user_signals

NOW = datetime(2026, 6, 11, 12, 0, 0)
CFG = RecConfig()


# --- mapping (real quiz-steps instrument) ---------------------------------- #

def test_survey_affinity_maps_person_who_facets():
    a = survey_affinity({"age_group": "25_34", "gender": "female", "nationality": "netherlands"})
    assert "person_who.age_group:age 25-34" in a
    assert "person_who.gender_and_age:female" in a
    assert "person_who.city_village_country:From: Netherlands" in a


def test_prefer_not_say_gender_dropped():
    assert not any("gender_and_age" in k for k in survey_affinity({"gender": "prefer_not_say"}))


def test_extract_demographics():
    d = extract_demographics({"age_group": "65_plus", "personal_connection": "yes", "noise": "x"})
    assert d == {"age": "65_plus", "personal_connection": "yes"}   # normalized field names


def test_kwb_ids_and_personalization_themes():
    a = survey_affinity({"q:age": "25_34", "q:gender": "female",
                         "q:personalization_theme": ["Forced Labor", "Family"],
                         "q:personalization_interest": ["Personal stories"]})
    assert "person_who.age_group:age 25-34" in a
    assert "person_who.gender_and_age:female" in a
    # persona theme picks == content taxonomy labels (1:1 match)
    assert a["theme_what:Forced Labor"] == 1.0 and a["theme_what:Family"] == 1.0
    assert "theme_how.type_of_stores:Personal stories" in a


def test_personalization_label_aliases_normalize_to_taxonomy():
    # slug / underscore / British spelling must all reach the canonical content label
    a = survey_affinity({"q:personalization_theme": ["forced_labour", "deportation"]})
    assert "theme_what:Forced Labor" in a          # 'forced_labour' -> canonical
    assert "theme_what:Deportation" in a           # slug -> Title label
    # already-clean labels pass through unchanged (case preserved)
    b = survey_affinity({"q:personalization_theme": ["Forced Labor"]})
    assert "theme_what:Forced Labor" in b


def test_normalizer_collects_multiselect():
    raw = {"event": "SURVEY_ANSWERED", "userId": "u", "timestamp": "2026-06-11T10:00:00Z",
           "properties": {"answers": [
               {"question_id": "q:personalization_theme", "question_type": "multi", "answer_id": "t1", "answer_value": "Forced Labor"},
               {"question_id": "q:personalization_theme", "question_type": "multi", "answer_id": "t2", "answer_value": "Family"},
           ]}}
    ev = normalize_event(raw)
    assert ev.survey_answers["q:personalization_theme"] == ["Forced Labor", "Family"]


# --- normalizer captures every answer by question_id ----------------------- #

def test_normalize_survey_submitted_keeps_all_answers():
    raw = {
        "event": "SURVEY_SUBMITTED", "userId": "u1", "timestamp": "2026-06-11T10:00:00Z",
        "properties": {"answers": [
            {"question_id": "age_group", "question_type": "choice", "answer_id": "a", "answer_value": "25_34"},
            {"question_id": "gender", "question_type": "choice", "answer_id": "b", "answer_value": "female"},
        ]},
    }
    ev = normalize_event(raw)
    assert ev.event == "SURVEY_SUBMITTED"
    assert ev.survey_answers["age_group"] == "25_34"
    assert ev.survey_answers["gender"] == "female"


# --- survey event folds into the user model -------------------------------- #

def test_survey_event_seeds_user_model():
    ev = InteractionEvent(
        user_id="u1", event="SURVEY_SUBMITTED", ts=NOW,
        survey_answers={"age_group": "25_34", "gender": "female", "nationality": "netherlands"},
    )
    sig = build_user_signals(user_id="u1", events=[ev], contents={}, vectors={}, now=NOW, cfg=CFG)
    keys = sig.tag_affinity.keys()
    assert any(k.startswith("person_who.age_group") for k in keys)
    assert any(k.startswith("person_who.gender_and_age") for k in keys)
    assert sig.demographics.get("age") == "25_34"  # stored -> retrievable via /usermodel


def test_province_maps_to_person_who_facet():
    # province tags live under person_who.province_netherlands (tags.json / live Qdrant)
    a = survey_affinity({"q:province": "Utrecht"})
    assert a["person_who.province_netherlands:Utrecht"] == 0.5
    b = survey_affinity({"province": "Zuid-Holland"})            # hyphen preserved (matters)
    assert "person_who.province_netherlands:Zuid-Holland" in b
    assert extract_demographics({"q:province": "Gelderland"})["province"] == "Gelderland"


def test_province_seeds_affinity_matching_content_tags():
    # a Utrecht visitor's model carries the SAME key content tagged Utrecht uses
    ev = InteractionEvent(user_id="u", event="SURVEY_SUBMITTED", ts=NOW,
                          survey_answers={"q:province": "Utrecht"})
    sig = build_user_signals(user_id="u", events=[ev], contents={}, vectors={}, now=NOW, cfg=CFG)
    assert "person_who.province_netherlands:utrecht" in sig.tag_affinity   # lowercased fold


def test_identify_payload_normalizes_to_demographic_event():
    raw = {"type": "identify", "userId": "u9", "timestamp": "2026-06-11T10:00:00Z",
           "traits": {"age": 30, "gender": "female", "country": "netherlands"}}
    ev = normalize_event(raw)
    assert ev.event == "IDENTIFY"
    assert ev.survey_answers["gender"] == "female" and ev.survey_answers["age"] == 30


def test_identify_event_seeds_user_model_demographics():
    # the app sends identify AFTER the survey -> traits seed the model like a survey
    ev = InteractionEvent(user_id="u9", event="IDENTIFY", ts=NOW,
                          survey_answers={"age": 30, "gender": "female", "country": "netherlands"})
    sig = build_user_signals(user_id="u9", events=[ev], contents={}, vectors={}, now=NOW, cfg=CFG)
    assert sig.demographics.get("age") == 30
    # numeric age bucketed into a person_who facet (cold-start bridge)
    assert any(k.startswith("person_who.age_group") for k in sig.tag_affinity)
    assert any(k.startswith("person_who.gender_and_age") for k in sig.tag_affinity)


def test_kwb_personalization_theme_drives_affinity():
    ev = InteractionEvent(user_id="u1", event="SURVEY_SUBMITTED", ts=NOW,
                          survey_answers={"q:personalization_theme": ["Forced Labor"], "q:age": "25_34"})
    sig = build_user_signals(user_id="u1", events=[ev], contents={}, vectors={}, now=NOW, cfg=CFG)
    assert "theme_what:forced labor" in sig.tag_affinity        # lowercased key -> matches content tags
    assert any(k.startswith("person_who.age_group") for k in sig.tag_affinity)
