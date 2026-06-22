"""Persona<->content tag match normalization, and the survey-side facet fixes."""
from ai_engine.recsys.taxonomy import normalize_key, normalize_label
from ai_engine.recsys.survey import survey_affinity, CORE_COUNTRIES


def test_normalize_is_symmetric_across_drift():
    # the survey side and the content side must produce the same key for these.
    assert normalize_label("Personal Stories") == normalize_label("personal stories")
    assert normalize_label("Sport & Theatre") == normalize_label("Sport & Theater")
    assert normalize_label("Düsseldorf") == "dusseldorf"
    assert normalize_key("THEME_WHAT:Forced Labour") == "theme_what:forced labor"


def test_province_uses_person_who_facet_per_authority():
    # tags.json puts Province under person_who, not place_where.
    aff = survey_affinity({"q:province": "Drenthe"})
    assert any(k.startswith("person_who.province_netherlands:") for k in aff)
    assert not any(k.startswith("place_where.province_netherlands:") for k in aff)


def test_non_core_nationality_emits_international_rollup():
    aff = survey_affinity({"q:nationality": "spain"})
    assert "person_who.city_village_country:From: Spain" in aff
    assert "person_who.city_village_country:International" in aff


def test_core_nationality_does_not_emit_international():
    assert "netherlands" in CORE_COUNTRIES
    aff = survey_affinity({"q:nationality": "netherlands"})
    assert not any(k.endswith(":International") for k in aff)
