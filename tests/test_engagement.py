from ai_engine.recsys.contracts import RecConfig
from ai_engine.recsys.contracts.enums import EndReason, Outcome
from ai_engine.recsys.signals.engagement import (
    estimate_reading_time, engagement_strength, classify_outcome,
)

CFG = RecConfig()


def _strength(dwell, est, reason, visits=1, rating=None):
    return engagement_strength(
        dwell_seconds=dwell, est_reading_time=est, end_reason=reason,
        visits=visits, survey_rating=rating, cfg=CFG,
    )


def test_reading_time_scales_with_words():
    assert estimate_reading_time(200, False, CFG) > estimate_reading_time(50, False, CFG)


def test_image_adds_reading_time():
    assert estimate_reading_time(50, True, CFG) > estimate_reading_time(50, False, CFG)


def test_full_read_with_next_is_positive():
    s = _strength(60, 30, EndReason.next_button)
    assert classify_outcome(s, CFG) == Outcome.positive


def test_quick_abandon_is_negative():
    s = _strength(1, 30, EndReason.abandon)
    assert classify_outcome(s, CFG) == Outcome.negative


def test_strength_monotonic_in_dwell():
    prev = float("-inf")
    for dwell in [1, 5, 10, 20, 40, 80, 160]:
        cur = _strength(dwell, 30, EndReason.next_button)
        assert cur >= prev
        prev = cur


def test_revisits_increase_strength():
    one = _strength(30, 30, EndReason.next_button, visits=1)
    many = _strength(30, 30, EndReason.next_button, visits=5)
    assert many > one


def test_positive_survey_increases_strength():
    base = _strength(30, 30, EndReason.close_button, rating=None)
    happy = _strength(30, 30, EndReason.close_button, rating=5)
    assert happy > base
