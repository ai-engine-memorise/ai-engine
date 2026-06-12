from ai_engine.recsys.contracts.models import Content, Tag, UserSignals
from ai_engine.recsys.ranking.scorers import (
    cosine, score_semantic, score_affinity, score_tag, score_aversion,
)


def test_cosine_bounds():
    assert cosine([1, 0], [1, 0]) == 1.0
    assert cosine([1, 0], [0, 1]) == 0.0
    assert cosine([1, 0], [-1, 0]) == -1.0
    assert cosine(None, [1, 0]) == 0.0


def test_semantic_in_unit_range_and_aligned_is_best():
    sig = UserSignals(user_id="u", taste_vector=[1.0, 0.0])
    aligned = score_semantic(sig, [1.0, 0.0])
    orthogonal = score_semantic(sig, [0.0, 1.0])
    opposite = score_semantic(sig, [-1.0, 0.0])
    assert aligned == 1.0
    assert orthogonal == 0.5
    assert opposite == 0.0
    for v in (aligned, orthogonal, opposite):
        assert 0.0 <= v <= 1.0


def test_tag_overlap_rewards_matching_tags():
    sig = UserSignals(user_id="u", tag_affinity={"theme_what:Forced Labor": 1.0})
    match = Content(id="x", tags=[Tag(facet="theme_what", label="Forced Labor", weight=1.0)])
    miss = Content(id="y", tags=[Tag(facet="theme_what", label="Family", weight=1.0)])
    assert score_tag(sig, match) > 0.0
    assert score_tag(sig, miss) == 0.0
    assert 0.0 <= score_tag(sig, match) <= 1.0


def test_tag_match_is_case_insensitive():
    # demographic bridge builds "...:Female"; taxonomy content tag is "female"
    sig = UserSignals(user_id="u", tag_affinity={"person_who.gender_and_age:Female": 1.0})
    c = Content(id="x", tags=[Tag(facet="person_who.gender_and_age", label="female", weight=1.0)])
    assert score_tag(sig, c) > 0.0


def test_tag_score_zero_without_affinity():
    sig = UserSignals(user_id="u", tag_affinity={})
    c = Content(id="x", tags=[Tag(facet="theme_what", label="Forced Labor")])
    assert score_tag(sig, c) == 0.0


def test_affinity_is_max_sim_to_any_liked_item():
    # two distinct tastes; centroid sits between, but max-sim rewards the near one
    liked = [(1.0, [1.0, 0.0]), (1.0, [0.0, 1.0])]
    near_first = score_affinity([1.0, 0.0], liked)     # identical to liked item #1
    between = score_affinity([0.707, 0.707], liked)    # the centroid direction
    assert near_first == 1.0                            # max-sim to a liked item -> full
    assert near_first > between                         # sharper than the centroid
    assert score_affinity([1.0, 0.0], []) == 0.0        # no likes -> 0
    assert 0.0 <= between <= 1.0


def test_affinity_scaled_by_like_strength():
    weak = score_affinity([1.0, 0.0], [(0.2, [1.0, 0.0])])
    strong = score_affinity([1.0, 0.0], [(1.0, [1.0, 0.0])])
    assert strong > weak                                # relative like weight matters


def test_aversion_penalizes_disliked_themes():
    sig = UserSignals(user_id="u", tag_aversion={"theme_what:Violence": 1.0})
    disliked = Content(id="x", tags=[Tag(facet="theme_what", label="Violence", weight=1.0)])
    other = Content(id="y", tags=[Tag(facet="theme_what", label="Family", weight=1.0)])
    assert score_aversion(sig, disliked) > 0.0          # shares disliked theme -> penalty
    assert score_aversion(sig, other) == 0.0
    assert 0.0 <= score_aversion(sig, disliked) <= 1.0
    assert score_aversion(UserSignals(user_id="u"), disliked) == 0.0  # no aversion -> 0
