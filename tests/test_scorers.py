from ai_engine.recsys.contracts.models import Content, Tag, UserSignals
from ai_engine.recsys.ranking.scorers import cosine, score_semantic, score_tag


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


def test_tag_score_zero_without_affinity():
    sig = UserSignals(user_id="u", tag_affinity={})
    c = Content(id="x", tags=[Tag(facet="theme_what", label="Forced Labor")])
    assert score_tag(sig, c) == 0.0
