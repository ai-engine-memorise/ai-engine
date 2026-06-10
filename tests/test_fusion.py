from ai_engine.recsys.contracts.config import FusionWeights
from ai_engine.recsys.contracts.models import ScoredCandidate
from ai_engine.recsys.ranking.fusion import weighted_fuse, mmr_rerank


def test_weighted_fuse_sums_weighted_scores():
    weights = FusionWeights(semantic=0.5, tag=0.3)
    fused, breakdown = weighted_fuse({"semantic": 0.8, "tag": 0.4}, weights)
    assert abs(breakdown["semantic"] - 0.4) < 1e-9
    assert abs(breakdown["tag"] - 0.12) < 1e-9
    assert abs(fused - 0.52) < 1e-9


def _sc(cid, score):
    return ScoredCandidate(content_id=cid, final_score=score)


def test_mmr_keeps_top_relevance_first():
    cands = [_sc("c1", 0.9), _sc("c2", 0.85), _sc("c3", 0.6)]
    vecs = {"c1": [1, 0], "c2": [1, 0], "c3": [0, 1]}
    for lam in (1.0, 0.5, 0.0):
        out = mmr_rerank(cands, vecs, lambda_=lam, limit=3)
        assert out[0].content_id == "c1"


def test_mmr_promotes_diversity_when_lambda_low():
    # c1,c2 near-identical (same vector); c3 different.
    cands = [_sc("c1", 0.9), _sc("c2", 0.85), _sc("c3", 0.6)]
    vecs = {"c1": [1, 0], "c2": [1, 0], "c3": [0, 1]}

    high_rel = mmr_rerank(cands, vecs, lambda_=1.0, limit=3)
    assert [c.content_id for c in high_rel] == ["c1", "c2", "c3"]  # pure relevance

    diverse = mmr_rerank(cands, vecs, lambda_=0.0, limit=3)
    # the different item is pulled ahead of the near-duplicate
    assert diverse.index(next(c for c in diverse if c.content_id == "c3")) < \
           diverse.index(next(c for c in diverse if c.content_id == "c2"))


def test_mmr_respects_limit():
    cands = [_sc(f"c{i}", 1.0 - i * 0.1) for i in range(5)]
    vecs = {f"c{i}": [1, 0] for i in range(5)}
    out = mmr_rerank(cands, vecs, lambda_=0.7, limit=2)
    assert len(out) == 2
