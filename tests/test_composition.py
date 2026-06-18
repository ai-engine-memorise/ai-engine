from ai_engine.recsys.composition import _build_config
from ai_engine.recsys.contracts.config import RecConfig


def test_config_defaults_when_no_env(monkeypatch):
    for k in ("RECSYS_W_SEMANTIC", "RECSYS_MMR_LAMBDA", "RECSYS_FINAL_LIMIT"):
        monkeypatch.delenv(k, raising=False)
    cfg = _build_config()
    d = RecConfig()
    assert cfg.fusion.semantic == d.fusion.semantic
    assert cfg.final_limit == d.final_limit


def test_env_overrides_weights_and_limits(monkeypatch):
    monkeypatch.setenv("RECSYS_W_SEMANTIC", "0.5")
    monkeypatch.setenv("RECSYS_W_AVERSION", "-0.4")
    monkeypatch.setenv("RECSYS_MMR_LAMBDA", "0.9")
    monkeypatch.setenv("RECSYS_FINAL_LIMIT", "8")
    monkeypatch.setenv("RECSYS_DISTRACTOR_PROBABILITY", "0.5")
    cfg = _build_config()
    assert cfg.fusion.semantic == 0.5
    assert cfg.fusion.aversion == -0.4
    assert cfg.mmr_lambda == 0.9
    assert cfg.final_limit == 8
    assert cfg.distractor_probability == 0.5


def test_bad_env_value_falls_back_to_default(monkeypatch):
    monkeypatch.setenv("RECSYS_W_TAG", "not_a_number")
    monkeypatch.setenv("RECSYS_FINAL_LIMIT", "xx")
    cfg = _build_config()
    assert cfg.fusion.tag == RecConfig().fusion.tag      # ignored, kept default
    assert cfg.final_limit == RecConfig().final_limit
