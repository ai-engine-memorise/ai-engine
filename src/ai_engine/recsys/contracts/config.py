from pydantic import BaseModel


class EngagementWeights(BaseModel):
    """How much each behavioral signal contributes to engagement strength."""
    dwell: float = 0.4
    completion: float = 0.3
    revisit: float = 0.2
    survey: float = 0.1


class FusionWeights(BaseModel):
    """How much each scorer contributes to the final fused score. Each scorer -> [0,1]."""
    semantic: float = 0.5
    tag: float = 0.3
    geo: float = 0.0          # off in this slice
    popularity: float = 0.0   # off by default


class RecConfig(BaseModel):
    """All tunables in one typed place so tests pin behavior by passing a config."""
    engagement: EngagementWeights = EngagementWeights()
    fusion: FusionWeights = FusionWeights()

    # reading-time estimate (drives dwell_ratio)
    reading_speed_wps: float = 4.2      # ~250 wpm
    img_extra_time: float = 1.3         # seconds added for an image

    # engagement shaping
    dwell_cap_ratio: float = 2.0        # dwell/estimate capped here before normalizing
    positive_threshold: float = 0.30    # strength above -> positive signal
    negative_threshold: float = -0.05   # strength below -> negative signal

    # recency + negatives
    half_life_days: float = 14.0        # exponential decay of signal weight
    soft_negative_weight: float = 0.30  # penalty for shown-but-not-viewed (impressions)

    # candidate generation + ranking
    pool_per_generator: int = 30
    final_limit: int = 10
    mmr_lambda: float = 0.7             # relevance(1) vs diversity(0)

    # distractor / novelty injection (exploration): one deliberately off-profile item,
    # placed in a fixed slot and labelled kind="distractor" so the UI can surface it.
    distractor_enabled: bool = True
    distractor_strategy: str = "max_dissimilar"  # max_dissimilar | unexplored_theme | random
    distractor_slot: int = 3                      # 0-based position in the returned list

    # cold/warm routing
    cold_start_min_positives: int = 1
