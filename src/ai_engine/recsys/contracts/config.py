from pydantic import BaseModel


class EngagementWeights(BaseModel):
    """How much each behavioral signal contributes to engagement strength."""
    dwell: float = 0.4
    completion: float = 0.3
    revisit: float = 0.2
    survey: float = 0.1


class FusionWeights(BaseModel):
    """How much each scorer contributes to the final fused score. Each scorer -> [0,1]."""
    semantic: float = 0.30    # similarity to whole-history taste CENTROID (blurred over likes)
    affinity: float = 0.25    # item-kNN: max similarity to ANY one liked item (sharp)
    tag: float = 0.25
    recency: float = 0.10     # sequence awareness: similarity to the MOST-RECENT view
    aversion: float = -0.25   # PENALTY: overlap with disliked themes (negative weight)
    geo: float = 0.20         # proximity to the request's location (only scored when one is given)
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

    # geo (independent of the tag filter): proximity scoring + optional radius filter
    geo_scale_m: float = 300.0          # exp(-distance/scale); ~camp-sized falloff
    geo_radius_m: float = 1000.0        # default radius when a geo filter is requested

    # distractor / novelty injection (exploration): one deliberately off-profile item,
    # placed in a fixed slot and labelled kind="distractor" so the UI can surface it.
    distractor_enabled: bool = True
    distractor_strategy: str = "max_dissimilar"   # max_dissimilar | unexplored_theme | random
    distractor_probability: float = 1.0           # 1.0 = always inject a distractor
    distractor_slots: list[int] = [3, 4]          # placed at one of these positions (random)

    # cold/warm routing
    cold_start_min_positives: int = 1
