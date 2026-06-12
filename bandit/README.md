# Contextual bandit — learned ranking policy

A linear contextual bandit that learns the **ranking weights** the static fusion sets by hand.
Global policy (one θ shared across users), trained **offline** from the durable Parquet logs.

```
context  x = [semantic, affinity, tag, recency, aversion, geo]   (per candidate)
reward   r = realized engagement strength of the view it produced (0 if shown, not opened)
model    E[r | x] = θ·x          serve score, + UCB bonus α·√(xᵀA⁻¹x) for exploration
update   A += x xᵀ ;  b += r x ;  θ = A⁻¹ b
prior    A0 = ridge·I ,  b0 = ridge·w_static   ⇒   θ0 = w_static
```

The prior is the key safety property: at θ0 the bandit ranks **identically** to the static
weighted fusion (verified in `tests/test_bandit.py`). Enabling it cannot regress day-one behavior;
it only adjusts weights as data accrues.

## How the loop closes

1. **Serve.** Every `/api/recommend` logs each item's feature vector `x` to `served/date=*/`
   and returns a `request_id`. Features are logged in **static mode too**, so you can fit a bandit
   from traffic served before ever turning it on.
2. **Reward.** The app echoes `request_id` (in `properties.details.request_id`) on the resulting
   `CONTENT_VIEW_*` events. Those land in `date=*/` with the reward signal (dwell, end_reason).
3. **Train.** `bandit/train.py` joins served⨝events on `(request_id, content_id)`, computes the
   reward per impression (engagement strength; shown-but-not-opened → 0), and fits θ.

```bash
# 1. collect traffic with EVENT_LOG_DIR set (static mode is fine)
EVENT_LOG_DIR=./data/eventlog  RECSYS_RANKING_MODE=static  uvicorn ai_engine.recsys.api:app

# 2. train
python bandit/train.py --log ./data/eventlog --out ./data/bandit_state.json
#    impressions=... samples=... rewarded(+)=...
#    feature   prior_theta  trained_theta   <- see the weights move

# 3. serve the learned policy
RECSYS_RANKING_MODE=bandit  BANDIT_STATE_PATH=./data/bandit_state.json  uvicorn ai_engine.recsys.api:app
```

Re-run the trainer on a schedule; it always starts from the prior, so each run is a fresh fit over
all accumulated data (not an incremental drift).

## Knobs

| env / config | default | meaning |
|---|---|---|
| `RECSYS_RANKING_MODE` | `static` | `static` weighted fusion, or `bandit` learned θ |
| `BANDIT_STATE_PATH` | – | trained state JSON; absent ⇒ serve at the prior |
| `RECSYS_BANDIT_ALPHA` / `bandit_alpha` | `0.3` | UCB exploration strength (0 = greedy) |
| `RECSYS_BANDIT_RIDGE` / `bandit_ridge` | `1.0` | prior strength (how tightly θ0 holds the weights) |
| `bandit_explore` | `true` | add the UCB bonus when serving |

## Scope / next

- **Global** policy (one θ). Per-segment θ (e.g. by persona bucket) is the next step — more tailored
  weighting, sparser data per arm.
- **Offline** batch training (delayed reward → robust). Online incremental `update()` exists on the
  policy for a future live-feedback hook (needs a Redis impression-context store keyed by request_id).
- Reward is a proxy (nominal reading-time est; logs lack word_count). Relative reward ordering drives
  learning; swap in the exact engagement strength once word_count is logged at serve.
