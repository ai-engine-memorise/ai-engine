"""Contextual bandit ranking policy (linear, LinUCB-style). PURE: no IO, no numpy.

Replaces the STATIC weighted fusion with a LEARNED linear reward model over the
SAME per-scorer features. The hand-set FusionWeights become the bandit's PRIOR
(θ0 = weights via a ridge prior), so enabling the bandit starts at EXACTLY the
current behavior and learns away from it as (features -> reward) data arrives.

    context  x = [semantic, affinity, tag, recency, aversion, geo]   (per candidate)
    action   = recommend a candidate
    reward   r = realized engagement strength of the view it produced
    model    E[r | x] = θ·x      with UCB exploration bonus  α·√(xᵀ A⁻¹ x)

Online update (LinUCB):  A += x xᵀ ;  b += r x ;  θ = A⁻¹ b.
The reward is delayed (a view ends after we serve), so updates are applied by the
OFFLINE trainer that joins the served-log (features) to the event-log (reward).
d is tiny (~6), so a hand-rolled matrix inverse keeps this dependency-free.
"""
from __future__ import annotations
import math
from typing import Optional, Sequence

# the per-candidate feature vector, in a FIXED order (popularity omitted: always 0 here).
FEATURE_ORDER: tuple[str, ...] = ("semantic", "affinity", "tag", "recency", "aversion", "geo")


def feature_vector(per: dict, order: Sequence[str] = FEATURE_ORDER) -> list[float]:
    """Ordered context vector from a per-scorer dict (missing scorer -> 0.0)."""
    return [float(per.get(name, 0.0)) for name in order]


# ----- small dense linear algebra (d <= ~8) -------------------------------- #

def _mat_inverse(M: list[list[float]]) -> list[list[float]]:
    """Inverse via Gauss-Jordan with partial pivoting. M is square, here SPD."""
    n = len(M)
    a = [list(row) + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(M)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(a[r][col]))
        if abs(a[piv][col]) < 1e-12:
            a[piv][col] += 1e-9  # ridge guard: never singular in practice (A = λI + ...)
        a[col], a[piv] = a[piv], a[col]
        d = a[col][col]
        a[col] = [v / d for v in a[col]]
        for r in range(n):
            if r != col and a[r][col] != 0.0:
                f = a[r][col]
                a[r] = [rv - f * cv for rv, cv in zip(a[r], a[col])]
    return [row[n:] for row in a]


def _matvec(M: list[list[float]], v: list[float]) -> list[float]:
    return [sum(mij * vj for mij, vj in zip(row, v)) for row in M]


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


# ----- the policy ---------------------------------------------------------- #

class LinearBandit:
    def __init__(self, A: list[list[float]], b: list[float], *,
                 alpha: float = 0.3, feature_order: Sequence[str] = FEATURE_ORDER,
                 ridge: float = 1.0, n_updates: int = 0):
        self.A = A
        self.b = b
        self.alpha = alpha
        self.feature_order = tuple(feature_order)
        self.d = len(feature_order)
        self.ridge = ridge          # prior strength (A0 = ridge*I); lets health() report data gain
        self.n_updates = n_updates  # rewarded impressions folded in so far

    @classmethod
    def with_prior(cls, weights: dict, *, ridge: float = 1.0, alpha: float = 0.3,
                   feature_order: Sequence[str] = FEATURE_ORDER) -> "LinearBandit":
        """Ridge prior centred on the static fusion weights: A0 = ridge*I, b0 = ridge*w
        => theta0 = A0^-1 b0 = w. Starts identical to weighted fusion, then learns."""
        order = tuple(feature_order)
        d = len(order)
        A = [[ridge if i == j else 0.0 for j in range(d)] for i in range(d)]
        b = [ridge * float(weights.get(name, 0.0)) for name in order]
        return cls(A, b, alpha=alpha, feature_order=order, ridge=ridge)

    def features(self, per: dict) -> list[float]:
        return feature_vector(per, self.feature_order)

    def update(self, x: list[float], reward: float, *, weight: float = 1.0) -> None:
        """Online LinUCB update: A += w*x xT ; b += w*reward*x. `weight` < 1 down-weights
        a sample (e.g. bootstrap data from another study, so live data dominates later)."""
        for i in range(self.d):
            self.b[i] += weight * reward * x[i]
            xi, row = weight * x[i], self.A[i]
            for j in range(self.d):
                row[j] += xi * x[j]
        self.n_updates += 1

    def health(self) -> dict:
        """Training diagnostics: how much data each weight has seen and how confident it is.
        std[i] = posterior std of theta_i (sqrt of A^-1 diagonal) -> shrinks with data.
        data[i] = A_ii - ridge = total x_i^2 mass observed -> 0 means that feature never fired."""
        A_inv = _mat_inverse(self.A)
        return {
            "n_updates": self.n_updates,
            "ridge": self.ridge,
            "feature_order": list(self.feature_order),
            "std": [max(A_inv[i][i], 0.0) ** 0.5 for i in range(self.d)],
            "data": [self.A[i][i] - self.ridge for i in range(self.d)],
        }

    def theta(self) -> list[float]:
        return _matvec(_mat_inverse(self.A), self.b)

    def rank_scores(self, feats: dict[str, list[float]], *, explore: bool = True) -> dict[str, float]:
        """Score each candidate by θ·x (+ UCB bonus). Inverts A once for the batch."""
        A_inv = _mat_inverse(self.A)
        th = _matvec(A_inv, self.b)
        out: dict[str, float] = {}
        for cid, x in feats.items():
            mean = _dot(th, x)
            if explore and self.alpha > 0:
                var = max(_dot(x, _matvec(A_inv, x)), 0.0)
                out[cid] = mean + self.alpha * math.sqrt(var)
            else:
                out[cid] = mean
        return out

    def to_dict(self) -> dict:
        return {"A": self.A, "b": self.b, "alpha": self.alpha,
                "feature_order": list(self.feature_order),
                "ridge": self.ridge, "n_updates": self.n_updates}

    @classmethod
    def from_dict(cls, d: dict) -> "LinearBandit":
        return cls(d["A"], d["b"], alpha=d.get("alpha", 0.3),
                   feature_order=d.get("feature_order", FEATURE_ORDER),
                   ridge=d.get("ridge", 1.0), n_updates=d.get("n_updates", 0))
