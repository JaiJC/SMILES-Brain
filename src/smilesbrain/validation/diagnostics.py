"""Diagnostics that explain *why* a validation result came out the way it did.

A spin test asks whether a predicted map is more like the observed map than a spatially
rotated version of itself. That is necessary but it does not say which part of the pipeline
is responsible when the answer is no. These three diagnostics separate the possibilities, and
between them they localise the failure:

1. `target_receptor_ranking` -- before any model, does the observed map even look like the
   density of the drug's own target receptor? If a D2 antagonist's map does not resemble the
   D2 map, nothing downstream can rescue it.

2. `architecture_ceiling` -- the best a weighted sum of the receptor maps could possibly do
   for this observed map, found by fitting the weights to the map itself and scoring them out
   of sample across regions. This is an oracle, not a result: it is the ceiling that any
   affinity vector, however perfect, is competing against. A low ceiling indicts the atlas or
   the maps; a high ceiling indicts the weights.

3. `receptor_permutation_null` -- the decisive test of the project's actual claim. It keeps
   the drug's affinity *values* and the real atlas, and only shuffles which receptor each
   value belongs to. If the true assignment is no better than a shuffled one, then knowing
   which receptor the drug binds carries no information about the map, independently of how
   good the affinity numbers are.

Why (3) and not simply reading the fitted weights: the 19 density maps are strongly collinear
(mean |r| ~ 0.34, PC1 ~ 38% of variance), so the fitted weight vector is unstable even where
the fitted *predictions* are stable. Bootstrapping the weights on real data gives correlations
as low as ~0.0 between resamples. Coefficients therefore cannot be interpreted one receptor at
a time, while (3) never looks at a coefficient.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from smilesbrain.projection import project_affinity
from smilesbrain.validation.spin import map_correlation, spin_test


# ------------------------------------------------------------------ 1. target receptor rank
@dataclass(frozen=True)
class TargetRank:
    drug: str
    best_target: str | None
    r: float
    rank: int | None          # 1 = strongest |r| of all receptors
    n_receptors: int
    p: float
    spin: bool
    ranking: list[tuple[str, float]]


def target_receptor_ranking(
    drug: str,
    observed: np.ndarray,
    atlas: pd.DataFrame,
    targets: dict[str, int] | None,
    perm_indices: np.ndarray | None = None,
) -> TargetRank:
    """Rank every receptor density map by |r| with the observed map; locate the drug's targets.

    `targets` is the drug's direction map (receptor -> +-1); only its keys are used here.
    """
    rs = {c: map_correlation(atlas[c].to_numpy(), observed) for c in atlas.columns}
    order = sorted(rs, key=lambda c: -abs(rs[c]))
    keys = [t for t in (targets or {}) if t in rs]
    if not keys:
        return TargetRank(drug, None, float("nan"), None, len(order), float("nan"), False,
                          [(c, rs[c]) for c in order])
    best = max(keys, key=lambda t: abs(rs[t]))
    stat = spin_test(atlas[best].to_numpy(), observed, perm_indices=perm_indices)
    return TargetRank(drug, best, rs[best], order.index(best) + 1, len(order),
                      stat["p"], stat["spin"], [(c, rs[c]) for c in order])


# ---------------------------------------------------------------------- 2. ceiling (oracle)
@dataclass(frozen=True)
class Ceiling:
    drug: str
    r_in_sample: float        # weights fit and scored on all regions: overfit upper bound
    r_cv: float               # weights fit out of fold, scored on held-out regions
    p_cv: float
    spin: bool
    weight_stability: float   # mean correlation between bootstrap weight vectors


def architecture_ceiling(
    drug: str,
    observed: np.ndarray,
    atlas: pd.DataFrame,
    perm_indices: np.ndarray | None = None,
    n_folds: int = 5,
    n_boot: int = 200,
    seed: int = 0,
) -> Ceiling:
    """Best achievable agreement for *any* weighting of the atlas, plus how stable it is.

    Ridge-regularised because the density maps are collinear; the penalty is chosen by inner
    cross-validation on the training regions only, so the held-out regions stay held out.
    """
    from sklearn.linear_model import RidgeCV
    from sklearn.model_selection import KFold

    X, y = atlas.to_numpy(), np.asarray(observed, dtype=float)
    ok = np.isfinite(y)
    X, y = X[ok], y[ok]
    alphas = np.logspace(-2, 3, 30)

    in_sample = map_correlation(RidgeCV(alphas=alphas).fit(X, y).predict(X), y)

    pred = np.full(y.shape, np.nan)
    for train, test in KFold(n_folds, shuffle=True, random_state=seed).split(X):
        pred[test] = RidgeCV(alphas=alphas).fit(X[train], y[train]).predict(X[test])
    full = np.full(ok.shape, np.nan)
    full[ok] = pred
    stat = spin_test(full, np.where(ok, observed, np.nan), perm_indices=perm_indices)

    rng = np.random.default_rng(seed)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        w, *_ = np.linalg.lstsq(X[idx], y[idx], rcond=None)
        boot.append(w)
    boot = np.array(boot)
    pairs = rng.integers(0, n_boot, (n_boot, 2))
    stability = float(np.mean([map_correlation(boot[a], boot[b]) for a, b in pairs if a != b]))

    return Ceiling(drug, float(in_sample), float(stat["r"]), float(stat["p"]), stat["spin"],
                   stability)


# --------------------------------------------------------- 3. receptor-permutation null
@dataclass(frozen=True)
class PermutationNull:
    drug: str
    r: float
    null_mean_abs_r: float
    p: float
    n_perm: int
    n_receptors: int


def receptor_permutation_null(
    drug: str,
    affinity: dict[str, float],
    observed: np.ndarray,
    atlas: pd.DataFrame,
    n_perm: int = 5000,
    weight_mode: str = "zscore",
    direction: dict[str, int] | None = None,
    seed: int = 0,
) -> PermutationNull | None:
    """Is the true receptor-to-affinity assignment better than a shuffled one?

    Keeps the affinity values and the atlas, permutes which receptor holds which value.
    Returns None when fewer than three receptors are available (nothing to permute), or when
    the comparison is degenerate (e.g. a constant observed map leaves no finite null).
    """
    usable = {k: v for k, v in affinity.items()
              if k in atlas.columns and v is not None and np.isfinite(v)}
    if len(usable) < 3:
        return None
    keys = list(usable)
    values = np.array([usable[k] for k in keys], dtype=float)
    rng = np.random.default_rng(seed)

    def score(profile):
        return map_correlation(
            project_affinity(profile, atlas, weight_mode=weight_mode,
                             direction=direction).region_scores,
            observed,
        )

    observed_r = score(usable)
    if not np.isfinite(observed_r):
        return None
    null = np.array([score(dict(zip(keys, rng.permutation(values)))) for _ in range(n_perm)])
    null = null[np.isfinite(null)]
    if null.size == 0:
        return None
    p = float((np.sum(np.abs(null) >= abs(observed_r)) + 1) / (null.size + 1))
    return PermutationNull(drug, float(observed_r), float(np.abs(null).mean()), p,
                           int(null.size), len(keys))


# ------------------------------------------------------------------------------ collinearity
def atlas_collinearity(atlas: pd.DataFrame) -> dict:
    """How interchangeable the receptor maps are -- the reason weights are uninterpretable."""
    X = atlas.to_numpy()
    corr = np.corrcoef(X.T)
    off = corr[~np.eye(corr.shape[0], dtype=bool)]
    centered = X - X.mean(axis=0)
    sv = np.linalg.svd(centered, compute_uv=False)
    var = sv**2 / (sv**2).sum()
    return {
        "mean_abs_r": float(np.abs(off).mean()),
        "pairs_above_0.5": int((np.abs(off) > 0.5).sum() // 2),
        "n_pairs": int(corr.shape[0] * (corr.shape[0] - 1) // 2),
        "pc1_variance": float(var[0]),
        "pc2_variance": float(var[1]) if var.size > 1 else float("nan"),
    }
