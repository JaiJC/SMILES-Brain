"""Learned receptor gains: a shared physiological transform from occupancy to a brain map.

The validation work found that plain affinity magnitude is not the weight that turns receptor
density maps into a drug's CBF map, even though *some* weighting of the 19 densities reproduces
those maps well (ceiling r 0.46-0.81). Two things are wrong with using affinity directly:

1. **Affinity is not occupancy.** Binding at a given free drug concentration saturates:
   occupancy = C / (C + Ki). A 1 nM and a 0.1 nM binder are both ~fully occupied at 1 uM.
2. **Receptors do not all push blood flow the same way.** Blocking D2 and blocking 5-HT2A need
   not move CBF with the same sign or size. That is a property of the receptor, shared across
   drugs, so it can be *learned* across drugs rather than assumed to be +1 for every receptor.

So the model is   map(region) = sum_r  gain_r * occupancy_r * density(region, r)   with one
gain per receptor, shared by all drugs and fitted by ridge regression on other drugs' observed
maps. Nothing here knows about any particular drug; all drug-specific information enters
through the occupancy vector, which is why a permutation of that vector is a valid null.

Unmeasured receptors contribute zero occupancy. In a linear model that is identical to dropping
them, and it avoids the z-scoring trap in `affinity_to_weights`, where centring over whichever
receptors happened to be measured changes the map arbitrarily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DEFAULT_RIDGE_GRID = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def occupancy(profile: dict[str, float], receptor_keys: list[str], conc_nM: float) -> np.ndarray:
    """Fractional occupancy C / (C + Ki) per receptor; 0 where the profile has no value.

    `profile` maps receptor -> pKi/pChEMBL (-log10 molar). Ki in nM is 10**(9 - pKi).
    """
    if not np.isfinite(conc_nM) or conc_nM <= 0:
        raise ValueError("conc_nM must be a positive finite number")
    out = np.zeros(len(receptor_keys))
    for i, key in enumerate(receptor_keys):
        value = profile.get(key)
        if value is None or not np.isfinite(value):
            continue
        ki_nM = 10.0 ** (9.0 - float(value))
        out[i] = conc_nM / (conc_nM + ki_nM)
    return out


def standardize(values: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance over finite entries; NaN stays NaN. Constant input -> zeros."""
    v = np.asarray(values, dtype=float)
    finite = np.isfinite(v)
    out = np.full(v.shape, np.nan)
    if not finite.any():
        return out
    centred = v[finite] - v[finite].mean()
    scale = centred.std()
    out[finite] = centred / scale if scale > 1e-12 else 0.0
    return out


def design_matrix(occ: np.ndarray, atlas: pd.DataFrame) -> np.ndarray:
    """(n_regions x n_receptors): density columns scaled by this drug's occupancy."""
    density = atlas.to_numpy(dtype=float)
    if occ.shape != (density.shape[1],):
        raise ValueError(f"occupancy must have shape ({density.shape[1]},), got {occ.shape}")
    return density * occ[None, :]


def fit_gains(designs: list[np.ndarray], targets: list[np.ndarray], ridge: float) -> np.ndarray:
    """Ridge-fit one gain per receptor on stacked (drug, region) rows.

    Each target is standardised first, so drugs with larger CBF effect sizes do not dominate
    (which also means gains are identified only up to a per-drug scale, not exactly). Design
    columns are centred within each drug, because the standardised target has zero mean and the
    model has no intercept; correlation-based scoring is unaffected by that shift. Regions where
    the target is not finite are skipped.
    """
    if len(designs) != len(targets) or not designs:
        raise ValueError("need one target per design and at least one drug")
    if ridge < 0:
        raise ValueError("ridge must be non-negative")
    xs, ys = [], []
    for design, target in zip(designs, targets):
        y = standardize(target)
        ok = np.isfinite(y) & np.isfinite(design).all(axis=1)
        xs.append(design[ok] - design[ok].mean(axis=0))
        ys.append(y[ok])
    x, y = np.vstack(xs), np.concatenate(ys)
    gram = x.T @ x + ridge * np.eye(x.shape[1])
    return np.linalg.solve(gram, x.T @ y)


def predict_map(occ: np.ndarray, atlas: pd.DataFrame, gains: np.ndarray) -> np.ndarray:
    """Predicted region scores for a drug with occupancy vector `occ`."""
    return design_matrix(occ, atlas) @ gains


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 3 or np.std(a[ok]) < 1e-12 or np.std(b[ok]) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def select_ridge(
    designs: list[np.ndarray],
    targets: list[np.ndarray],
    grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
) -> float:
    """Pick the ridge strength by inner leave-one-drug-out on the *training* drugs only.

    Ties (and an all-NaN score) go to the larger, more conservative strength. With fewer than
    three training drugs there is nothing to cross-validate, so the largest value is used.
    """
    if len(designs) < 3:
        return float(max(grid))
    best, best_score = float(max(grid)), -np.inf
    for lam in sorted(grid, reverse=True):
        scores = []
        for held in range(len(designs)):
            rest = [i for i in range(len(designs)) if i != held]
            gains = fit_gains([designs[i] for i in rest], [targets[i] for i in rest], lam)
            scores.append(_corr(designs[held] @ gains, targets[held]))
        score = np.nanmean(scores) if np.isfinite(scores).any() else -np.inf
        if score > best_score + 1e-12:
            best, best_score = float(lam), float(score)
    return best
