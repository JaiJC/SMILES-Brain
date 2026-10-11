"""Spatial statistics for comparing two brain maps.

Comparing brain maps with a naive Pearson p-value is wrong: neighbouring regions are
correlated, so the effective number of independent samples is far below the region count and
naive p-values are wildly anti-conservative. The field-standard fix is the *spin test*:
rotate one map on the cortical sphere many times to build a null that preserves spatial
autocorrelation, then ask where the real correlation falls in that null.

Proper spins need the parcellation's spherical coordinates (e.g. via `neuromaps`). To keep
this module dependency-light and testable, it accepts a precomputed permutation matrix
(n_regions x n_perm). If none is given it falls back to naive permutation and loudly flags
that the resulting p-value does NOT control for spatial autocorrelation -- use it only for
smoke tests, never for a result you would report.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.stats import pearsonr


def map_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 3:
        return float("nan")
    return float(pearsonr(a[mask], b[mask])[0])


def spin_test(
    a: np.ndarray,
    b: np.ndarray,
    perm_indices: np.ndarray | None = None,
    n_perm: int = 1000,
    seed: int = 0,
) -> dict:
    """Correlate maps `a` and `b` and p-value it against a spatial null.

    perm_indices: (n_regions x n_perm) int array of spun region orderings. When provided the
    p-value is a valid spin-test p-value. When None, a naive permutation null is used and a
    warning is emitted.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    r_obs = map_correlation(a, b)

    if perm_indices is None:
        warnings.warn(
            "No spin permutations supplied; using naive permutation. The p-value does NOT "
            "account for spatial autocorrelation and must not be reported as a spin test.",
            RuntimeWarning,
            stacklevel=2,
        )
        rng = np.random.default_rng(seed)
        null = np.array(
            [map_correlation(a, rng.permutation(b)) for _ in range(n_perm)]
        )
        spun = False
    else:
        null = np.array([map_correlation(a, b[perm_indices[:, i]])
                         for i in range(perm_indices.shape[1])])
        spun = True

    null = null[np.isfinite(null)]
    p = float((np.sum(np.abs(null) >= abs(r_obs)) + 1) / (null.size + 1))
    return {"r": r_obs, "p": p, "n_perm": int(null.size), "spin": spun}
