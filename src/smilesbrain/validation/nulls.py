"""Generate spatial-autocorrelation-preserving spin nulls for a parcellation.

A valid spin test rotates the parcels on the cortical sphere, so it needs each parcel's
spherical centroid and which hemisphere it belongs to. We compute permutation *indices*
(n_regions x n_perm) once and cache them to .npy; the harness then reuses them for every
drug, which is both correct and fast.

This leans on `netneurotools.stats.gen_spinsamples` (the standard implementation). It is an
optional dependency (`netneurotools>=0.2.5,<0.3`; 0.3 removed `stats.gen_spinsamples`): if it
is missing or too new, we raise with install guidance rather than silently
falling back to an invalid null. The naive permutation null lives in `spin.py` and is for
smoke tests only.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def gen_spins_from_coords(
    coords: np.ndarray,
    hemiid: np.ndarray,
    n_perm: int = 1000,
    seed: int = 0,
) -> np.ndarray:
    """Return an (n_regions x n_perm) array of spun region orderings.

    coords: (n_regions x 3) spherical centroid coordinates per parcel.
    hemiid: (n_regions,) with 0 for left, 1 for right hemisphere.
    """
    try:
        from netneurotools import stats as nnstats
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RuntimeError(
            "netneurotools>=0.2.5,<0.3 is required to generate spin nulls "
            "(`pip install 'netneurotools>=0.2.5,<0.3'`). The naive permutation null in spin.py is for "
            "smoke tests only and must not be reported as a spin test."
        ) from exc
    if not hasattr(nnstats, "gen_spinsamples"):  # pragma: no cover - env dependent
        raise RuntimeError(
            "installed netneurotools has no stats.gen_spinsamples (removed in 0.3); "
            "install 'netneurotools>=0.2.5,<0.3'."
        )
    spins = nnstats.gen_spinsamples(coords, hemiid, n_rotate=n_perm, seed=seed)
    return np.asarray(spins, dtype=int)


def save_spins(spins: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, spins)


def load_spins(path: str | Path, n_regions: int) -> np.ndarray:
    spins = np.load(path)
    if spins.shape[0] != n_regions:
        raise ValueError(
            f"spin file has {spins.shape[0]} regions, expected {n_regions}; "
            f"regenerate for the current parcellation."
        )
    return spins
