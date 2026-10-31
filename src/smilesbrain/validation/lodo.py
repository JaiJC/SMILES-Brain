"""Leave-one-drug-out test of the learned-gains model, with the nulls that keep it honest.

For each held-out drug the receptor gains are fitted only on *other* drugs' observed maps (and,
optionally, never on drugs of the same pharmacological group), then used to predict the held-out
drug's map from its occupancy vector alone. Three comparators decide whether that means anything:

* **Receptor-permutation null** (the decisive one, one-sided). Keep the drug's occupancy values
  and the fitted gains, shuffle only which receptor holds which value. If knowing *which*
  receptors a drug binds matters, the true assignment must beat its shuffles.
* **Mean-map baseline.** Predict the held-out map as the average of the training maps. Drug maps
  share a common pattern, so a model can score a positive r without using the drug at all; the
  model has to beat this to claim drug-specific information.
* **Spin null** (optional, needs spin permutations). Tests the map correlation against spatially
  rotated observed maps, as everywhere else in this repo. Never reported as a spin test without
  real permutations.

The success criterion is fixed in `docs/PREREGISTRATION.md` before any real data are run;
`PREREG` below mirrors its numbers and `scripts/learned_gains.py` refuses to run without it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import binom

from smilesbrain.projection.gains import (
    DEFAULT_RIDGE_GRID,
    design_matrix,
    fit_gains,
    occupancy,
    select_ridge,
    standardize,
)

PREREG = {
    "alpha": 0.05,
    "min_receptors": 3,        # a drug needs >= this many measured/predicted receptors
    "min_train_drugs": 3,
    "min_evaluable_drugs": 6,
    "min_beating_null": 4,     # and at least half of the evaluable drugs
    "primary_conc_nM": 100.0,
}


@dataclass(frozen=True)
class LodoRow:
    drug: str
    tier: str
    conc_nM: float
    n_receptors: int
    n_train: int
    train_drugs: tuple[str, ...]
    ridge: float
    r_model: float
    r_mean_map: float
    r_null_mean: float
    p_perm: float            # one-sided: P(shuffled r >= true r)
    n_perm: int
    p_spin: float            # one-sided; NaN without real spin permutations
    spin: bool

    @property
    def delta_vs_mean_map(self) -> float:
        return self.r_model - self.r_mean_map


def _corr_rows(preds: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Pearson r of each row of `preds` (k x n) with `target` (n), ignoring non-finite target."""
    ok = np.isfinite(target)
    t = target[ok] - target[ok].mean()
    p = preds[:, ok] - preds[:, ok].mean(axis=1, keepdims=True)
    denom = np.sqrt((p**2).sum(axis=1)) * np.sqrt((t**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (p @ t) / denom


def _one_sided_p(null: np.ndarray, observed: float) -> tuple[float, int]:
    null = null[np.isfinite(null)]
    if not np.isfinite(observed) or null.size == 0:
        return float("nan"), int(null.size)
    return float((np.sum(null >= observed) + 1) / (null.size + 1)), int(null.size)


def lodo_evaluate(
    train_profiles: dict[str, dict[str, float]],
    test_profiles: dict[str, dict[str, float]],
    observed: dict[str, np.ndarray],
    atlas: pd.DataFrame,
    *,
    conc_nM: float = PREREG["primary_conc_nM"],
    tier: str = "A",
    groups: dict[str, str] | None = None,
    n_perm: int = 5000,
    ridge_grid: tuple[float, ...] = DEFAULT_RIDGE_GRID,
    perm_indices: np.ndarray | None = None,
    seed: int = 0,
) -> tuple[list[LodoRow], dict[str, str]]:
    """Evaluate every drug that has an observed map and a test profile. Returns (rows, skipped).

    `train_profiles` are the *measured* affinities used to fit gains on the other drugs;
    `test_profiles` are what the held-out drug is predicted from (measured for tier A,
    structure-predicted with its chemotype held out for tier B).
    """
    keys = list(atlas.columns)
    groups = groups or {}
    group_of = lambda d: groups.get(d, d)  # noqa: E731 - tiny local helper
    n_min = PREREG["min_receptors"]
    rows: list[LodoRow] = []
    skipped: dict[str, str] = {}
    for drug in sorted(test_profiles):
        if drug not in observed:
            skipped[drug] = "no observed map"
            continue
        test = {k: v for k, v in test_profiles[drug].items()
                if k in keys and v is not None and np.isfinite(v)}
        if len(test) < n_min:
            skipped[drug] = f"only {len(test)} receptors in the test profile (need {n_min})"
            continue
        train = [d for d in sorted(train_profiles)
                 if d != drug and d in observed and group_of(d) != group_of(drug)
                 and sum(np.isfinite(v) for k, v in train_profiles[d].items() if k in keys) >= n_min]
        if len(train) < PREREG["min_train_drugs"]:
            skipped[drug] = f"only {len(train)} training drugs (need {PREREG['min_train_drugs']})"
            continue

        designs = [design_matrix(occupancy(train_profiles[d], keys, conc_nM), atlas) for d in train]
        targets = [observed[d] for d in train]
        lam = select_ridge(designs, targets, ridge_grid)
        gains = fit_gains(designs, targets, lam)

        obs = np.asarray(observed[drug], dtype=float)
        occ = occupancy(test, keys, conc_nM)
        density = atlas.to_numpy(dtype=float)
        scaled = density * gains[None, :]                       # regions x receptors
        pred = scaled @ occ
        r_model = float(_corr_rows(pred[None, :], obs)[0])
        mean_map = np.nanmean(np.vstack([standardize(observed[d]) for d in train]), axis=0)
        r_mean = float(_corr_rows(mean_map[None, :], obs)[0])

        rng = np.random.default_rng(seed + sum(map(ord, drug)))
        idx = np.array([keys.index(k) for k in test])
        values = occ[idx]
        shuffled = np.zeros((n_perm, len(keys)))
        shuffled[:, idx] = np.array([rng.permutation(values) for _ in range(n_perm)])
        null = _corr_rows(shuffled @ scaled.T, obs)
        p_perm, n_valid = _one_sided_p(null, r_model)

        if perm_indices is not None:
            spun = np.array([_corr_rows(pred[None, :], obs[perm_indices[:, i]])[0]
                             for i in range(perm_indices.shape[1])])
            p_spin, _ = _one_sided_p(spun, r_model)
        else:
            p_spin = float("nan")

        rows.append(LodoRow(
            drug=drug, tier=tier, conc_nM=float(conc_nM), n_receptors=len(test),
            n_train=len(train), train_drugs=tuple(train), ridge=lam, r_model=r_model,
            r_mean_map=r_mean, r_null_mean=float(np.nanmean(null)), p_perm=p_perm,
            n_perm=n_valid, p_spin=p_spin, spin=perm_indices is not None,
        ))
    return rows, skipped


def summarise_lodo(rows: list[LodoRow]) -> dict:
    """Apply the pre-registered criterion to one tier/concentration. Never says 'success' on
    too few drugs: that is reported as inconclusive, not as a pass or a fail."""
    n = len(rows)
    alpha = PREREG["alpha"]
    beating = sum(1 for r in rows if np.isfinite(r.p_perm) and r.p_perm < alpha)
    deltas = [r.delta_vs_mean_map for r in rows if np.isfinite(r.delta_vs_mean_map)]
    r_model = [r.r_model for r in rows if np.isfinite(r.r_model)]
    needed = max(PREREG["min_beating_null"], int(np.ceil(n / 2))) if n else None
    median_delta = float(np.median(deltas)) if deltas else float("nan")
    if n < PREREG["min_evaluable_drugs"]:
        verdict = "INCONCLUSIVE (too few evaluable drugs)"
    elif beating >= needed and median_delta > 0:
        verdict = "CRITERION MET"
    else:
        verdict = "CRITERION NOT MET"
    return {
        "n_drugs": n,
        "n_beating_null": beating,
        "needed": needed,
        "binomial_p_of_count": float(binom.sf(beating - 1, n, alpha)) if n else float("nan"),
        "median_r_model": float(np.median(r_model)) if r_model else float("nan"),
        "median_delta_vs_mean_map": median_delta,
        "n_spin_significant": sum(1 for r in rows if np.isfinite(r.p_spin) and r.p_spin < alpha),
        "all_spin": bool(rows) and all(r.spin for r in rows),
        "verdict": verdict,
    }


def rows_to_frame(rows: list[LodoRow]) -> pd.DataFrame:
    return pd.DataFrame([
        {**{k: v for k, v in vars(r).items() if k != "train_drugs"},
         "delta_vs_mean_map": r.delta_vs_mean_map, "train_drugs": ";".join(r.train_drugs)}
        for r in rows
    ])
