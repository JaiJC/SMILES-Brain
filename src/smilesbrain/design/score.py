"""Solve receptor weights for a desired regional brain profile and score molecules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetProfile:
    """A desired region profile and the receptor weights that best explain it."""

    receptor_weights: np.ndarray
    target: np.ndarray
    observed_regions: np.ndarray
    residual_norm: float
    regularization: float


def fit_target_profile(
    atlas: pd.DataFrame,
    target: np.ndarray,
    *,
    regularization: float = 1e-6,
) -> TargetProfile:
    """Fit receptor weights to a regional target using masked ridge least squares.

    `target` must have one value per atlas row; NaN entries are ignored. This lets callers
    specify only high/low region sets without inventing values for unspecified regions.
    """
    values = np.asarray(target, dtype=float)
    if values.shape != (len(atlas),):
        raise ValueError(f"target must have shape ({len(atlas)},), got {values.shape}")
    if not np.isfinite(regularization) or regularization < 0:
        raise ValueError("regularization must be finite and non-negative")
    observed = np.isfinite(values)
    if not observed.any():
        raise ValueError("target must contain at least one finite region value")
    matrix = atlas.to_numpy(dtype=float)
    if matrix.ndim != 2 or not np.isfinite(matrix[observed]).all():
        raise ValueError("atlas must be a finite 2-D matrix on target regions")
    design = matrix[observed]
    response = values[observed]
    gram = design.T @ design
    if regularization:
        gram = gram + regularization * np.eye(gram.shape[0])
    weights = np.linalg.solve(gram, design.T @ response)
    residual = design @ weights - response
    return TargetProfile(
        receptor_weights=weights,
        target=values.copy(),
        observed_regions=observed,
        residual_norm=float(np.linalg.norm(residual)),
        regularization=float(regularization),
    )


def score_affinity(
    affinity: np.ndarray | list[float],
    target_profile: TargetProfile,
) -> float:
    """Return a higher-is-better score for affinity matching solved receptor weights."""
    values = np.asarray(affinity, dtype=float)
    weights = target_profile.receptor_weights
    if values.shape != weights.shape:
        raise ValueError(f"affinity must have shape {weights.shape}, got {values.shape}")
    finite = np.isfinite(values) & np.isfinite(weights)
    if not finite.any():
        return float("-inf")
    error = values[finite] - weights[finite]
    return float(-np.mean(error**2))
