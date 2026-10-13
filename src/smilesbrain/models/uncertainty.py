"""Prediction uncertainty and fingerprint applicability-domain utilities."""

from __future__ import annotations

import numpy as np


def tanimoto_to_training_set(
    query: np.ndarray, training: np.ndarray, batch_size: int = 2048
) -> float:
    """Return the maximum Tanimoto similarity between one bit vector and training data."""
    if training.ndim != 2 or query.ndim != 1 or training.shape[1] != query.shape[0]:
        raise ValueError("query and training must have compatible 1-D/2-D shapes")
    query = query.astype(bool, copy=False)
    if len(training) == 0:
        raise ValueError("training fingerprint matrix must not be empty")
    best = 0.0
    query_count = int(query.sum())
    for start in range(0, len(training), batch_size):
        block = training[start : start + batch_size].astype(bool, copy=False)
        intersection = np.logical_and(block, query).sum(axis=1)
        union = np.logical_or(block, query).sum(axis=1)
        similarity = np.divide(
            intersection,
            union,
            out=np.ones_like(intersection, dtype=float),
            where=union != 0,
        )
        if query_count == 0:
            similarity[block.sum(axis=1) != 0] = 0.0
        if len(similarity):
            best = max(best, float(similarity.max()))
    return best


def ensemble_prediction_stats(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return mean and sample standard deviation for ensemble predictions."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 3:
        raise ValueError("predictions must have shape (n_models, n_samples, n_receptors)")
    observed = np.isfinite(values)
    count = observed.sum(axis=0)
    totals = np.where(observed, values, 0.0).sum(axis=0)
    mean = np.divide(totals, count, out=np.full(count.shape, np.nan), where=count > 0)
    centered = np.where(observed, values - mean, 0.0)
    divisor = np.maximum(count - 1, 1)
    uncertainty = np.sqrt(
        np.divide(
            (centered**2).sum(axis=0),
            divisor,
            out=np.full(count.shape, np.nan),
            where=count > 0,
        )
    )
    return mean, uncertainty


def flag_out_of_domain(max_similarity: float, threshold: float = 0.3) -> bool:
    """Flag a molecule when its closest training fingerprint is below the threshold."""
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("applicability threshold must be between 0 and 1")
    return bool(max_similarity < threshold)
