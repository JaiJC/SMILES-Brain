"""Shared helpers for scoring candidate molecules against a target regional profile."""

from __future__ import annotations

import numpy as np

from smilesbrain.features import FingerprintConfig, morgan_fingerprint
from smilesbrain.models.uncertainty import ensemble_prediction_stats, tanimoto_to_training_set


def region_indices(spec: str | None, n_regions: int) -> list[int]:
    """Parse a comma-separated list of zero-based region indices."""
    if not spec:
        return []
    try:
        indices = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("region lists must be comma-separated integers") from exc
    if any(index < 0 or index >= n_regions for index in indices):
        raise ValueError(f"region indices must be between 0 and {n_regions - 1}")
    return sorted(set(indices))


def high_low_target(n_regions: int, high: list[int], low: list[int]) -> np.ndarray:
    """+1 on `high`, -1 on `low`, NaN (unconstrained) elsewhere."""
    if set(high) & set(low):
        raise ValueError("a region cannot be both high and low")
    target = np.full(n_regions, np.nan)
    target[high] = 1.0
    target[low] = -1.0
    return target


def predict_bundle(
    bundle: dict, smiles: list[str], n_receptors: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict affinities for SMILES with a fingerprint-model bundle (joblib artifact).

    Returns (predictions, per-receptor ensemble std, max training Tanimoto), each with one
    row/entry per SMILES. Unparseable SMILES give NaN rows.
    """
    fp = bundle.get("fp", FingerprintConfig())
    model = bundle["model"]
    training = bundle.get("training_X")
    predictions, uncertainty, similarities = [], [], []
    for smi in smiles:
        x = morgan_fingerprint(smi, fp)
        if x is None:
            predictions.append(np.full(n_receptors, np.nan))
            uncertainty.append(np.full(n_receptors, np.nan))
            similarities.append(np.nan)
            continue
        if "models" in bundle:
            ensemble = np.stack([member.predict(x[None, :]) for member in bundle["models"]])
            mean, std = ensemble_prediction_stats(ensemble)
            pred, unc = mean[0], std[0]
        else:
            pred = model.predict(x[None, :])[0]
            unc = np.full(n_receptors, np.nan)
        predictions.append(pred)
        uncertainty.append(unc)
        similarities.append(
            tanimoto_to_training_set(x, training) if training is not None else np.nan
        )
    return np.asarray(predictions), np.asarray(uncertainty), np.asarray(similarities)
