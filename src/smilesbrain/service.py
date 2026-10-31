"""Single code path from SMILES to a predicted affinity vector and region map.

Shared by `scripts/predict.py` and the demo app (`app/app.py`) so both always do the same
thing, including the DEMO fallback: when no trained artifact or real atlas exists, a tiny
synthetic model/atlas is used and the caller is told (`demo_model`, `demo_atlas`) so it can
show a banner. DEMO outputs carry no scientific meaning.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smilesbrain.config import ReceptorConfig
from smilesbrain.features import FingerprintConfig, morgan_fingerprint
from smilesbrain.models.uncertainty import (
    ensemble_prediction_stats,
    flag_out_of_domain,
    tanimoto_to_training_set,
)
from smilesbrain.projection import ProjectionResult, project_affinity

DEMO_BANNER = (
    "DEMO MODE: using synthetic {what}. Outputs are NOT scientifically meaningful. "
    "Train a real model and fetch the real atlas for genuine predictions."
)


def load_atlas(path: Path, rc: ReceptorConfig) -> tuple[pd.DataFrame, bool]:
    """Real atlas if present, else a synthetic one. Returns (atlas, is_demo)."""
    if Path(path).exists():
        return pd.read_csv(path, index_col=0), False
    from smilesbrain.data.synthetic import synthetic_atlas

    return synthetic_atlas(rc), True


def load_model(path: Path, rc: ReceptorConfig) -> tuple[dict, bool]:
    """Trained model bundle if present, else a freshly fit synthetic one. (bundle, is_demo)."""
    path = Path(path)
    if path.exists():
        if path.suffix == ".pt":
            from smilesbrain.models.dmpnn import MaskedDMPNNRegressor

            return {
                "model": MaskedDMPNNRegressor.load(path, rc.atlas_keys),
                "keys": rc.atlas_keys,
                "smiles_model": True,
            }, False
        return joblib.load(path), False
    from smilesbrain.data.synthetic import synthetic_labels, synthetic_smiles
    from smilesbrain.features import featurize_many
    from smilesbrain.models.baseline import MaskedMultiTaskRidge

    smiles = synthetic_smiles(300)
    X, _kept = featurize_many(smiles)
    Y = synthetic_labels(X, rc.n_receptors, seed=1)
    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys).fit(X, Y)
    return {
        "model": model,
        "fp": FingerprintConfig(),
        "keys": rc.atlas_keys,
        "training_X": X,
    }, True


@dataclass(frozen=True)
class MoleculePrediction:
    smiles: str
    keys: list[str]
    affinity: dict[str, float]            # receptor -> predicted pChEMBL (NaN if unavailable)
    uncertainty: np.ndarray               # per receptor, aligned with `keys` (NaN if unknown)
    max_training_tanimoto: float          # NaN if the artifact has no training fingerprints
    out_of_domain: bool
    projection: ProjectionResult

    @property
    def region_scores(self) -> np.ndarray:
        return self.projection.region_scores


def predict_molecule(
    smiles: str,
    bundle: dict,
    atlas: pd.DataFrame,
    *,
    direction: dict[str, int] | None = None,
    weight_mode: str = "zscore",
) -> MoleculePrediction:
    """SMILES -> affinity vector (+uncertainty, applicability domain) -> region map.

    Raises ValueError if the SMILES cannot be parsed.
    """
    model, keys = bundle["model"], bundle["keys"]
    max_similarity = np.nan
    if bundle.get("smiles_model"):
        pred, unc = model.predict_with_uncertainty([smiles])
        pred_vec, uncertainty_vec = pred[0], unc[0]
    else:
        x = morgan_fingerprint(smiles, bundle.get("fp"))
        if x is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        if "models" in bundle:
            ensemble = np.stack([member.predict(x[None, :]) for member in bundle["models"]])
            mean, std = ensemble_prediction_stats(ensemble)
            pred_vec, uncertainty_vec = mean[0], std[0]
        else:
            pred_vec = model.predict(x[None, :])[0]
            uncertainty_vec = np.full(len(keys), np.nan)
        training_X = bundle.get("training_X")
        if training_X is not None:
            max_similarity = tanimoto_to_training_set(x, training_X)
    affinity = {k: float(v) for k, v in zip(keys, pred_vec)}
    projection = project_affinity(affinity, atlas, weight_mode=weight_mode, direction=direction)
    return MoleculePrediction(
        smiles=smiles,
        keys=list(keys),
        affinity=affinity,
        uncertainty=np.asarray(uncertainty_vec, dtype=float),
        max_training_tanimoto=float(max_similarity),
        out_of_domain=bool(flag_out_of_domain(max_similarity))
        if np.isfinite(max_similarity) else False,
        projection=projection,
    )
