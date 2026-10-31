"""NaN-aware multi-task affinity baseline (pure scikit-learn, no GPU).

ChEMBL labels are sparse: most molecules are assayed against only a handful of the 19
receptors. A standard multi-output regressor cannot ingest NaN targets. This model sidesteps
that by fitting one estimator per receptor on *that receptor's observed rows only*, while all
receptors share the same input featurisation. It is the honest, fast baseline every later
model (D-MPNN, etc.) must beat.

Serialisable with joblib. Predicts a dense (n_molecules x n_receptors) matrix; receptors the
model never saw data for predict NaN so downstream code can skip them explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import Ridge


@dataclass
class MaskedMultiTaskRidge:
    receptor_keys: list[str]
    alpha: float = 1.0
    estimators_: dict[str, object] = field(default_factory=dict)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> "MaskedMultiTaskRidge":
        if Y.shape[1] != len(self.receptor_keys):
            raise ValueError(
                f"Y has {Y.shape[1]} columns but {len(self.receptor_keys)} receptor keys given"
            )
        base = Ridge(alpha=self.alpha)
        for j, key in enumerate(self.receptor_keys):
            col = Y[:, j]
            obs = ~np.isnan(col)
            if obs.sum() < 5:  # too few points to fit anything trustworthy
                continue
            est = clone(base)
            est.fit(X[obs], col[obs])
            self.estimators_[key] = est
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.full((X.shape[0], len(self.receptor_keys)), np.nan, dtype=float)
        for j, key in enumerate(self.receptor_keys):
            est = self.estimators_.get(key)
            if est is not None:
                out[:, j] = est.predict(X)
        return out

    @property
    def fitted_receptors(self) -> list[str]:
        return [k for k in self.receptor_keys if k in self.estimators_]
