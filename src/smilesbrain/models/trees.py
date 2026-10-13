"""NaN-aware tree baselines for sparse multi-receptor affinity labels."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass
class MaskedMultiTaskGBDT:
    """Fit one HistGradientBoostingRegressor per receptor on observed labels only."""

    receptor_keys: list[str]
    max_iter: int = 200
    learning_rate: float = 0.05
    max_leaf_nodes: int = 31
    l2_regularization: float = 0.0
    random_state: int = 0
    estimators_: dict[str, HistGradientBoostingRegressor] = field(default_factory=dict)

    def fit(self, X: np.ndarray, Y: np.ndarray) -> MaskedMultiTaskGBDT:
        if Y.ndim != 2 or Y.shape[1] != len(self.receptor_keys):
            raise ValueError(
                f"Y has {Y.shape[1]} columns but {len(self.receptor_keys)} receptor keys given"
            )
        for j, key in enumerate(self.receptor_keys):
            observed = np.isfinite(Y[:, j])
            if observed.sum() < 5:
                continue
            estimator = HistGradientBoostingRegressor(
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                max_leaf_nodes=self.max_leaf_nodes,
                l2_regularization=self.l2_regularization,
                random_state=self.random_state,
            )
            estimator.fit(X[observed], Y[observed, j])
            self.estimators_[key] = estimator
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = np.full((X.shape[0], len(self.receptor_keys)), np.nan, dtype=float)
        for j, key in enumerate(self.receptor_keys):
            estimator = self.estimators_.get(key)
            if estimator is not None:
                predictions[:, j] = estimator.predict(X)
        return predictions

    @property
    def fitted_receptors(self) -> list[str]:
        return [key for key in self.receptor_keys if key in self.estimators_]
