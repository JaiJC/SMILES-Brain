"""End-to-end pipeline tests that do not require RDKit or PyTorch.

They use synthetic features/atlas directly so CI stays light while still exercising the real
training, projection, and validation code paths and, crucially, column alignment.
"""

import numpy as np
import pytest

from smilesbrain.config import load_receptors
from smilesbrain.data.synthetic import synthetic_atlas, synthetic_labels
from smilesbrain.models.baseline import MaskedMultiTaskRidge
from smilesbrain.projection import project_affinity
from smilesbrain.validation import map_correlation, spin_test


@pytest.fixture
def rc():
    return load_receptors()


def test_config_has_19_receptors(rc):
    assert rc.n_receptors == 19
    assert len(set(rc.atlas_keys)) == 19
    assert rc.n_regions == 100


def test_baseline_learns_synthetic(rc):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 128)).astype("float32")
    Y = synthetic_labels(X, rc.n_receptors, missing_rate=0.5, seed=0)
    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys, alpha=0.1).fit(X[:240], Y[:240])
    pred = model.predict(X[240:])
    # On learnable synthetic data the model must beat predicting the train mean.
    j = rc.index_of(model.fitted_receptors[0])
    t = Y[240:, j]
    m = np.isfinite(t)
    train_mean = np.nanmean(Y[:240, j])
    base_err = np.mean((t[m] - train_mean) ** 2)
    model_err = np.mean((t[m] - pred[m, j]) ** 2)
    assert model_err < base_err


def test_prediction_shape_matches_receptors(rc):
    rng = np.random.default_rng(1)
    X = rng.normal(size=(100, 64)).astype("float32")
    Y = synthetic_labels(X, rc.n_receptors, seed=1)
    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys).fit(X, Y)
    pred = model.predict(X[:5])
    assert pred.shape == (5, rc.n_receptors)


def test_projection_alignment_and_shape(rc):
    atlas = synthetic_atlas(rc, seed=2)
    affinity = {k: float(i) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas, weight_mode="zscore")
    assert res.region_scores.shape == (rc.n_regions,)
    assert set(res.receptors_used) <= set(atlas.columns)


def test_projection_drops_nan_receptors(rc):
    atlas = synthetic_atlas(rc, seed=3)
    affinity = {k: (np.nan if i % 2 else 1.0) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas, weight_mode="raw")
    assert all(not np.isnan(affinity[k]) for k in res.receptors_used)
    assert np.isfinite(res.region_scores).all()


def test_spin_test_naive_warns_and_returns_p(rc):
    rng = np.random.default_rng(4)
    a = rng.normal(size=rc.n_regions)
    b = a + rng.normal(scale=0.1, size=rc.n_regions)
    with pytest.warns(RuntimeWarning):
        out = spin_test(a, b, perm_indices=None, n_perm=200)
    assert out["spin"] is False
    assert 0.0 <= out["p"] <= 1.0
    assert out["r"] > 0.8  # strongly correlated by construction


def test_map_correlation_handles_nan():
    a = np.array([1.0, 2.0, np.nan, 4.0])
    b = np.array([1.0, 2.1, 3.0, 3.9])
    r = map_correlation(a, b)
    assert np.isfinite(r) and r > 0.9
