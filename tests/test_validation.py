"""Validation-harness tests (no RDKit/PyTorch/network)."""

import warnings

import numpy as np
import pytest

from smilesbrain.config import load_receptors
from smilesbrain.data.synthetic import (
    synthetic_atlas,
    synthetic_observed_map,
)
from smilesbrain.validation import load_catalog
from smilesbrain.validation.harness import evaluate_catalog, evaluate_one


@pytest.fixture
def rc():
    return load_receptors()


def test_catalog_loads_and_has_smiles():
    catalog = load_catalog()
    assert len(catalog) >= 3
    for entry in catalog:
        assert entry.smiles and entry.drug and entry.file
        assert entry.expected_sign in (None, "positive", "negative")


def test_expected_sign_value_mapping():
    catalog = {c.drug: c for c in load_catalog()}
    # at least one of each sign exists in the curated catalog
    signs = {c.expected_sign_value for c in catalog.values()}
    assert 1 in signs and -1 in signs


def _affinity(rc, seed):
    rng = np.random.default_rng(seed)
    return {k: float(5 + rng.normal()) for k in rc.atlas_keys}


def test_evaluate_one_recovers_signal(rc):
    atlas = synthetic_atlas(rc, seed=0)
    aff = _affinity(rc, seed=1)
    observed = synthetic_observed_map(atlas, aff, noise=0.3, seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = evaluate_one("demo", aff, observed, atlas, perm_indices=None,
                           expected_sign=1)
    # observed is the true projection plus modest noise -> strong positive correlation
    assert res.r > 0.7
    assert res.sign_ok is True
    assert res.spin is False
    assert 0.0 <= res.p <= 1.0
    assert res.n_receptors_used == rc.n_receptors


def test_evaluate_catalog_with_spin_indices(rc):
    atlas = synthetic_atlas(rc, seed=0)
    catalog = load_catalog()
    affinity_fns = {c.drug: _affinity(rc, seed=i) for i, c in enumerate(catalog)}

    def predict(smiles):
        # map SMILES back to a drug deterministically for the test
        for c in catalog:
            if c.smiles == smiles:
                return affinity_fns[c.drug]
        return _affinity(rc, seed=99)

    observed = {c.drug: synthetic_observed_map(atlas, affinity_fns[c.drug], noise=0.4, seed=i)
                for i, c in enumerate(catalog)}
    perms = np.column_stack([np.random.default_rng(k).permutation(rc.n_regions)
                             for k in range(200)])

    results = evaluate_catalog(catalog, predict, atlas, observed, perm_indices=perms)
    assert len(results) == len(observed)
    assert results["spin"].all()           # spin=True when perms supplied
    assert (results["r"] > 0.5).all()        # all should recover their own signal


def test_evaluate_catalog_skips_missing_maps(rc):
    atlas = synthetic_atlas(rc, seed=0)
    catalog = load_catalog()
    observed = {}  # none available
    results = evaluate_catalog(catalog, lambda s: _affinity(rc, 0), atlas, observed)
    assert len(results) == 0
    assert list(results.columns) == [
        "drug", "r", "p", "spin", "n_perm", "n_receptors_used", "sign_ok"
    ]
