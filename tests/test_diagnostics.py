"""Diagnostics tests: each must fire on a planted signal and stay quiet when there is none."""

import numpy as np
import pandas as pd
import pytest

from smilesbrain.config import load_receptors
from smilesbrain.data.synthetic import synthetic_atlas
from smilesbrain.projection import project_affinity
from smilesbrain.validation.diagnostics import (
    architecture_ceiling,
    atlas_collinearity,
    receptor_permutation_null,
    target_receptor_ranking,
)


@pytest.fixture
def rc():
    return load_receptors()


@pytest.fixture
def atlas(rc):
    return synthetic_atlas(rc, seed=0)


def test_target_ranking_finds_a_planted_target(atlas):
    """A map that IS the D2 density map must rank D2 first."""
    observed = atlas["D2"].to_numpy() + np.random.default_rng(0).normal(0, 0.05, len(atlas))
    out = target_receptor_ranking("demo", observed, atlas, {"D2": -1})
    assert out.best_target == "D2" and out.rank == 1 and out.r > 0.9
    assert out.ranking[0][0] == "D2"


def test_target_ranking_reports_a_poor_rank_honestly(atlas):
    """A map built from a different receptor must not flatter the named target."""
    observed = atlas["VAChT"].to_numpy()
    out = target_receptor_ranking("demo", observed, atlas, {"D2": -1})
    assert out.best_target == "D2"
    assert out.rank > 1                       # D2 is not the best explanation
    assert out.ranking[0][0] == "VAChT"


def test_target_ranking_handles_a_drug_with_no_known_target(atlas):
    out = target_receptor_ranking("demo", atlas["D2"].to_numpy(), atlas, None)
    assert out.best_target is None and out.rank is None and len(out.ranking) == atlas.shape[1]


def test_ceiling_is_high_for_a_map_inside_the_atlas_span(atlas):
    """An exact weighted sum of the density maps must be nearly perfectly recoverable."""
    rng = np.random.default_rng(1)
    w = rng.normal(size=atlas.shape[1])
    observed = atlas.to_numpy() @ w
    out = architecture_ceiling("demo", observed, atlas, n_boot=40)
    assert out.r_in_sample > 0.99
    assert out.r_cv > 0.95                     # generalises across regions, not just fitted


def test_ceiling_is_low_for_a_map_unrelated_to_the_atlas(atlas):
    """Pure noise must not be explainable out of fold, even with 19 free weights."""
    observed = np.random.default_rng(2).normal(size=len(atlas))
    out = architecture_ceiling("demo", observed, atlas, n_boot=40)
    assert out.r_cv < 0.5
    assert out.r_in_sample > out.r_cv          # in-sample is the overfit upper bound


def test_ceiling_tolerates_missing_regions(atlas):
    observed = atlas.to_numpy() @ np.ones(atlas.shape[1])
    observed[:5] = np.nan
    out = architecture_ceiling("demo", observed, atlas, n_boot=20)
    assert np.isfinite(out.r_cv) and out.r_cv > 0.9


def test_permutation_null_detects_a_true_receptor_assignment(atlas, rc):
    """When the map really is built from this affinity vector, the true order must win."""
    affinity = {k: float(v) for k, v in
                zip(rc.atlas_keys, np.random.default_rng(3).normal(6, 1.5, rc.n_receptors))}
    observed = project_affinity(affinity, atlas).region_scores
    out = receptor_permutation_null("demo", affinity, observed, atlas, n_perm=400)
    assert out.p < 0.01
    assert abs(out.r) > out.null_mean_abs_r


def test_permutation_null_rejects_an_irrelevant_assignment(atlas, rc):
    """An affinity vector unrelated to the map must not beat its own shuffles."""
    rng = np.random.default_rng(4)
    affinity = {k: float(v) for k, v in zip(rc.atlas_keys, rng.normal(6, 1.5, rc.n_receptors))}
    observed = rng.normal(size=len(atlas))
    out = receptor_permutation_null("demo", affinity, observed, atlas, n_perm=400)
    assert out.p > 0.05
    assert out.n_receptors == rc.n_receptors


def test_permutation_null_needs_three_receptors(atlas):
    obs = np.random.default_rng(0).normal(size=len(atlas))
    assert receptor_permutation_null("demo", {"D2": 7.0, "DAT": 6.0}, obs, atlas, n_perm=10) is None
    # NaN and off-atlas entries are dropped before the count
    assert receptor_permutation_null("demo", {"D2": 7.0, "DAT": np.nan, "NOPE": 5.0},
                                     obs, atlas, n_perm=10) is None
    assert receptor_permutation_null("demo", {"D2": 7.0, "DAT": np.nan, "MU": 5.0, "H3": 6.0},
                                     obs, atlas, n_perm=10) is not None


def test_permutation_null_returns_none_on_a_degenerate_map(atlas):
    """A constant observed map has no defined correlation; say so instead of emitting nan."""
    assert receptor_permutation_null("demo", {"D2": 7.0, "MU": 5.0, "H3": 6.0},
                                     np.zeros(len(atlas)), atlas, n_perm=10) is None


def test_collinearity_separates_independent_from_duplicated_maps(rc):
    rng = np.random.default_rng(5)
    indep = pd.DataFrame(rng.normal(size=(100, 5)), columns=list("abcde"))
    base = rng.normal(size=100)
    dup = pd.DataFrame({c: base + rng.normal(0, 0.05, 100) for c in "abcde"})
    assert atlas_collinearity(indep)["mean_abs_r"] < 0.25
    assert atlas_collinearity(dup)["mean_abs_r"] > 0.9
    assert atlas_collinearity(dup)["pc1_variance"] > 0.9
    assert atlas_collinearity(indep)["n_pairs"] == 10
