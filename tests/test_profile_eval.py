"""Profile-level evaluation of the affinity model: metrics, baselines and cluster bootstrap."""

import numpy as np
import pytest

from smilesbrain.validation.profile_eval import (
    cluster_bootstrap_mean,
    knn_profile_predict,
    paired_cluster_bootstrap,
    spearman_rows,
    top1_rows,
)


def _data(n=400, k=10, seed=0, observed=0.6):
    rng = np.random.default_rng(seed)
    true = 6 + rng.standard_normal((n, k))
    mask = rng.random((n, k)) < observed
    true[~mask] = np.nan
    return true, rng


def test_spearman_rows_perfect_inverse_constant_and_sparse():
    true = np.array([[1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, np.nan],
                     [5.0, 5.0, 5.0, 5.0], [1.0, 2.0, np.nan, np.nan]])
    pred = np.array([[10.0, 20.0, 30.0, 40.0], [3.0, 2.0, 1.0, 9.0],
                     [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]])
    rho = spearman_rows(pred, true)
    assert rho[0] == pytest.approx(1.0)
    assert rho[1] == pytest.approx(-1.0)          # the unmeasured 4th receptor is ignored
    assert np.isnan(rho[2]) and np.isnan(rho[3])  # constant truth / only 2 measured
    with pytest.raises(ValueError):
        spearman_rows(pred[:, :3], true)


def test_shared_baseline_vector_broadcasts():
    true, _ = _data()
    base = np.arange(true.shape[1], dtype=float)
    assert np.array_equal(spearman_rows(base, true, 3), spearman_rows(np.tile(base, (len(true), 1)), true, 3),
                          equal_nan=True)


def test_informative_prediction_beats_noise_and_chance():
    true, rng = _data(1500, seed=1)
    good = np.where(np.isfinite(true), true + 0.5 * rng.standard_normal(true.shape), 6.0)
    noise = 6 + rng.standard_normal(true.shape)
    assert np.nanmean(spearman_rows(good, true)) > 0.6
    assert abs(np.nanmean(spearman_rows(noise, true))) < 0.05
    hit, chance = top1_rows(good, true)
    hit_n, chance_n = top1_rows(noise, true)
    assert np.nanmean(hit) > np.nanmean(chance) + 0.2
    assert abs(np.nanmean(hit_n) - np.nanmean(chance_n)) < 0.05


def test_top1_handles_ties_and_too_few_receptors():
    true = np.array([[3.0, 3.0, 1.0, np.nan], [1.0, 2.0, np.nan, np.nan]])
    pred = np.array([[0.0, 9.0, 1.0, 5.0], [1.0, 2.0, 3.0, 4.0]])
    hit, chance = top1_rows(pred, true)
    assert hit[0] == 1.0 and chance[0] == pytest.approx(2 / 3)    # tied winners 0 and 1
    assert np.isnan(hit[1]) and np.isnan(chance[1])


def test_receptor_mean_baseline_is_not_beaten_by_a_molecule_blind_model():
    """If every molecule truly has the same receptor ordering, a constant guess does as well as
    any model; a model must show it uses the molecule, so the *difference* must be ~0 here."""
    rng = np.random.default_rng(2)
    order = np.arange(10, dtype=float)
    true = order[None, :] + 0.3 * rng.standard_normal((800, 10))
    true[rng.random(true.shape) < 0.4] = np.nan
    baseline = spearman_rows(order, true)
    blind_model = spearman_rows(order + 0.001 * rng.standard_normal(true.shape), true)
    clusters = rng.integers(0, 80, size=len(true))
    mean, lo, hi = paired_cluster_bootstrap(blind_model, baseline, clusters, n_boot=500)
    assert lo <= 0 <= hi


def test_cluster_bootstrap_widens_when_rows_are_not_independent():
    rng = np.random.default_rng(3)
    cluster_effect = rng.standard_normal(40)
    clusters = np.repeat(np.arange(40), 25)
    values = cluster_effect[clusters] + 0.1 * rng.standard_normal(len(clusters))
    _, lo_c, hi_c = cluster_bootstrap_mean(values, clusters, n_boot=1000)
    _, lo_r, hi_r = cluster_bootstrap_mean(values, np.arange(len(values)), n_boot=1000)
    assert (hi_c - lo_c) > 3 * (hi_r - lo_r)       # treating analogues as independent is overconfident
    m, lo, hi = cluster_bootstrap_mean(np.array([np.nan, np.nan]), np.array([0, 1]))
    assert np.isnan(m)


def test_knn_baseline_copies_close_analogues_and_respects_missing_labels():
    labels = np.array([[8.0, np.nan], [6.0, 5.0], [4.0, np.nan], [np.nan, 9.0]])
    sim = np.array([[0.9, 0.1, 0.1, 0.1],       # nearest analogue is molecule 0
                    [0.1, 0.1, 0.1, 0.9]])      # nearest analogue is molecule 3
    out = knn_profile_predict(sim, labels, k=1)
    assert out[0, 0] == 8.0 and out[1, 1] == 9.0
    assert out[0, 1] == 5.0 or out[0, 1] == 9.0   # only molecules 1 and 3 measured receptor 1
    # k larger than the pool averages all measured molecules, weighted by similarity
    wide = knn_profile_predict(sim, labels, k=10)
    assert wide[0, 0] == pytest.approx((0.9 * 8 + 0.1 * 6 + 0.1 * 4) / 1.1)
    no_data = knn_profile_predict(sim, np.full((4, 2), np.nan), k=3)
    assert np.isnan(no_data).all()
    with pytest.raises(ValueError):
        knn_profile_predict(sim[:, :3], labels)


def test_knn_beats_receptor_mean_when_profiles_are_molecule_specific():
    rng = np.random.default_rng(4)
    n_clusters, per, k = 30, 20, 8
    centers = rng.standard_normal((n_clusters, k)) * 2 + 6
    cluster = np.repeat(np.arange(n_clusters), per)
    labels = centers[cluster] + 0.2 * rng.standard_normal((len(cluster), k))
    test_idx = np.arange(0, len(cluster), per)                       # one per cluster
    train_idx = np.setdiff1d(np.arange(len(cluster)), test_idx)
    sim = (cluster[test_idx][:, None] == cluster[train_idx][None, :]).astype(float) * 0.8 + 0.05
    pred = knn_profile_predict(sim, labels[train_idx], k=10)
    rho_knn = np.nanmean(spearman_rows(pred, labels[test_idx]))
    rho_mean = np.nanmean(spearman_rows(labels[train_idx].mean(axis=0), labels[test_idx]))
    assert rho_knn > 0.9 and rho_knn > rho_mean + 0.3
