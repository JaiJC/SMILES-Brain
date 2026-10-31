"""Learned receptor gains and the leave-one-drug-out test: recovery, calibration, honesty."""

import numpy as np
import pytest

from smilesbrain.config import load_receptors
from smilesbrain.data.synthetic import synthetic_atlas
from smilesbrain.projection.gains import (
    design_matrix,
    fit_gains,
    occupancy,
    predict_map,
    select_ridge,
    standardize,
)
from smilesbrain.validation.lodo import PREREG, lodo_evaluate, rows_to_frame, summarise_lodo

N_DRUGS = 8


@pytest.fixture(scope="module")
def rc():
    return load_receptors()


def _profiles(rc, rng, n=N_DRUGS):
    out = {}
    for i in range(n):
        keys = rng.choice(rc.atlas_keys, size=int(rng.integers(8, 17)), replace=False)
        out[f"drug{i}"] = {k: float(rng.uniform(5.5, 9.0)) for k in keys}
    return out


def _planted_maps(rc, atlas, profiles, rng, noise, gains=None):
    gains = rng.standard_normal(rc.n_receptors) if gains is None else gains
    maps = {}
    for d, prof in profiles.items():
        clean = standardize(predict_map(occupancy(prof, rc.atlas_keys, 100.0), atlas, gains))
        maps[d] = clean + noise * rng.standard_normal(clean.shape)
    return maps


def _unrelated_maps(atlas, profiles, rng):
    """Maps a weighted sum of densities can explain, but with weights unrelated to affinity."""
    density = atlas.to_numpy()
    return {d: standardize(density @ rng.standard_normal(density.shape[1])) for d in profiles}


# ------------------------------------------------------------------------------- the model
def test_occupancy_follows_the_binding_isotherm(rc):
    keys = rc.atlas_keys
    occ = occupancy({keys[0]: 9.0, keys[1]: 6.0, keys[2]: float("nan")}, keys, conc_nM=1.0)
    assert occ[0] == pytest.approx(0.5)                   # pKi 9 = Ki 1 nM; C = Ki -> half
    assert occ[1] == pytest.approx(1 / (1 + 1000))        # Ki 1000 nM at 1 nM
    assert (occ[2:] == 0).all()                           # missing / NaN -> zero, not dropped
    assert occupancy({keys[1]: 6.0}, keys, 1000.0)[1] > occ[1]   # more drug, more bound
    with pytest.raises(ValueError):
        occupancy({}, keys, 0.0)


def test_fit_gains_recovers_planted_gains_without_noise(rc):
    rng = np.random.default_rng(0)
    atlas = synthetic_atlas(rc, seed=1)
    true = rng.standard_normal(rc.n_receptors)
    profiles = _profiles(rc, rng, 10)
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.0, gains=true)
    designs = [design_matrix(occupancy(p, rc.atlas_keys, 100.0), atlas) for p in profiles.values()]
    est = fit_gains(designs, list(maps.values()), ridge=1e-8)
    # targets are standardised per drug, so gains are identified up to per-drug scale, not exactly
    scale = np.linalg.lstsq(true[:, None], est, rcond=None)[0][0]
    assert scale > 0 and np.corrcoef(est, true)[0, 1] > 0.9


def test_fit_and_select_validate_inputs(rc):
    atlas = synthetic_atlas(rc, seed=1)
    with pytest.raises(ValueError):
        fit_gains([], [], 1.0)
    with pytest.raises(ValueError):
        fit_gains([np.zeros((3, 3))], [np.zeros(3)], -1.0)
    d = design_matrix(np.ones(rc.n_receptors), atlas)
    assert select_ridge([d, d], [np.arange(100.0)] * 2) == 1000.0   # <3 drugs -> most conservative
    assert np.isnan(standardize(np.array([np.nan, np.nan])).astype(float)).all()
    assert (standardize(np.ones(5)) == 0).all()


# ----------------------------------------------------------------- the evaluation + nulls
def test_planted_signal_is_recovered_and_criterion_is_met(rc):
    rng = np.random.default_rng(3)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng)
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.3)
    rows, skipped = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=500)
    s = summarise_lodo(rows)
    assert not skipped and s["n_drugs"] == N_DRUGS
    assert s["median_r_model"] > 0.6 and s["median_delta_vs_mean_map"] > 0
    assert s["n_beating_null"] >= 6 and s["verdict"] == "CRITERION MET"


def test_unrelated_maps_do_not_pass_the_criterion(rc):
    """High ceiling, no affinity information: the real-data situation. Must not look like a win."""
    atlas = synthetic_atlas(rc, seed=4)
    verdicts, false_pos = [], []
    for seed in range(20):
        rng = np.random.default_rng(100 + seed)
        profiles = _profiles(rc, rng)
        maps = _unrelated_maps(atlas, profiles, rng)
        rows, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=300, seed=seed)
        verdicts.append(summarise_lodo(rows)["verdict"])
        false_pos += [r.p_perm < 0.05 for r in rows]
    assert "CRITERION MET" not in verdicts
    assert np.mean(false_pos) < 0.12          # nominal 5%; loose bound for a 160-drug sample


def test_common_pattern_without_drug_information_loses_to_the_mean_map(rc):
    """If every map is the same pattern, a model can score r > 0 without using the drug."""
    rng = np.random.default_rng(7)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng)
    common = standardize(atlas.to_numpy() @ rng.standard_normal(rc.n_receptors))
    maps = {d: common + 0.2 * rng.standard_normal(common.shape) for d in profiles}
    rows, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=300)
    s = summarise_lodo(rows)
    assert np.median([r.r_mean_map for r in rows]) > 0.9       # the baseline is strong
    assert s["median_delta_vs_mean_map"] <= 0 and s["verdict"] != "CRITERION MET"


def test_groups_exclude_same_group_drugs_from_training(rc):
    rng = np.random.default_rng(5)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng)
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.3)
    groups = {"drug0": "g", "drug1": "g", "drug2": "g"}
    rows, _ = lodo_evaluate(profiles, profiles, maps, atlas, groups=groups, n_perm=50)
    by = {r.drug: r for r in rows}
    assert not {"drug1", "drug2"} & set(by["drug0"].train_drugs)
    assert "drug0" not in by["drug0"].train_drugs and by["drug0"].n_train == N_DRUGS - 3
    assert by["drug5"].n_train == N_DRUGS - 1                  # ungrouped: leave-one-out


def test_tier_b_style_prediction_uses_a_different_test_profile(rc):
    rng = np.random.default_rng(6)
    atlas = synthetic_atlas(rc, seed=4)
    train = _profiles(rc, rng)
    maps = _planted_maps(rc, atlas, train, rng, noise=0.2)
    wrong = {d: {k: 12.0 - v for k, v in p.items()} for d, p in train.items()}   # inverted affinity
    right, _ = lodo_evaluate(train, train, maps, atlas, n_perm=50)
    bad, _ = lodo_evaluate(train, wrong, maps, atlas, n_perm=50, tier="B")
    assert np.median([r.r_model for r in right]) > np.median([r.r_model for r in bad])
    assert {r.tier for r in bad} == {"B"}


def test_skips_are_reported_not_silent(rc):
    rng = np.random.default_rng(8)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng, 5)
    profiles["thin"] = {rc.atlas_keys[0]: 8.0, rc.atlas_keys[1]: 7.0}
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.3)
    del maps["drug4"]
    rows, skipped = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=50)
    assert "thin" in skipped and "receptors" in skipped["thin"]
    assert skipped["drug4"] == "no observed map"
    assert {r.drug for r in rows} == {"drug0", "drug1", "drug2", "drug3"}
    few, sk = lodo_evaluate(profiles, profiles, maps, atlas, groups={f"drug{i}": "g" for i in range(4)})
    assert few == [] and all("training drugs" in v for k, v in sk.items() if k not in ("drug4", "thin"))


def test_too_few_drugs_is_inconclusive_not_a_verdict(rc):
    rng = np.random.default_rng(9)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng, 5)
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.1)
    rows, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=100)
    assert len(rows) < PREREG["min_evaluable_drugs"]
    assert summarise_lodo(rows)["verdict"].startswith("INCONCLUSIVE")
    assert summarise_lodo([])["verdict"].startswith("INCONCLUSIVE")


def test_spin_permutations_are_used_only_when_supplied_and_flagged(rc):
    rng = np.random.default_rng(10)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng)
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.3)
    plain, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=50)
    assert not any(r.spin for r in plain) and all(np.isnan(r.p_spin) for r in plain)
    perms = np.stack([np.roll(np.arange(rc.n_regions), k) for k in range(1, 40)], axis=1)
    spun, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=50, perm_indices=perms)
    assert all(r.spin and np.isfinite(r.p_spin) for r in spun)
    assert summarise_lodo(spun)["all_spin"] and not summarise_lodo(plain)["all_spin"]


def test_results_are_reproducible_and_tabulate(rc):
    rng = np.random.default_rng(11)
    atlas = synthetic_atlas(rc, seed=4)
    profiles = _profiles(rc, rng)
    maps = _planted_maps(rc, atlas, profiles, rng, noise=0.3)
    a, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=200, seed=1)
    b, _ = lodo_evaluate(profiles, profiles, maps, atlas, n_perm=200, seed=1)
    frame = rows_to_frame(a)
    assert frame.equals(rows_to_frame(b))       # NaN-safe (p_spin is NaN without spins)
    assert {"drug", "r_model", "r_mean_map", "p_perm", "delta_vs_mean_map", "train_drugs"} <= set(frame)
