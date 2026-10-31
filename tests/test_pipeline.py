"""End-to-end pipeline tests that do not require RDKit or PyTorch.

They use synthetic features/atlas directly so CI stays light while still exercising the real
training, projection, and validation code paths and, crucially, column alignment.
"""

import numpy as np
import pandas as pd
import pytest

from smilesbrain.config import load_receptors
from smilesbrain.data.assemble import merge_affinity_sources
from smilesbrain.data.fetch_pdsp import download_database, ki_to_pki, normalize_pdsp
from smilesbrain.data.synthetic import synthetic_atlas, synthetic_labels
from smilesbrain.models.baseline import MaskedMultiTaskRidge
from smilesbrain.models.trees import MaskedMultiTaskGBDT
from smilesbrain.models.uncertainty import (
    ensemble_prediction_stats,
    flag_out_of_domain,
    tanimoto_to_training_set,
)
from smilesbrain.projection import parse_direction_spec, project_affinity
from smilesbrain.design import fit_target_profile, score_affinity
from smilesbrain.validation import map_correlation, spin_test


@pytest.fixture
def rc():
    return load_receptors()


def test_config_has_19_receptors(rc):
    assert rc.n_receptors == 19
    assert len(set(rc.atlas_keys)) == 19
    assert rc.n_regions == 100


def test_affinity_source_merge_is_idempotent():
    chembl = pd.DataFrame(
        {"canonical_smiles": ["C", "CC"], "pchembl": [6.0, 7.0]}
    )
    pdsp = pd.DataFrame(
        {"canonical_smiles": ["C", "CCC"], "pchembl": [8.0, 5.0]}
    )
    once = merge_affinity_sources(chembl, pdsp)
    twice = merge_affinity_sources(
        once[["canonical_smiles", "pchembl"]], None
    )
    pd.testing.assert_frame_equal(
        once[["canonical_smiles", "pchembl"]],
        twice[["canonical_smiles", "pchembl"]],
    )


def test_pdsp_normalization_filters_target_and_converts_nm_to_pki():
    raw = pd.DataFrame(
        {
            "Receptor": ["Dopamine D2 receptor", "Dopamine D1 receptor"],
            "SMILES": ["CC", "CCC"],
            "Ki value": [10.0, 10.0],
            "Ki note": ["nM", "nM"],
        }
    )
    result = normalize_pdsp(raw, "D2")
    assert result["canonical_smiles"].tolist() == ["CC"]
    assert result["pchembl"].tolist() == [8.0]
    assert result["source"].tolist() == ["pdsp"]


def test_pdsp_rejects_implausible_pki_values():
    assert ki_to_pki("10", "nM") == pytest.approx(8.0)
    assert ki_to_pki("4", "M") is None
    assert ki_to_pki("1e-16", "M") is None


def test_pdsp_download_accepts_legacy_csv_encoding(monkeypatch):
    class Response:
        content = "Receptor,SMILES,Ki value,Ki note\nD2,CC,10,µM\n".encode("cp1252")

        def raise_for_status(self):
            return None

    monkeypatch.setattr(
        "smilesbrain.data.fetch_pdsp.requests.get",
        lambda *args, **kwargs: Response(),
    )
    result = download_database("https://example.invalid/pdsp.csv")
    assert result.loc[0, "Receptor"] == "D2"


def test_dmpnn_device_maps_to_chemprop_gpu_index():
    from smilesbrain.models.dmpnn import _gpu_index

    assert _gpu_index("cuda") == "0"
    assert _gpu_index("cuda:2") == "2"
    with pytest.raises(ValueError, match="unsupported"):
        _gpu_index("mps")


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


def test_gbdt_masks_missing_targets(rc):
    rng = np.random.default_rng(7)
    X = rng.normal(size=(80, 16))
    Y = synthetic_labels(X, rc.n_receptors, missing_rate=0.2, seed=7)
    model = MaskedMultiTaskGBDT(
        receptor_keys=rc.atlas_keys, max_iter=10, max_leaf_nodes=7
    ).fit(X, Y)
    assert model.predict(X[:3]).shape == (3, rc.n_receptors)
    assert model.fitted_receptors


def test_applicability_domain_and_ensemble_stats():
    training = np.array([[1, 0, 1], [0, 1, 0]], dtype=np.float32)
    assert tanimoto_to_training_set(np.array([1, 0, 1], dtype=np.float32), training) == 1.0
    assert flag_out_of_domain(0.1, threshold=0.3)
    means, uncertainty = ensemble_prediction_stats(
        np.array([[[1.0, 2.0]], [[3.0, 4.0]]])
    )
    np.testing.assert_allclose(means, [[2.0, 3.0]])
    np.testing.assert_allclose(uncertainty, [[np.sqrt(2.0), np.sqrt(2.0)]])


def test_applicability_domain_rejects_empty_training():
    with pytest.raises(ValueError, match="must not be empty"):
        tanimoto_to_training_set(np.array([1, 0]), np.empty((0, 2)))


def test_projection_alignment_and_shape(rc):
    atlas = synthetic_atlas(rc, seed=2)
    affinity = {k: float(i) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas, weight_mode="zscore")
    assert res.region_scores.shape == (rc.n_regions,)
    assert set(res.receptors_used) <= set(atlas.columns)


def test_inverse_design_exact_affinity_scores_best(rc):
    atlas = synthetic_atlas(rc, seed=12)
    target = np.full(rc.n_regions, np.nan)
    target[:5] = 1.0
    target[-5:] = -1.0
    profile = fit_target_profile(atlas, target)
    exact = score_affinity(profile.receptor_weights, profile)
    perturbed = score_affinity(profile.receptor_weights + 1.0, profile)
    assert exact > perturbed


def test_projection_drops_nan_receptors(rc):
    atlas = synthetic_atlas(rc, seed=3)
    affinity = {k: (np.nan if i % 2 else 1.0) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas, weight_mode="raw")
    assert all(not np.isnan(affinity[k]) for k in res.receptors_used)
    assert np.isfinite(res.region_scores).all()


def test_projection_defaults_to_magnitude_only_with_note(rc):
    atlas = synthetic_atlas(rc, seed=5)
    affinity = {k: float(i) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas)
    assert res.directions == {}
    assert "Magnitude-only" in res.note
    same = project_affinity(affinity, atlas, direction={})
    np.testing.assert_allclose(res.region_scores, same.region_scores)


@pytest.mark.parametrize("mode", ["raw", "zscore", "softmax"])
def test_flipping_direction_flips_that_receptors_contribution(rc, mode):
    atlas = synthetic_atlas(rc, seed=6)
    affinity = {k: 4.0 + 0.3 * i for i, k in enumerate(rc.atlas_keys)}
    all_agonist = {k: 1 for k in rc.atlas_keys}
    base = project_affinity(affinity, atlas, weight_mode=mode, direction=all_agonist)
    np.testing.assert_allclose(  # explicit all-agonist == magnitude-only
        base.region_scores, project_affinity(affinity, atlas, weight_mode=mode).region_scores
    )
    flipped_key = "D2"
    flipped = project_affinity(
        affinity, atlas, weight_mode=mode, direction={**all_agonist, flipped_key: -1}
    )
    contribution = base.weights[flipped_key] * atlas[flipped_key].to_numpy()
    np.testing.assert_allclose(flipped.region_scores, base.region_scores - 2 * contribution)
    assert flipped.weights[flipped_key] == -base.weights[flipped_key]
    assert flipped.direction_unknown == []
    # flipping every receptor negates the whole map
    all_antagonist = project_affinity(
        affinity, atlas, weight_mode=mode, direction={k: -1 for k in rc.atlas_keys}
    )
    np.testing.assert_allclose(all_antagonist.region_scores, -base.region_scores)


def test_partial_direction_reports_unknown_receptors(rc):
    atlas = synthetic_atlas(rc, seed=7)
    affinity = {k: float(i) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas, direction={"D2": -1})
    assert res.directions == {"D2": -1}
    assert "D2" not in res.direction_unknown and len(res.direction_unknown) == rc.n_receptors - 1
    assert "unknown" in res.note


def test_direction_validation(rc):
    atlas = synthetic_atlas(rc, seed=8)
    affinity = {k: 1.0 * i for i, k in enumerate(rc.atlas_keys)}
    with pytest.raises(ValueError, match="not in the atlas"):
        project_affinity(affinity, atlas, direction={"D2x": -1})
    with pytest.raises(ValueError, match=r"\+1 \(agonist\) or -1"):
        project_affinity(affinity, atlas, direction={"D2": 0.5})
    assert parse_direction_spec("5HT2a=+1, D2=antagonist,DAT=-1") == {
        "5HT2a": 1, "D2": -1, "DAT": -1}
    with pytest.raises(ValueError):
        parse_direction_spec("D2")


def test_both_parcellation_configs_load_and_wire_n_regions():
    base = load_receptors()
    assert base.parcellation.key == "schaefer100" and base.n_regions == 100
    for key, n, subcortex in [("schaefer100", 100, False), ("schaefer400", 400, False),
                              ("cammoun033", 83, True)]:
        rc = load_receptors(parcellation=key)
        assert (rc.n_regions, rc.parcellation.subcortex) == (n, subcortex)
        assert rc.atlas_keys == base.atlas_keys  # receptor order independent of parcellation
        scale = rc.parcellation.pet_dir.rsplit("/", 1)[-1].removeprefix("scale")
        assert rc.parcellation.atlas_file == f"data/atlas/density_scale{scale}.csv"
    with pytest.raises(ValueError, match="unknown parcellation"):
        load_receptors(parcellation="nope")


@pytest.mark.parametrize("key", ["schaefer100", "schaefer400", "cammoun033"])
def test_projection_runs_unchanged_on_each_parcellation(key):
    rc = load_receptors(parcellation=key)
    atlas = synthetic_atlas(rc, seed=1)
    affinity = {k: float(i) for i, k in enumerate(rc.atlas_keys)}
    res = project_affinity(affinity, atlas, direction={"D2": -1})
    assert res.region_scores.shape == (rc.n_regions,)
    assert len(atlas) == rc.n_regions


def _fake_atlas_downloads(monkeypatch, n=83):
    """Patch the network calls in fetch_atlas; record which files were requested."""
    from smilesbrain.data import fetch_atlas

    asked = []

    def fake_vec(fname, pet_dir):
        asked.append((pet_dir, fname))
        return np.random.default_rng(abs(hash(fname)) % 2**31).normal(size=n)

    monkeypatch.setattr(fetch_atlas, "_download_vector", fake_vec)
    return fetch_atlas, asked


def test_fetch_atlas_uses_parcellation_dir_and_requests_only_selected_tracers(monkeypatch):
    fetch_atlas, asked = _fake_atlas_downloads(monkeypatch, n=83)
    df = fetch_atlas.build_atlas(parcellation="cammoun033")
    assert df.shape == (83, 19)
    assert {d for d, _ in asked} == {"data/PET_parcellated/scale033"}
    requested = {f.removesuffix(".csv") for _, f in asked}
    expected = {stem for maps in fetch_atlas.TRACER_SELECTION.values() for stem, _ in maps}
    assert requested == expected
    # the tracers the paper discards must never be downloaded
    for discarded in ("D2_raclopride_hc7_alakurtti", "GABAa_flumazenil_hc6_dukart",
                      "5HT2a_mdl_hc3_talbot", "DAT_fepe2i_hc6_sasaki", "NAT_MRB_hc10_hesse",
                      "5HT1a_cumi_hc8_beliveau", "MU_carfentanil_hc39_turtonen"):
        assert discarded not in requested


def test_fetch_atlas_rejects_wrong_region_count(monkeypatch):
    fetch_atlas, _ = _fake_atlas_downloads(monkeypatch, n=100)
    with pytest.raises(RuntimeError, match="expects n_regions=83"):
        fetch_atlas.build_atlas(parcellation="cammoun033")


def test_atlas_filename_mapping_selects_paper_tracers_and_drops_the_rest():
    from smilesbrain.data.fetch_atlas import _key_from_filename

    assert _key_from_filename("GABAa-bz_flumazenil_hc16_norgaard.csv") == "GABAa"
    assert _key_from_filename("D2_flb457_hc37_smith.csv") == "D2"
    assert _key_from_filename("D2_raclopride_hc7_alakurtti.csv") is None   # n=7, discarded
    assert _key_from_filename("GABAa_flumazenil_hc6_dukart.csv") is None   # n=6, discarded


def test_multi_tracer_receptors_are_sample_size_weighted(monkeypatch):
    """D2 combines FLB457 n=37 and n=55; the n=55 map must carry more weight."""
    from smilesbrain.data import fetch_atlas

    a = np.linspace(-1, 1, 100)
    b = np.roll(a, 25)
    vecs = {"D2_flb457_hc37_smith.csv": a, "D2_flb457_hc55_sandiego.csv": b}
    monkeypatch.setattr(fetch_atlas, "_download_vector",
                        lambda f, d: vecs.get(f, np.random.default_rng(1).normal(size=100)))
    col = fetch_atlas.build_atlas(parcellation="schaefer100")["D2"].to_numpy()
    assert abs(np.corrcoef(col, b)[0, 1]) > abs(np.corrcoef(col, a)[0, 1])


def test_atlas_columns_are_zscored(monkeypatch):
    fetch_atlas, _ = _fake_atlas_downloads(monkeypatch, n=100)
    df = fetch_atlas.build_atlas(parcellation="schaefer100")
    assert np.allclose(df.mean(axis=0), 0, atol=1e-8)
    assert np.allclose(df.std(axis=0, ddof=0), 1, atol=1e-6)


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
