"""P4-T1 (observed-map ingestion) and P4-T3 (spin nulls) tests, on tiny synthetic inputs."""

import json

import numpy as np
import pytest

from smilesbrain.validation import load_observed_map
from smilesbrain.validation.nulls import gen_spins_from_coords, load_spins

nib = pytest.importorskip("nibabel")
pytest.importorskip("nilearn")

from smilesbrain.data.parcellate import (  # noqa: E402
    parcellate_nifti,
    write_observed_map,
    write_provenance,
)


def _labels_img(n_regions=4):
    lab = np.zeros((8, 8, 8), dtype=np.int16)
    for k in range(n_regions):          # 2x8x8 slabs, labels 1..n
        lab[2 * k : 2 * k + 2] = k + 1
    return nib.Nifti1Image(lab, np.eye(4))


def _stat_img(values_per_label, nan_voxel=False):
    data = np.zeros((8, 8, 8), dtype=np.float32)
    for k, v in enumerate(values_per_label):
        data[2 * k : 2 * k + 2] = v
    if nan_voxel:
        data[0, 0, 0] = np.nan
    return nib.Nifti1Image(data, np.eye(4))


def test_parcellate_recovers_region_values_in_label_order():
    vals = parcellate_nifti(_stat_img([1.5, -2.0, 0.25, 3.0]), _labels_img(), 4)
    np.testing.assert_allclose(vals, [1.5, -2.0, 0.25, 3.0], rtol=1e-6)


def test_parcellate_ignores_nan_voxels_and_marks_empty_regions_nan():
    vals = parcellate_nifti(_stat_img([1.0, 2.0, 3.0, 4.0], nan_voxel=True), _labels_img(), 4)
    np.testing.assert_allclose(vals, [1.0, 2.0, 3.0, 4.0], rtol=1e-6)  # NaN is not read as 0
    allnan = np.full((8, 8, 8), np.nan, dtype=np.float32)
    allnan[2:4] = 5.0                                                    # only label 2 has data
    only2 = parcellate_nifti(nib.Nifti1Image(allnan, np.eye(4)), _labels_img(), 4)
    assert only2[1] == pytest.approx(5.0) and np.isnan(only2[[0, 2, 3]]).all()
    # label 4 absent from the label image -> NaN in slot 4, other slots stay aligned
    vals = parcellate_nifti(_stat_img([1.0, 2.0, 3.0, 4.0]), _labels_img(3), 4)
    np.testing.assert_allclose(vals[:3], [1.0, 2.0, 3.0], rtol=1e-6)
    assert np.isnan(vals[3])


def test_zero_as_missing_stops_masked_zeros_shrinking_the_mean():
    data = np.zeros((8, 8, 8), dtype=np.float32)
    data[0:2, :, :4] = 3.0                        # label 1: only half its voxels are in-mask
    img = nib.Nifti1Image(data, np.eye(4))
    assert parcellate_nifti(img, _labels_img(), 4)[0] == pytest.approx(1.5)
    flagged = parcellate_nifti(img, _labels_img(), 4, zero_as_missing=True)
    assert flagged[0] == pytest.approx(3.0) and np.isnan(flagged[1:]).all()


def test_parcellate_resamples_to_coarser_stat_grid():
    stat = nib.Nifti1Image(np.full((4, 4, 4), 7.0, dtype=np.float32), np.diag([2, 2, 2, 1]))
    vals = parcellate_nifti(stat, _labels_img(), 4)
    assert np.isfinite(vals).any() and np.nanmax(np.abs(vals - 7.0)) < 1e-5


def test_label_ids_outside_parcellation_rejected():
    with pytest.raises(ValueError, match="not the configured parcellation"):
        parcellate_nifti(_stat_img([1, 2, 3, 4]), _labels_img(4), 3)


def test_csv_accepted_by_harness_loader_and_provenance_written(tmp_path):
    vals = parcellate_nifti(_stat_img([1.0, 2.0, 3.0, 4.0]), _labels_img(3), 4)  # one NaN
    csv = write_observed_map(vals, tmp_path / "drug.csv", 4)
    loaded = load_observed_map(csv, 4)
    assert loaded.shape == (4,) and np.isnan(loaded[3])
    with pytest.raises(ValueError, match="expected 5"):
        load_observed_map(csv, 5)

    nii = tmp_path / "in.nii.gz"
    nib.save(_stat_img([1, 2, 3, 4]), nii)
    side = write_provenance(csv, vals, drug="demo", source="unit test", n_subjects=12,
                            contrast="drug > placebo", parcellation="toy", input_nifti=nii)
    rec = json.loads(side.read_text())
    assert (rec["source"], rec["n_subjects"], rec["contrast"]) == ("unit test", 12, "drug > placebo")
    assert rec["n_regions"] == 4 and rec["n_regions_missing"] == 1
    assert len(rec["input_sha256"]) == 64


def test_real_schaefer100_ids_match_atlas_row_count():
    from smilesbrain.config import load_receptors

    assert load_receptors().n_regions == 100  # label ids 1..100 <-> atlas rows 0..99


def test_spins_preserve_hemisphere_and_load_checks_region_count(tmp_path):
    pytest.importorskip("netneurotools")
    rng = np.random.default_rng(0)
    n = 40
    xyz = rng.normal(size=(n, 3))
    coords = 100 * xyz / np.linalg.norm(xyz, axis=1, keepdims=True)
    hemi = np.repeat([0, 1], n // 2)
    coords[hemi == 1, 0] = np.abs(coords[hemi == 1, 0])
    coords[hemi == 0, 0] = -np.abs(coords[hemi == 0, 0])
    spins = gen_spins_from_coords(coords, hemi, n_perm=50, seed=1)
    assert spins.shape == (n, 50)
    assert spins.min() >= 0 and spins.max() < n                # valid indices (not nec. 1:1)
    assert (hemi[spins] == hemi[:, None]).all()                # hemisphere preserved
    assert not (spins == np.arange(n)[:, None]).all()          # actually rotates
    path = tmp_path / "spins.npy"
    np.save(path, spins)
    assert load_spins(path, n).shape == (n, 50)
    with pytest.raises(ValueError, match="expected 41"):
        load_spins(path, 41)
