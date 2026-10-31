"""Offline data paths added so the experiment can run without OSF/ChEMBL access."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from smilesbrain.config import load_receptors
from smilesbrain.data.assemble import assemble_from_merged

ROOT = Path(__file__).resolve().parents[1]


def _coords_module():
    spec = importlib.util.spec_from_file_location(
        "fetch_parcel_coords", ROOT / "scripts" / "fetch_parcel_coords.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sphere(n, seed):
    pts = np.random.default_rng(seed).standard_normal((n, 3))
    return 100.0 * pts / np.linalg.norm(pts, axis=1, keepdims=True)


def test_centroids_follow_netneurotools_surface_method():
    mod = _coords_module()
    names = [b"Background+FreeSurfer_Defined_Medial_Wall", b"A", b"B"]
    lh_s, rh_s = _sphere(30, 0), _sphere(30, 1)
    lh_l = np.array([0] * 6 + [1] * 12 + [2] * 12)
    rh_l = np.array([0] * 6 + [1] * 12 + [2] * 12)
    df = mod.centroids_from_annotations(lh_l, names, lh_s, rh_l, names, rh_s)
    assert list(df["hemi"]) == [0, 0, 1, 1]                       # medial wall dropped, LH first
    for row, (sphere, lab) in zip(df.itertuples(), [(lh_s, 1), (lh_s, 2), (rh_s, 1), (rh_s, 2)]):
        got = np.array([row.x, row.y, row.z])
        mean = sphere[(lh_l if sphere is lh_s else rh_l) == lab].mean(axis=0)
        expect = sphere[np.argmin(np.linalg.norm(sphere - mean, axis=1))]
        assert np.allclose(got, expect)                           # snapped to a real vertex
        assert np.isclose(np.linalg.norm(got), 100.0)             # and so lies on the sphere
    with pytest.raises(ValueError):
        mod.centroids_from_annotations(lh_l[:10], names, lh_s, rh_l, names, rh_s)


def test_assemble_from_merged_orders_columns_and_rejects_duplicates(tmp_path):
    rc = load_receptors()
    a, b = rc.atlas_keys[5], rc.atlas_keys[1]                      # deliberately out of order
    pd.DataFrame({"canonical_smiles": ["CCO", "CCN"], "pchembl": [6.0, 7.0],
                  "source": ["chembl", "pdsp"]}).to_csv(tmp_path / f"{a}.csv", index=False)
    pd.DataFrame({"canonical_smiles": ["CCN", "CCC"], "pchembl": [8.0, 5.0],
                  "source": ["chembl", "chembl"]}).to_csv(tmp_path / f"{b}.csv", index=False)
    table = assemble_from_merged(tmp_path)
    assert list(table.columns) == [k for k in rc.atlas_keys if k in (a, b)]   # canonical order
    assert table.loc["CCN", a] == 7.0 and table.loc["CCN", b] == 8.0
    assert np.isnan(table.loc["CCO", b]) and set(table.index) == {"CCO", "CCN", "CCC"}
    pd.DataFrame({"canonical_smiles": ["CCO", "CCO"], "pchembl": [6.0, 6.5]}).to_csv(
        tmp_path / f"{a}.csv", index=False)
    with pytest.raises(ValueError):
        assemble_from_merged(tmp_path)
    with pytest.raises(FileNotFoundError):
        assemble_from_merged(tmp_path / "nowhere")
