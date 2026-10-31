"""Tests for the shared prediction service, surface helpers and the demo app."""

from pathlib import Path

import numpy as np
import pytest

from smilesbrain.viz.surface import labels_to_vertex_scores, render_fallback, symmetric_limit

NICOTINE = "CN1CCC[C@H]1c1cccnc1"
APP = Path(__file__).resolve().parents[1] / "app" / "app.py"


def test_labels_to_vertex_scores_maps_parcels_and_blanks_the_rest():
    scores = np.array([10.0, 20.0, 30.0])
    out = labels_to_vertex_scores(np.array([0, 1, 3, 2, 4, 1]), scores)
    np.testing.assert_array_equal(np.isnan(out), [True, False, False, False, True, False])
    np.testing.assert_allclose(out[[1, 2, 3, 5]], [10.0, 30.0, 20.0, 10.0])


def test_symmetric_limit():
    assert symmetric_limit(np.array([-3.0, 1.0, np.nan])) == 3.0
    assert symmetric_limit(np.zeros(4)) == 1.0
    assert symmetric_limit(np.array([np.nan])) == 1.0


def test_fallback_chart_is_written_and_labelled(tmp_path):
    pytest.importorskip("matplotlib")
    out = render_fallback(np.random.default_rng(0).standard_normal(100), tmp_path / "f.png")
    assert out.exists() and out.stat().st_size > 1000


def test_plot_map_rejects_misaligned_scores(tmp_path):
    import importlib.util

    import pandas as pd

    spec = importlib.util.spec_from_file_location(
        "plot_map", Path(__file__).resolve().parents[1] / "scripts" / "plot_map.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    good = tmp_path / "g.csv"
    pd.DataFrame({"region": range(100), "score": np.arange(100.0)}).to_csv(good, index=False)
    assert len(mod.load_scores(good, 100)) == 100
    with pytest.raises(SystemExit):
        mod.load_scores(good, 400)
    bad = tmp_path / "b.csv"
    pd.DataFrame({"region": list(range(99, -1, -1)), "score": np.arange(100.0)}).to_csv(
        bad, index=False)
    with pytest.raises(SystemExit):
        mod.load_scores(bad, 100)


def test_service_demo_prediction_and_bad_smiles():
    pytest.importorskip("rdkit")
    from smilesbrain.config import load_receptors
    from smilesbrain.service import load_atlas, load_model, predict_molecule

    rc = load_receptors()
    bundle, demo_model = load_model(Path("does/not/exist.joblib"), rc)
    atlas, demo_atlas = load_atlas(Path("does/not/exist.csv"), rc)
    assert demo_model and demo_atlas  # DEMO fallbacks are flagged, never silent
    pred = predict_molecule(NICOTINE, bundle, atlas)
    assert pred.region_scores.shape == (rc.n_regions,)
    assert list(pred.affinity) == rc.atlas_keys
    assert pred.projection.directions == {}
    with pytest.raises(ValueError):
        predict_molecule("not-a-smiles", bundle, atlas)


def test_app_renders_a_map_and_the_demo_and_limitations_banners(tmp_path):
    """Hermetic: forces DEMO mode by pointing at missing files, so it does not depend on whether
    this machine happens to have a trained artifact or real atlas on disk."""
    pytest.importorskip("rdkit")
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    sidebar = {t.label: t for t in at.sidebar.text_input}
    sidebar["Model artifact"].set_value(str(tmp_path / "none.joblib"))
    sidebar["Atlas CSV (blank = parcellation default)"].set_value(str(tmp_path / "none.csv"))
    at.run()
    assert not at.exception
    banners = " ".join(e.value for e in at.error) + " " + " ".join(w.value for w in at.warning)
    assert "DEMO MODE" in banners
    assert "binding affinity did not predict" in banners
    smiles_box = [t for t in at.text_input if t.label == "SMILES"][0]
    assert smiles_box.value == NICOTINE
    smiles_box.set_value("not-a-smiles").run()
    assert any("Could not parse" in e.value for e in at.error)
    smiles_box.set_value(NICOTINE).run()
    assert not at.exception and len(at.dataframe) >= 1
