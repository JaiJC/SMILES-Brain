"""End-to-end run of scripts/learned_gains.py on synthetic inputs in the real file formats."""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")
import joblib  # noqa: E402

from smilesbrain.config import load_receptors  # noqa: E402
from smilesbrain.data.synthetic import _SMILES_BANK, synthetic_atlas, synthetic_labels  # noqa: E402
from smilesbrain.features import FingerprintConfig, featurize_many  # noqa: E402
from smilesbrain.models.baseline import MaskedMultiTaskRidge  # noqa: E402
from smilesbrain.projection.gains import occupancy, predict_map, standardize  # noqa: E402
from smilesbrain.validation import load_catalog  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    d = tmp_path_factory.mktemp("lgworld")
    rc = load_receptors()
    atlas = synthetic_atlas(rc, seed=3)
    atlas.to_csv(d / "atlas.csv")
    catalog = load_catalog()
    rng = np.random.default_rng(0)

    smiles = list(dict.fromkeys(list(_SMILES_BANK) + [e.smiles for e in catalog]))
    fp = FingerprintConfig()
    X, kept = featurize_many(smiles, fp)
    smiles = [smiles[i] for i in kept]
    Y = synthetic_labels(X, rc.n_receptors, missing_rate=0.4, seed=1)
    labels = pd.DataFrame(Y, index=smiles, columns=rc.atlas_keys)
    # give every catalog drug a measured profile of 10 receptors
    profiles = {}
    for e in catalog:
        keys = rng.choice(rc.atlas_keys, size=10, replace=False)
        profiles[e.drug] = {k: float(rng.uniform(5.5, 9.0)) for k in keys}
        labels.loc[e.smiles, :] = np.nan
        for k, v in profiles[e.drug].items():
            labels.loc[e.smiles, k] = v
    labels.to_csv(d / "labels.csv")

    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys, alpha=10.0).fit(X, Y)
    joblib.dump({"model": model, "fp": fp, "keys": rc.atlas_keys, "training_X": X}, d / "model.joblib")

    gains = rng.standard_normal(rc.n_receptors)
    (d / "data" / "validation").mkdir(parents=True)
    for e in catalog:
        clean = standardize(predict_map(occupancy(profiles[e.drug], rc.atlas_keys, 100.0), atlas, gains))
        pd.Series(clean + 0.3 * rng.standard_normal(clean.shape)).to_csv(
            d / e.file, index=False, header=False)
    (d / "docs").mkdir()
    shutil.copy(ROOT / "docs" / "PREREGISTRATION.md", d / "docs" / "PREREGISTRATION.md")
    return d, rc


def _run(d, extra=()):
    return subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "learned_gains.py"),
         "--artifact", str(d / "model.joblib"), "--atlas", str(d / "atlas.csv"),
         "--labels", str(d / "labels.csv"), "--n-perm", "200", *extra],
        cwd=d, capture_output=True, text=True)


def test_runner_end_to_end_recovers_planted_gains_and_records_prereg(world):
    """Plumbing, not power: detection power is measured by scripts/lodo_power.py. This planted
    world sits near the 4-of-8 threshold (3 at 200 shuffles, 4 at 2000), so the verdict itself is
    deliberately not asserted; the held-out correlations, which are not threshold-sensitive, are."""
    d, _ = world
    done = _run(d)
    assert done.returncode == 0, done.stderr[-2000:]
    report = (d / "artifacts" / "learned_gains_report.md").read_text()
    assert "Tier A (measured affinity, the gate): CRITERION" in report
    assert "SHA-256" in report and "no spin nulls supplied" in report
    table = pd.read_csv(d / "artifacts" / "learned_gains.csv")
    primary_a = table[(table.tier == "A") & (table.conc_nM == 100.0)
                      & (table.grouping == "leave-one-drug-out")]
    assert len(primary_a) == 8 and primary_a["r_model"].median() > 0.8
    assert primary_a["delta_vs_mean_map"].median() > 0.3
    assert {"A", "B"} == set(table["tier"]) and set(table["grouping"]) == {
        "leave-one-drug-out", "leave-class-out"}
    assert set(table["conc_nM"]) == {10.0, 100.0, 1000.0}
    assert (d / "artifacts" / "learned_gains_report_manifest.json").exists()
    antipsych = table[(table.grouping == "leave-class-out") & (table.drug == "haloperidol")]
    assert not {"risperidone", "olanzapine"} & set(antipsych.iloc[0]["train_drugs"].split(";"))


def test_runner_is_reproducible(world):
    d, _ = world
    _run(d)
    first = (d / "artifacts" / "learned_gains.csv").read_bytes()
    _run(d)
    assert (d / "artifacts" / "learned_gains.csv").read_bytes() == first


def test_runner_refuses_without_prereg_or_inputs(world, tmp_path):
    d, _ = world
    done = _run(d, ["--prereg", str(tmp_path / "missing.md")])
    assert done.returncode != 0 and "committed" in (done.stderr + done.stdout)
    done = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "learned_gains.py"),
         "--artifact", str(tmp_path / "none.joblib"), "--prereg", str(d / "docs" / "PREREGISTRATION.md")],
        cwd=tmp_path, capture_output=True, text=True)
    assert done.returncode != 0 and "refusing DEMO MODE" in (done.stderr + done.stdout)
