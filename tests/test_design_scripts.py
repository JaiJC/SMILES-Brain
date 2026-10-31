"""End-to-end checks of scripts/design_screen.py and scripts/design_generate.py (needs RDKit)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("rdkit")
import joblib  # noqa: E402

from smilesbrain.config import load_receptors  # noqa: E402
from smilesbrain.data.synthetic import synthetic_atlas  # noqa: E402
from smilesbrain.service import load_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SEEDS = [
    "CN1CCC[C@H]1c1cccnc1",
    "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1",
    "Cc1ncc2CN=C(c3ccccc3F)c3cc(Cl)ccc3-n12",
]
TARGET = ["--high-regions", "1,2,3,4", "--low-regions", "95,96,97,98"]


@pytest.fixture(scope="module")
def inputs(tmp_path_factory):
    d = tmp_path_factory.mktemp("design")
    rc = load_receptors()
    bundle, _ = load_model(Path("missing.joblib"), rc)
    joblib.dump(bundle, d / "model.joblib")
    synthetic_atlas(rc, seed=2).to_csv(d / "atlas.csv")
    (d / "seeds.txt").write_text("\n".join(SEEDS) + "\n")
    return d


def _run(script, args, check=True):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          capture_output=True, text=True, check=check)


def _common(d):
    return ["--artifact", str(d / "model.joblib"), "--atlas", str(d / "atlas.csv"), *TARGET]


def test_screen_is_reproducible_ranked_and_flags_domain(inputs):
    outs = []
    for i in (1, 2):
        out = inputs / f"screen{i}.csv"
        _run("design_screen.py", ["--smiles-file", str(inputs / "seeds.txt"), *_common(inputs),
                                  "--out", str(out), "--seed", "3"])
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    df = pd.read_csv(inputs / "screen1.csv")
    assert df["score"].is_monotonic_decreasing and len(df) == len(SEEDS)
    assert df["score"].between(-1, 1).all()
    nicotine = df[df["smiles"] == SEEDS[0]].iloc[0]  # part of the synthetic training set
    assert not nicotine["out_of_domain"] and nicotine["max_training_tanimoto"] == pytest.approx(1.0)
    assert df[df["smiles"] == SEEDS[1]].iloc[0]["out_of_domain"]


def test_generate_is_reproducible_labelled_and_beats_random(inputs):
    outs = []
    for i in (1, 2):
        out = inputs / f"gen{i}.csv"
        done = _run("design_generate.py", ["--seed-smiles-file", str(inputs / "seeds.txt"),
                                           *_common(inputs), "--population", "12", "--generations", "5",
                                           "--top", "5", "--out", str(out), "--seed", "0"])
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert "UNVALIDATED HYPOTHESIS" in done.stdout
    df = pd.read_csv(inputs / "gen1.csv")
    assert set(df["status"]) == {"UNVALIDATED HYPOTHESIS"} and len(df) == 5
    assert not set(df["smiles"]) & set(SEEDS)  # proposals are novel, never the seeds
    assert df["score"].is_monotonic_decreasing
    line = next(ln for ln in done.stdout.splitlines() if ln.startswith("top-5 mean score"))
    evolved, random_ = (float(x.split("=")[1]) for x in line.split(":")[1].split())
    assert evolved > random_


@pytest.mark.parametrize("script", ["design_screen.py", "design_generate.py"])
def test_design_scripts_refuse_demo_mode(script, tmp_path):
    seeds = tmp_path / "s.txt"
    seeds.write_text(SEEDS[0] + "\n")
    flag = "--smiles-file" if script == "design_screen.py" else "--seed-smiles-file"
    done = _run(script, [flag, str(seeds), "--artifact", str(tmp_path / "none.joblib"),
                         "--atlas", str(tmp_path / "none.csv"), *TARGET], check=False)
    assert done.returncode != 0 and "refusing DEMO MODE" in (done.stderr + done.stdout)
