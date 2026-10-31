"""Scripts added at project close: NeuroVault scouting and the profile evaluation guard."""

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(script, args, cwd):
    return subprocess.run([sys.executable, str(ROOT / "scripts" / script), *args],
                          cwd=cwd, capture_output=True, text=True)


def _col(i, name, desc="", images=3):
    return {"id": i, "name": name, "description": desc, "DOI": None, "number_of_images": images,
            "number_of_experimental_units": None, "group_comparison": None,
            "coordinate_space": None, "publication_status": "p"}


def test_scouting_filters_locally_collapses_series_and_warns(tmp_path):
    cache = tmp_path / "c.json"
    series = [_col(100 + i, f"Nicotine meta analysis: included : 2025-01-0{i} 01:0", "nicotine")
              for i in range(5)]
    cache.write_text(json.dumps([
        _col(1, "Ketamine and placebo study", "ketamine vs placebo"),
        _col(2, "Methylphenidate task", "ritalin"),
        _col(3, "Unrelated visual cortex", "faces"),
        _col(4, "AkiNikolaidis's temporary collection", "ketamine"),    # temporary: excluded
        _col(5, "Empty ketamine collection", "ketamine", images=0),     # no images: excluded
        *series]))
    out = tmp_path / "s.md"
    done = _run("scout_neurovault.py", ["--cache", str(cache), "--out", str(out)], tmp_path)
    assert done.returncode == 0, done.stderr[-800:]
    md = out.read_text()
    row = {line.split("|")[1].strip(): [c.strip() for c in line.split("|")[2:5]]
           for line in md.splitlines() if line.startswith("| ") and line.count("|") == 5}
    assert row["ketamine"] == ["1", "1", "1"]            # one real hit, with placebo wording
    assert row["methylphenidate"][0] == "1" and row["psilocybin"][0] == "0"
    assert row["nicotine"][:2] == ["5", "1"]             # 5 collections, collapsed to 1 series
    assert "A hit is not a usable validation map" in md


def test_profile_eval_refuses_without_real_inputs(tmp_path):
    config = tmp_path / "cfg.yaml"
    config.write_text(
        "data: {labels: nowhere.parquet}\nfeatures: {n_bits: 2048, radius: 2}\n"
        "split: {scheme: scaffold, test_frac: 0.2, seed: 0}\n"
        "output: {artifact: nowhere.joblib}\n")
    done = _run("eval_profiles.py", ["--config", str(config)], tmp_path)
    assert done.returncode != 0 and "refusing to evaluate on synthetic data" in (
        done.stderr + done.stdout)
