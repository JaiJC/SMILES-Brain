"""Figure/number regeneration from saved artifacts (no model, no network)."""

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from smilesbrain.validation.summary import check_tiers, summarise_tiers, summary_markdown

ROOT = Path(__file__).resolve().parents[1]


def _tiers(spin=True):
    rows = []
    for tier, rs, ps in (("A", [-0.4, 0.1, 0.5], [0.01, 0.6, 0.02]),
                         ("B", [0.0, 0.2, -0.1], [0.9, 0.3, 0.5])):
        for i, (r, p) in enumerate(zip(rs, ps)):
            rows.append(dict(tier=tier, drug=f"d{i}", r=r, p=p, spin=spin, n_perm=1000,
                             sign_ok=[True, False, None][i], p_bh=min(1.0, p * 3)))
    return pd.DataFrame(rows)


def test_summarise_counts_are_computed_from_the_table():
    s = summarise_tiers(_tiers())
    assert s["A"]["n_drugs"] == 3 and s["A"]["n_significant"] == 2
    assert s["A"]["n_significant_bh"] == 1 and s["B"]["n_significant"] == 0
    assert (s["A"]["n_sign_correct"], s["A"]["n_sign_testable"]) == (1, 2)
    assert s["A"]["median_r"] == pytest.approx(0.1)
    assert s["A"]["all_spin_tests"] is True


def test_naive_null_is_labelled_in_the_summary():
    md = summary_markdown(summarise_tiers(_tiers(spin=False)))
    assert "NAIVE" in md
    assert "spin (1000 rotations)" in summary_markdown(summarise_tiers(_tiers()))


def test_missing_columns_or_rows_fail_loudly():
    with pytest.raises(ValueError):
        check_tiers(_tiers().drop(columns=["p_bh"]))
    with pytest.raises(ValueError):
        check_tiers(_tiers().iloc[0:0])


def test_make_figures_end_to_end_and_fails_without_inputs(tmp_path):
    pytest.importorskip("matplotlib")
    tiers = tmp_path / "t.csv"
    _tiers().to_csv(tiers, index=False)
    compare = tmp_path / "m.csv"
    pd.DataFrame({"backend": ["ridge", "gbdt"], "mean_r2_non_sparse": [0.367, 0.387]}).to_csv(
        compare, index=False)
    out = tmp_path / "figs"
    run = [sys.executable, str(ROOT / "scripts" / "make_figures.py"),
           "--tiers", str(tiers), "--model-compare", str(compare), "--out-dir", str(out)]
    done = subprocess.run(run, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
    for name in ("fig_validation_r.png", "fig_model_compare.png", "numbers.json", "numbers.md"):
        assert (out / name).stat().st_size > 0
    first = (out / "numbers.json").read_text()
    subprocess.run(run, capture_output=True, text=True, check=True)
    assert (out / "numbers.json").read_text() == first  # regeneration is deterministic

    missing = subprocess.run(run[:3] + [str(tmp_path / "nope.csv")] + run[4:],
                             capture_output=True, text=True)
    assert missing.returncode != 0 and "not found" in (missing.stderr + missing.stdout)
