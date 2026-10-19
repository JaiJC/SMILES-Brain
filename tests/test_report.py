"""P4-T4: the zero-shot guarantee and the two-tier report plumbing (needs RDKit)."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("rdkit")

from smilesbrain.validation.report import (  # noqa: E402
    TierRow,
    assert_held_out,
    benjamini_hochberg,
    build_markdown,
    held_out_mask,
    label_keys,
    measured_affinity,
    plot_report,
    structure_keys,
)

HALOPERIDOL = "O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"
HALO_SALT = "Cl.O=C(CCCN1CCC(O)(c2ccc(Cl)cc2)CC1)c1ccc(F)cc1"             # hydrochloride
HALO_ANALOGUE = "O=C(CCCN1CCC(O)(c2ccc(Br)cc2)CC1)c1ccc(F)cc1"             # same scaffold, Br
UNRELATED = "CN1CCC[C@H]1c1cccnc1"                                          # nicotine
ISO = "C[C@@H]1C[C@H](C)CN1"                                                # arbitrary


def test_keys_use_parent_fragment_so_salts_match_free_base():
    a, b = structure_keys(HALOPERIDOL), structure_keys(HALO_SALT)
    assert (a.connectivity, a.scaffold) == (b.connectivity, b.scaffold)
    assert structure_keys(UNRELATED).scaffold != a.scaffold


def test_held_out_mask_removes_molecule_salt_and_scaffold_analogues_only():
    train = [HALOPERIDOL, HALO_SALT, HALO_ANALOGUE, UNRELATED, ISO]
    keep, n_mol, n_scaf = held_out_mask(label_keys(train), structure_keys(HALOPERIDOL))
    assert keep.tolist() == [False, False, False, True, True]
    assert (n_mol, n_scaf) == (2, 1)


def test_assert_held_out_passes_on_filtered_set_and_catches_leaks():
    train = [HALOPERIDOL, HALO_SALT, HALO_ANALOGUE, UNRELATED]
    keep, _, _ = held_out_mask(label_keys(train), structure_keys(HALOPERIDOL))
    assert_held_out([s for s, k in zip(train, keep) if k], HALOPERIDOL)   # no raise
    for leaky in ([UNRELATED, HALOPERIDOL], [UNRELATED, HALO_SALT], [UNRELATED, HALO_ANALOGUE]):
        with pytest.raises(AssertionError, match="zero-shot leak"):
            assert_held_out(leaky, HALOPERIDOL)


def test_measured_affinity_matches_stereo_only_when_query_defines_it():
    lsd = "CCN(CC)C(=O)[C@H]1CN([C@@H]2Cc3c[nH]c4cccc(c34)C2=C1)C"
    iso_lsd = "CCN(CC)C(=O)[C@@H]1CN([C@@H]2Cc3c[nH]c4cccc(c34)C2=C1)C"
    labels = pd.DataFrame({"D2": [6.0, 8.0, 5.0], "5HT2a": [np.nan, 9.0, 7.0]},
                          index=[lsd, "Cl." + lsd, iso_lsd])
    prof, n = measured_affinity(labels, label_keys(labels.index.tolist()), lsd)
    assert n == 2 and prof == {"D2": 7.0, "5HT2a": 9.0}      # salt matched, diastereomer not
    prof2, n2 = measured_affinity(labels, label_keys(labels.index.tolist()), HALOPERIDOL)
    assert (prof2, n2) == ({}, 0)


def test_benjamini_hochberg_known_values():
    np.testing.assert_allclose(benjamini_hochberg([0.01, 0.04, 0.03, 0.5]),
                               [0.04, 0.0533333, 0.0533333, 0.5], rtol=1e-5)


def _row(tier, drug, r, p, spin=True):
    rng = np.random.default_rng(0)
    return TierRow(tier, drug, "CBF", 20, 15, r, p, spin, 1000, 1, r > 0, n_train=100,
                   n_removed_molecule=2, n_removed_scaffold=1, max_tanimoto=0.4,
                   scaffold="c1ccccc1", scores=rng.normal(size=30), observed=rng.normal(size=30))


def test_report_markdown_and_figure_show_both_tiers(tmp_path):
    rows = [_row("A", "d1", 0.5, 0.01), _row("B", "d1", 0.2, 0.3), _row("A", "d2", -0.1, 0.6),
            _row("B", "d2", 0.3, 0.04, spin=False)]
    md = build_markdown(rows, {"parcellation": "schaefer100", "n_regions": 100,
                               "weight_mode": "zscore", "seed": 0})
    for needle in ("Tier A", "Tier B", "Zero-shot guarantee", "spin (1000)", "NAIVE permutation",
                   "magnitude-only", "Benjamini"):
        assert needle in md
    fig = plot_report(rows, tmp_path / "r.png")
    assert fig.exists() and fig.stat().st_size > 5000


def test_pluggable_zero_shot_backend_is_refit_per_drug(monkeypatch):
    """Tier B must call the supplied backend, refit on the chemotype-filtered training set."""
    import numpy as np
    import pandas as pd

    from smilesbrain.data.synthetic import synthetic_atlas
    from smilesbrain.config import load_receptors
    from smilesbrain.features import FingerprintConfig
    from smilesbrain.validation.report import two_tier_evaluation

    rc = load_receptors()
    atlas = synthetic_atlas(rc, seed=0)
    rng = np.random.default_rng(0)
    smiles = [HALOPERIDOL, HALO_SALT, HALO_ANALOGUE, UNRELATED, ISO,
              "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CC(=O)Oc1ccccc1C(=O)O"]
    labels = pd.DataFrame(rng.normal(6, 1, size=(len(smiles), rc.n_receptors)),
                          index=smiles, columns=rc.atlas_keys)

    class Entry:
        drug, smiles_, measure, file = "haloperidol", HALOPERIDOL, "CBF", ""
        expected_sign_value, direction = -1, {"D2": -1}

        def __init__(self):
            self.smiles = HALOPERIDOL

    seen = {}

    def refit(X, Y, train_smiles):
        seen["n"] = len(train_smiles)
        seen["smiles"] = list(train_smiles)
        return lambda smi, x: np.full(rc.n_receptors, 7.0)

    rows = two_tier_evaluation([Entry()], {"haloperidol": rng.normal(size=rc.n_regions)},
                               atlas, labels, FingerprintConfig(), 1.0, rc.atlas_keys,
                               refit=refit)
    assert seen["n"] == 4  # the drug, its salt and its scaffold analogue are gone
    assert HALOPERIDOL not in seen["smiles"] and HALO_ANALOGUE not in seen["smiles"]
    assert {r.tier for r in rows} == {"A", "A-dir", "B", "B-dir"}
    assert next(r for r in rows if r.tier == "B-dir").directed == {"D2": -1}
    # the direction-restricted tiers project through exactly the directed receptors
    assert next(r for r in rows if r.tier == "A-dir").n_receptors_used == 1
    assert next(r for r in rows if r.tier == "B").n_receptors_used == rc.n_receptors
