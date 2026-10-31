"""Affinity profiles for catalog drugs: measured (tier A) and zero-shot predicted (tier B).

These are the same two profile sources `report.two_tier_evaluation` uses, factored out so other
analyses (the learned-gains test) draw from exactly the same inputs and the same held-out
guarantee: for tier B, every training molecule sharing the drug's molecule or Bemis-Murcko
scaffold is removed and `assert_held_out` re-derives the keys from the SMILES actually used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smilesbrain.validation.report import (
    assert_held_out,
    held_out_mask,
    label_keys,
    measured_affinity,
    structure_keys,
)


def measured_profiles(catalog, labels: pd.DataFrame, receptor_keys: list[str]
                      ) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Median measured pChEMBL per receptor for each catalog drug. Returns (profiles, n_rows)."""
    keys = label_keys(labels.index.tolist())
    sub = labels[receptor_keys]
    profiles, n_rows = {}, {}
    for entry in catalog:
        profile, n = measured_affinity(sub, keys, entry.smiles)
        if profile:
            profiles[entry.drug] = profile
            n_rows[entry.drug] = n
    return profiles, n_rows


def zero_shot_profiles(catalog, labels: pd.DataFrame, fp_cfg, receptor_keys: list[str], refit,
                       drugs: set[str] | None = None) -> tuple[dict[str, dict[str, float]], dict[str, dict]]:
    """Predicted profile per drug from a model refit with the drug's chemotype removed.

    `refit(X, Y, smiles) -> fn(smiles, fingerprint) -> vector` is the backend factory from
    `report` (`ridge_refit`, `gbdt_refit`, `dmpnn_refit`). Returns (profiles, diagnostics)
    where diagnostics holds n_train, removals and the nearest remaining Tanimoto similarity.
    """
    from smilesbrain.features import featurize_many, morgan_fingerprint
    from smilesbrain.models.uncertainty import tanimoto_to_training_set

    smiles_all = labels.index.tolist()
    keys_all = label_keys(smiles_all)
    X_all, kept = featurize_many(smiles_all, fp_cfg)
    Y_all = labels[receptor_keys].to_numpy(dtype=float)[kept]
    keys_kept = [keys_all[i] for i in kept]
    smiles_kept = [smiles_all[i] for i in kept]

    profiles: dict[str, dict[str, float]] = {}
    diagnostics: dict[str, dict] = {}
    for entry in catalog:
        if drugs is not None and entry.drug not in drugs:
            continue
        drug_keys = structure_keys(entry.smiles)
        if drug_keys is None:
            raise ValueError(f"{entry.drug}: catalog SMILES does not parse")
        keep, n_mol, n_scaf = held_out_mask(keys_kept, drug_keys)
        train_smiles = [s for s, k in zip(smiles_kept, keep) if k]
        assert_held_out(train_smiles, entry.smiles)            # provable, not assumed
        predict_fn = refit(X_all[keep], Y_all[keep], train_smiles)
        x = morgan_fingerprint(entry.smiles, fp_cfg)
        pred = np.asarray(predict_fn(entry.smiles, x), dtype=float)
        profiles[entry.drug] = {k: float(v) for k, v in zip(receptor_keys, pred) if np.isfinite(v)}
        diagnostics[entry.drug] = {
            "n_train": int(keep.sum()), "n_removed_molecule": int(n_mol),
            "n_removed_scaffold": int(n_scaf),
            "max_tanimoto": float(tanimoto_to_training_set(x, X_all[keep])),
        }
    return profiles, diagnostics
