"""Evaluation harness: score predicted brain maps against observed drug maps.

For each drug we project its affinity profile through the atlas, correlate the predicted map
with the observed map, and spin-test the correlation. Two profile sources are supported:

  - "predicted": affinity from the Stage-1 model, from SMILES (the zero-shot claim).
  - "measured":  affinity from a PDSP/literature CSV (reproduces the classic measured-affinity
                 projection -- a sanity check the pipeline should pass before zero-shot is
                 trustworthy).

Also reported: whether the correlation's sign matches `expected_sign` (a cheap, honest prior
check), and how many receptors actually contributed to the projection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from smilesbrain.projection import project_affinity
from smilesbrain.validation.spin import spin_test

AffinityFn = Callable[[str], dict]


@dataclass(frozen=True)
class DrugResult:
    drug: str
    r: float
    p: float
    spin: bool
    n_perm: int
    n_receptors_used: int
    sign_ok: bool | None


def evaluate_one(
    drug: str,
    affinity: dict,
    observed_map: np.ndarray,
    atlas: pd.DataFrame,
    perm_indices: np.ndarray | None = None,
    weight_mode: str = "zscore",
    expected_sign: int | None = None,
) -> DrugResult:
    proj = project_affinity(affinity, atlas, weight_mode=weight_mode)
    stat = spin_test(proj.region_scores, observed_map, perm_indices=perm_indices)
    sign_ok = None if expected_sign is None else bool(np.sign(stat["r"]) == np.sign(expected_sign))
    return DrugResult(
        drug=drug,
        r=stat["r"],
        p=stat["p"],
        spin=stat["spin"],
        n_perm=stat["n_perm"],
        n_receptors_used=len(proj.receptors_used),
        sign_ok=sign_ok,
    )


def evaluate_catalog(
    catalog,
    affinity_fn: AffinityFn,
    atlas: pd.DataFrame,
    observed_maps: dict[str, np.ndarray],
    perm_indices: np.ndarray | None = None,
    weight_mode: str = "zscore",
) -> pd.DataFrame:
    """Evaluate every catalog drug that has an observed map available.

    affinity_fn:  SMILES -> {receptor_key: pChEMBL}.
    observed_maps: drug name -> observed region vector (already loaded/aligned).
    """
    rows: list[DrugResult] = []
    for entry in catalog:
        if entry.drug not in observed_maps:
            continue
        aff = affinity_fn(entry.smiles)
        rows.append(
            evaluate_one(
                drug=entry.drug,
                affinity=aff,
                observed_map=observed_maps[entry.drug],
                atlas=atlas,
                perm_indices=perm_indices,
                weight_mode=weight_mode,
                expected_sign=entry.expected_sign_value,
            )
        )
    if not rows:
        return pd.DataFrame(
            columns=["drug", "r", "p", "spin", "n_perm", "n_receptors_used", "sign_ok"]
        )
    return pd.DataFrame([r.__dict__ for r in rows])
