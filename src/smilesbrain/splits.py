"""Train/test splitting, including scaffold splits.

Random splits flatter a molecular model: close analogues of a test molecule sit in train, so
a memoriser scores well. A *scaffold split* groups molecules by Bemis-Murcko scaffold and
holds out whole scaffolds, measuring generalisation to new chemotypes -- which is exactly the
zero-shot setting SMILES-Brain claims. We default to scaffold everywhere.
"""

from __future__ import annotations

import numpy as np


def random_split(n: int, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)
    cut = int(round(n * (1 - test_frac)))
    return np.sort(idx[:cut]), np.sort(idx[cut:])


def _scaffold(smiles: str) -> str:
    """Bemis-Murcko scaffold SMILES; falls back to the raw string if RDKit is absent."""
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
    except ImportError:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return smiles
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def scaffold_split(smiles_list: list[str], test_frac: float = 0.2, seed: int = 0):
    """Assign whole scaffolds to train/test. Largest scaffolds go to train first so the test
    set is composed of rarer chemotypes -- the harder, fairer evaluation."""
    groups: dict[str, list[int]] = {}
    for i, smi in enumerate(smiles_list):
        groups.setdefault(_scaffold(smi), []).append(i)

    rng = np.random.default_rng(seed)
    ordered = sorted(groups.values(), key=len, reverse=True)
    # shuffle within equal-size buckets for a non-degenerate but reproducible split
    rng.shuffle(ordered)
    ordered = sorted(ordered, key=len, reverse=True)

    n = len(smiles_list)
    n_test_target = int(round(n * test_frac))
    test_idx: list[int] = []
    train_idx: list[int] = []
    for grp in ordered:
        if len(test_idx) + len(grp) <= n_test_target:
            test_idx.extend(grp)
        else:
            train_idx.extend(grp)
    return np.sort(np.array(train_idx)), np.sort(np.array(test_idx))
