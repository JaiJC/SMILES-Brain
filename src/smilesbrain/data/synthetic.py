"""Synthetic data generators for offline development, tests, and CI.

NONE of this is scientifically meaningful. It exists so the full pipeline -- featurise,
train, project, validate -- can run end-to-end without network access or heavy real
downloads, and so tests can assert on shapes, alignment, and numerical behaviour
deterministically. Real runs use `fetch_affinity` + `fetch_atlas`.

The synthetic affinity labels are generated from the fingerprint via a fixed random linear
map, so a model *should* be able to learn them -- this lets a smoke test confirm training
actually reduces error, not just that it runs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from smilesbrain.config import ReceptorConfig

# A few real drug-like SMILES to make synthetic tables look realistic and parse under RDKit.
_SMILES_BANK = [
    "CN1CCC[C@H]1c1cccnc1",            # nicotine
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",    # caffeine
    "CC(=O)Oc1ccccc1C(=O)O",           # aspirin
    "CNC(C)Cc1ccccc1",                 # methamphetamine
    "OC(=O)C1=CC=CC=C1O",              # salicylic acid
    "CC(C)NCC(O)c1ccc(O)c(O)c1",       # isoprenaline
    "Clc1ccccc1C2=NCC(=O)Nc3ccc(cc23)[N+](=O)[O-]",  # clonazepam-like
    "COc1ccc2cc(ccc2c1)C(C)C(=O)O",    # naproxen
    "CN(C)CCCN1c2ccccc2Sc2ccccc21",    # promazine-like
    "OC1=CC=CC=C1",                    # phenol
]


def synthetic_smiles(n: int, seed: int = 0) -> list[str]:
    rng = np.random.default_rng(seed)
    return [_SMILES_BANK[i] for i in rng.integers(0, len(_SMILES_BANK), size=n)]


def synthetic_labels(
    features: np.ndarray, n_receptors: int, missing_rate: float = 0.6, seed: int = 0
) -> np.ndarray:
    """Learnable targets: y = features @ W + noise, with random entries masked to NaN."""
    rng = np.random.default_rng(seed)
    w = rng.normal(0, 1.0 / np.sqrt(features.shape[1]), size=(features.shape[1], n_receptors))
    y = features @ w + rng.normal(0, 0.1, size=(features.shape[0], n_receptors))
    y = 5.0 + 2.0 * (y - y.mean(0)) / (y.std(0) + 1e-8)  # push into a pChEMBL-ish range
    mask = rng.random(y.shape) < missing_rate
    y[mask] = np.nan
    return y


def synthetic_observed_map(
    atlas: pd.DataFrame,
    affinity: dict,
    noise: float = 0.5,
    seed: int = 0,
) -> np.ndarray:
    """A fake 'observed' drug map that is the true projection plus noise.

    Used only in tests: a correct harness should recover a positive, often significant,
    correlation between the predicted and this noised-true map.
    """
    from smilesbrain.projection import project_affinity

    rng = np.random.default_rng(seed)
    truth = project_affinity(affinity, atlas, weight_mode="zscore").region_scores
    return truth + rng.normal(0, noise * (truth.std() + 1e-8), size=truth.shape)


def synthetic_atlas(cfg: ReceptorConfig, seed: int = 0) -> pd.DataFrame:
    """Region x receptor density matrix with mild spatial smoothness per receptor."""
    rng = np.random.default_rng(seed)
    n = cfg.n_regions
    base = np.linspace(0, 1, n)[:, None]
    noise = rng.normal(0, 0.3, size=(n, cfg.n_receptors))
    loadings = rng.normal(0, 1, size=(1, cfg.n_receptors))
    mat = base * loadings + noise
    mat = (mat - mat.mean(0)) / (mat.std(0) + 1e-8)
    df = pd.DataFrame(mat, columns=cfg.atlas_keys)
    df.index.name = "region"
    return df
