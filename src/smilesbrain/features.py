"""Molecular featurisation: SMILES -> fixed-length vector.

The baseline uses Morgan (ECFP-like) fingerprints via RDKit. RDKit is an optional
dependency so that config/projection code and CI stay importable without it; the import is
deferred and raises a clear message only when featurisation is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class RDKitUnavailable(RuntimeError):
    pass


def _require_rdkit():
    try:
        from rdkit import Chem  # noqa: F401
        from rdkit.Chem import AllChem  # noqa: F401
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")  # silence per-molecule parse warnings
        return Chem, AllChem
    except ImportError as exc:  # pragma: no cover - env dependent
        raise RDKitUnavailable(
            "RDKit is required for featurisation. Install with `pip install -e '.[chem]'`."
        ) from exc


@dataclass(frozen=True)
class FingerprintConfig:
    n_bits: int = 2048
    radius: int = 2


def canonicalize(smiles: str) -> str | None:
    """Return RDKit canonical SMILES, or None if the string does not parse."""
    Chem, _ = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def morgan_fingerprint(smiles: str, cfg: FingerprintConfig | None = None) -> np.ndarray | None:
    """SMILES -> binary Morgan fingerprint. None if the molecule does not parse."""
    cfg = cfg or FingerprintConfig()
    Chem, AllChem = _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bitvect = AllChem.GetMorganFingerprintAsBitVect(mol, cfg.radius, nBits=cfg.n_bits)
    arr = np.zeros((cfg.n_bits,), dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray

    ConvertToNumpyArray(bitvect, arr)
    return arr


def featurize_many(
    smiles_list: list[str], cfg: FingerprintConfig | None = None
) -> tuple[np.ndarray, list[int]]:
    """Featurise a list of SMILES.

    Returns the stacked feature matrix and the indices of rows that parsed successfully,
    so callers can drop the corresponding labels for molecules that failed.
    """
    cfg = cfg or FingerprintConfig()
    feats: list[np.ndarray] = []
    kept: list[int] = []
    for i, smi in enumerate(smiles_list):
        fp = morgan_fingerprint(smi, cfg)
        if fp is not None:
            feats.append(fp)
            kept.append(i)
    if not feats:
        return np.empty((0, cfg.n_bits), dtype=np.float32), []
    return np.vstack(feats), kept
