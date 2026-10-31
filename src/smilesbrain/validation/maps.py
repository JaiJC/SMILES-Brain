"""Load and align observed drug-response maps to the atlas parcellation.

An observed map is one value per parcel, stored as a CSV with either a single unnamed column
of values or columns [region, value]. We align by region index and validate the length
against the expected region count so a mismatched parcellation fails loudly rather than
silently correlating misaligned vectors.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_observed_map(path: str | Path, n_regions: int) -> np.ndarray:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Observed map not found: {path}. See configs/validation_maps.yaml `source` "
            f"and docs/VALIDATION.md for how to fetch and parcellate it."
        )
    df = pd.read_csv(path, header=None if _looks_headerless(path) else "infer")
    if df.shape[1] == 1:
        values = df.iloc[:, 0].to_numpy(dtype=float)
    elif {"region", "value"}.issubset({str(c).lower() for c in df.columns}):
        cols = {str(c).lower(): c for c in df.columns}
        df = df.sort_values(cols["region"])
        values = df[cols["value"]].to_numpy(dtype=float)
    else:
        # assume last column holds the values
        values = df.iloc[:, -1].to_numpy(dtype=float)

    if values.shape[0] != n_regions:
        raise ValueError(
            f"{path}: got {values.shape[0]} regions, expected {n_regions}. "
            f"The map must be in the same parcellation as the atlas."
        )
    return values


def _looks_headerless(path: Path) -> bool:
    with open(path) as fh:
        first = fh.readline().strip().split(",")[0]
    try:
        float(first)
        return True
    except ValueError:
        return False
