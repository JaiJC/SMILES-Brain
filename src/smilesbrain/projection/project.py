"""Stage 2: receptor affinities x receptor density atlas -> whole-brain effect map.

The core model is deliberately simple and interpretable:

    effect(region) = sum_r  weight_r * density(region, r)

where `density` is the z-scored Hansen atlas (region x receptor) and `weight_r` is derived
from the molecule's predicted affinity at receptor r. A region lights up when it is rich in
the receptors the molecule binds. This mirrors the published receptor-map regression approach
(Dukart 2018; Hansen 2022), but with weights coming from a *structure-predicted* affinity
vector instead of lab assays -- the novel part.

Receptors with no prediction (NaN) or no atlas map are dropped from the sum, and the set of
receptors actually used is returned so the caller can report coverage honestly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectionResult:
    region_scores: np.ndarray          # (n_regions,) predicted effect per region
    receptors_used: list[str]          # receptor keys that contributed
    weights: dict[str, float]          # weight per used receptor


def affinity_to_weights(
    affinity: dict[str, float], mode: str = "zscore"
) -> dict[str, float]:
    """Turn predicted pChEMBL values into projection weights.

    Modes:
      - "raw":     use pChEMBL directly.
      - "zscore":  standardise across receptors (default) so the map reflects the molecule's
                   *relative* receptor preference, not its overall potency.
      - "softmax": emphasise the dominant receptor(s); useful for selective compounds.
    """
    keys = [k for k, v in affinity.items() if v is not None and not np.isnan(v)]
    vals = np.array([affinity[k] for k in keys], dtype=float)
    if vals.size == 0:
        return {}
    if mode == "raw":
        w = vals
    elif mode == "zscore":
        w = (vals - vals.mean()) / (vals.std() + 1e-8)
    elif mode == "softmax":
        e = np.exp(vals - vals.max())
        w = e / e.sum()
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    return dict(zip(keys, w))


def project_affinity(
    affinity: dict[str, float],
    atlas: pd.DataFrame,
    weight_mode: str = "zscore",
) -> ProjectionResult:
    """Project one molecule's affinity vector through the atlas.

    `affinity` maps receptor atlas_key -> predicted pChEMBL.
    `atlas` is a (region x receptor) DataFrame with receptor atlas_keys as columns.
    """
    weights = affinity_to_weights(affinity, mode=weight_mode)
    used = [k for k in weights if k in atlas.columns]
    if not used:
        raise ValueError(
            "No overlap between predicted receptors and atlas columns. "
            f"predicted={list(weights)[:5]}..., atlas={list(atlas.columns)[:5]}..."
        )
    w = np.array([weights[k] for k in used], dtype=float)
    scores = atlas[used].to_numpy() @ w
    return ProjectionResult(
        region_scores=scores,
        receptors_used=used,
        weights={k: float(weights[k]) for k in used},
    )
