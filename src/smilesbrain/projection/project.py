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

Direction. Affinity says a molecule binds a receptor, not whether it activates or blocks it.
Callers may therefore pass an optional per-receptor functional sign (+1 agonist / -1
antagonist) which multiplies that receptor's weight, so flipping a sign exactly negates its
contribution to the map. Without it the projection is magnitude-only, and the result says so.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ProjectionResult:
    region_scores: np.ndarray          # (n_regions,) predicted effect per region
    receptors_used: list[str]          # receptor keys that contributed
    weights: dict[str, float]          # effective (signed) weight per used receptor
    directions: dict[str, int]         # +1/-1 applied per used receptor; empty if magnitude-only
    direction_unknown: list[str]       # used receptors with no supplied direction
    note: str                          # human-readable statement of what was (not) modelled


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


AGONIST, ANTAGONIST = 1, -1

_DIRECTION_WORDS = {"agonist": AGONIST, "antagonist": ANTAGONIST, "+": AGONIST, "-": ANTAGONIST}


def parse_direction_spec(spec: str) -> dict[str, int]:
    """Parse "5HT2a=+1,D2=-1" (also "D2=antagonist") into {atlas_key: +1/-1}."""
    out: dict[str, int] = {}
    for item in filter(None, (t.strip() for t in spec.split(","))):
        key, sep, val = item.partition("=")
        val = val.strip().lower()
        if not sep or not key.strip() or val == "":
            raise ValueError(f"bad direction entry {item!r}; expected KEY=+1|-1|agonist|antagonist")
        if val in _DIRECTION_WORDS:
            out[key.strip()] = _DIRECTION_WORDS[val]
        else:
            try:
                out[key.strip()] = _check_sign(key.strip(), float(val))
            except ValueError as exc:
                raise ValueError(f"bad direction value in {item!r}: {exc}") from None
    return out


def _check_sign(key: str, value: float) -> int:
    if value not in (AGONIST, ANTAGONIST):
        raise ValueError(f"direction for {key!r} must be +1 (agonist) or -1 (antagonist), got {value!r}")
    return int(value)


def project_affinity(
    affinity: dict[str, float],
    atlas: pd.DataFrame,
    weight_mode: str = "zscore",
    direction: dict[str, int] | None = None,
) -> ProjectionResult:
    """Project one molecule's affinity vector through the atlas.

    `affinity` maps receptor atlas_key -> predicted pChEMBL.
    `atlas` is a (region x receptor) DataFrame with receptor atlas_keys as columns.
    `direction` optionally maps atlas_key -> +1 (agonist) / -1 (antagonist). Weights are
    computed from affinity first and the sign is applied afterwards, so flipping one
    receptor's sign negates exactly that receptor's contribution. Receptors used but absent
    from `direction` keep their magnitude-only (+1) weight and are listed in
    `direction_unknown`. Keys in `direction` that are not atlas columns raise (typo guard).
    """
    weights = affinity_to_weights(affinity, mode=weight_mode)
    used = [k for k in weights if k in atlas.columns]
    if not used:
        raise ValueError(
            "No overlap between predicted receptors and atlas columns. "
            f"predicted={list(weights)[:5]}..., atlas={list(atlas.columns)[:5]}..."
        )

    signs: dict[str, int] = {}
    unknown = list(used)
    if direction:
        stray = sorted(set(direction) - set(atlas.columns))
        if stray:
            raise ValueError(f"direction given for receptors not in the atlas: {stray}")
        signs = {k: _check_sign(k, direction[k]) for k in used if k in direction}
        unknown = [k for k in used if k not in signs]

    w = np.array([weights[k] * signs.get(k, 1) for k in used], dtype=float)
    scores = atlas[used].to_numpy() @ w
    return ProjectionResult(
        region_scores=scores,
        receptors_used=used,
        weights=dict(zip(used, (float(x) for x in w))),
        directions=signs,
        direction_unknown=unknown,
        note=_direction_note(signs, unknown),
    )


def _direction_note(signs: dict[str, int], unknown: list[str]) -> str:
    if not signs:
        return ("Magnitude-only projection: functional direction (agonist/antagonist) was not "
                "supplied, so every receptor contributes as if the molecule were an agonist.")
    if unknown:
        return (f"Direction-aware for {len(signs)} receptor(s); direction unknown for "
                f"{', '.join(unknown)}, which contribute magnitude-only (+1).")
    return f"Direction-aware projection: functional direction supplied for all {len(signs)} used receptors."
