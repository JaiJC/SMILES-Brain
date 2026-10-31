#!/usr/bin/env python
"""Rank a SMILES library against an inverse-designed regional target profile.

Example:
    python scripts/design_screen.py --smiles-file candidates.txt \
        --high-regions 1,2,3 --low-regions 98,99,100

Outputs are hypotheses, not validated biological effects.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smilesbrain.config import load_receptors
from smilesbrain.features import FingerprintConfig, morgan_fingerprint
from smilesbrain.models.uncertainty import (
    ensemble_prediction_stats,
    flag_out_of_domain,
    tanimoto_to_training_set,
)
from smilesbrain.design.score import fit_target_profile, score_affinity


def _region_indices(spec: str | None, n_regions: int) -> list[int]:
    if not spec:
        return []
    try:
        indices = [int(value.strip()) for value in spec.split(",") if value.strip()]
    except ValueError as exc:
        raise ValueError("region lists must be comma-separated integers") from exc
    if any(index < 0 or index >= n_regions for index in indices):
        raise ValueError(f"region indices must be between 0 and {n_regions - 1}")
    return sorted(set(indices))


def _load_candidates(path: Path | None, values: list[str]) -> list[str]:
    candidates = list(values)
    if path:
        candidates.extend(
            line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not candidates:
        raise ValueError("provide --smiles or --smiles-file")
    return list(dict.fromkeys(candidates))


def _predict(bundle: dict, smiles: list[str], keys: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    fp = bundle.get("fp", FingerprintConfig())
    model = bundle["model"]
    training = bundle.get("training_X")
    predictions, uncertainty, similarities = [], [], []
    for smi in smiles:
        x = morgan_fingerprint(smi, fp)
        if x is None:
            predictions.append(np.full(len(keys), np.nan))
            uncertainty.append(np.full(len(keys), np.nan))
            similarities.append(np.nan)
            continue
        if "models" in bundle:
            ensemble = np.stack([member.predict(x[None, :]) for member in bundle["models"]])
            mean, std = ensemble_prediction_stats(ensemble)
            pred, unc = mean[0], std[0]
        else:
            pred = model.predict(x[None, :])[0]
            unc = np.full(len(keys), np.nan)
        predictions.append(pred)
        uncertainty.append(unc)
        similarities.append(
            tanimoto_to_training_set(x, training) if training is not None else np.nan
        )
    return np.asarray(predictions), np.asarray(uncertainty), np.asarray(similarities)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", action="append", default=[])
    parser.add_argument("--smiles-file", type=Path)
    parser.add_argument("--artifact", type=Path, default=Path("artifacts/affinity_baseline.joblib"))
    parser.add_argument("--parcellation", default=None)
    parser.add_argument("--atlas", type=Path, default=None)
    parser.add_argument("--high-regions", default=None, help="zero-based region indices")
    parser.add_argument("--low-regions", default=None, help="zero-based region indices")
    parser.add_argument("--target", type=Path, help="CSV with one target value per region")
    parser.add_argument("--regularization", type=float, default=1e-6)
    parser.add_argument("--out", type=Path, default=Path("artifacts/design_screen.csv"))
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    np.random.seed(args.seed)

    rc = load_receptors(parcellation=args.parcellation)
    atlas_path = args.atlas or Path(rc.parcellation.atlas_file)
    if not atlas_path.exists() or not args.artifact.exists():
        raise SystemExit("real atlas and model artifact are required; refusing DEMO MODE for design.")
    atlas = pd.read_csv(atlas_path, index_col=0)
    if list(atlas.columns) != rc.atlas_keys or len(atlas) != rc.n_regions:
        raise SystemExit("atlas does not match the configured parcellation and receptor order")

    if args.target:
        target = pd.read_csv(args.target, header=None).to_numpy().ravel()
    else:
        target = np.full(rc.n_regions, np.nan)
        high = _region_indices(args.high_regions, rc.n_regions)
        low = _region_indices(args.low_regions, rc.n_regions)
        if set(high) & set(low):
            raise SystemExit("a region cannot be both high and low")
        target[high] = 1.0
        target[low] = -1.0
    profile = fit_target_profile(atlas, target, regularization=args.regularization)

    bundle = joblib.load(args.artifact)
    smiles = _load_candidates(args.smiles_file, args.smiles)
    predictions, uncertainties, similarities = _predict(bundle, smiles, rc.atlas_keys)
    rows = []
    for i, smi in enumerate(smiles):
        rows.append({
            "smiles": smi,
            "score": score_affinity(predictions[i], profile),
            "max_training_tanimoto": similarities[i],
            "out_of_domain": (
                bool(flag_out_of_domain(float(similarities[i])))
                if np.isfinite(similarities[i]) else False
            ),
            "prediction_uncertainty_rmse": (
                float(np.sqrt(np.nanmean(uncertainties[i] ** 2)))
                if np.isfinite(uncertainties[i]).any() else np.nan
            ),
            "predicted_affinity": ";".join(
                f"{key}={value:.6g}" for key, value in zip(rc.atlas_keys, predictions[i])
                if np.isfinite(value)
            ),
        })
    result = pd.DataFrame(rows).sort_values(["score", "smiles"], ascending=[False, True])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)
    print("UNVALIDATED HYPOTHESES: ranking chemical structures against a fitted regional profile.")
    print(f"target fit residual norm: {profile.residual_norm:.6g}")
    print(result.to_string(index=False))
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
