#!/usr/bin/env python
"""Validate predicted brain maps against observed drug-response maps.

    python scripts/evaluate.py \
        --artifact artifacts/affinity_baseline.joblib \
        --atlas data/atlas/density_scale100.csv \
        --spins data/atlas/spins_scale100.npy

For every drug in configs/validation_maps.yaml whose observed map file exists, this predicts
the affinity profile from SMILES, projects it, and spin-tests the correlation with the
observed map. Prints a results table and writes it to CSV.

Runs in DEMO mode on synthetic model/atlas/maps when real inputs are absent, so the harness
is demonstrable offline (clearly labelled; a naive null is used and flagged).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smilesbrain.config import load_receptors
from smilesbrain.validation import evaluate_catalog, load_catalog, load_observed_map
from smilesbrain.validation.nulls import load_spins


def _demo():
    """Build a fully synthetic but self-consistent evaluation to show the harness working."""
    from smilesbrain.data.synthetic import (
        synthetic_atlas,
        synthetic_labels,
        synthetic_observed_map,
        synthetic_smiles,
    )
    from smilesbrain.features import FingerprintConfig, featurize_many, morgan_fingerprint
    from smilesbrain.models.baseline import MaskedMultiTaskRidge

    rc = load_receptors()
    atlas = synthetic_atlas(rc, seed=0)
    smiles = synthetic_smiles(300)
    X, kept = featurize_many(smiles)
    Y = synthetic_labels(X, rc.n_receptors, seed=1)
    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys).fit(X, Y)
    fp = FingerprintConfig()

    def predict(smi: str) -> dict:
        x = morgan_fingerprint(smi, fp)
        vec = model.predict(x[None, :])[0]
        return {k: float(v) for k, v in zip(rc.atlas_keys, vec)}

    catalog = load_catalog()
    observed = {}
    for entry in catalog:
        observed[entry.drug] = synthetic_observed_map(atlas, predict(entry.smiles),
                                                      noise=0.6, seed=hash(entry.drug) % 1000)
    return rc, atlas, predict, catalog, observed, None


def _real(artifact: Path, atlas_path: Path, spins_path: Path | None):
    from smilesbrain.features import morgan_fingerprint

    rc = load_receptors()
    bundle = joblib.load(artifact)
    model, fp, keys = bundle["model"], bundle["fp"], bundle["keys"]
    atlas = pd.read_csv(atlas_path, index_col=0)

    def predict(smi: str) -> dict:
        x = morgan_fingerprint(smi, fp)
        if x is None:
            return {}
        vec = model.predict(x[None, :])[0]
        return {k: float(v) for k, v in zip(keys, vec)}

    catalog = load_catalog()
    observed = {}
    for entry in catalog:
        if Path(entry.file).exists():
            observed[entry.drug] = load_observed_map(entry.file, rc.n_regions)
    perms = load_spins(spins_path, rc.n_regions) if spins_path and spins_path.exists() else None
    return rc, atlas, predict, catalog, observed, perms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/affinity_baseline.joblib", type=Path)
    ap.add_argument("--atlas", default="data/atlas/density_scale100.csv", type=Path)
    ap.add_argument("--spins", default="data/atlas/spins_scale100.npy", type=Path)
    ap.add_argument("--weight-mode", default="zscore", choices=["raw", "zscore", "softmax"])
    ap.add_argument("--out", default="artifacts/validation_results.csv", type=Path)
    args = ap.parse_args()

    demo = not (args.artifact.exists() and args.atlas.exists())
    if demo:
        print("=" * 70)
        print("DEMO MODE: synthetic model/atlas/observed-maps; naive null (NOT a spin test).")
        print("Provide --artifact and --atlas (and --spins) for a real evaluation.")
        print("=" * 70)
        rc, atlas, predict, catalog, observed, perms = _demo()
    else:
        rc, atlas, predict, catalog, observed, perms = _real(args.artifact, args.atlas, args.spins)

    if not observed:
        print("No observed drug maps found. Populate data/validation/ per "
              "configs/validation_maps.yaml and docs/VALIDATION.md.")
        return

    results = evaluate_catalog(catalog, predict, atlas, observed,
                               perm_indices=perms, weight_mode=args.weight_mode)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # naive-null warning already surfaced
        pass

    print(f"\nvalidation ({'naive null' if perms is None else 'spin test'}):")
    print(results.to_string(index=False))
    if "sign_ok" in results and results["sign_ok"].notna().any():
        frac = results["sign_ok"].mean(skipna=True)
        print(f"\nexpected-sign agreement: {frac:.0%}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.out, index=False)
    print(f"results -> {args.out}")


if __name__ == "__main__":
    main()
