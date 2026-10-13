#!/usr/bin/env python
"""Predict a molecule's whole-brain effect map from SMILES.

    python scripts/predict.py --smiles "CN1CCC[C@H]1c1cccnc1"

Loads the trained Stage-1 affinity model and the Hansen atlas, predicts the 19-receptor
affinity vector, and projects it onto the atlas. Prints the predicted affinity vector, the
receptors actually used, and the top/bottom brain regions. Writes the per-region scores to
CSV for visualisation (see scripts/plot_map.py).

Falls back to a freshly fit synthetic model + synthetic atlas if no trained artifacts are
present, so the command is demonstrable out of the box (clearly labelled DEMO).
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
from smilesbrain.projection import project_affinity
from smilesbrain.utils import set_global_seed, write_run_manifest


def _load_atlas(path: Path, rc):
    if path.exists():
        return pd.read_csv(path, index_col=0), False
    from smilesbrain.data.synthetic import synthetic_atlas
    return synthetic_atlas(rc), True


def _load_model(path: Path, rc):
    if path.exists():
        if path.suffix == ".pt":
            from smilesbrain.models.dmpnn import MaskedDMPNNRegressor

            return {
                "model": MaskedDMPNNRegressor.load(path, rc.atlas_keys),
                "keys": rc.atlas_keys,
                "smiles_model": True,
            }, False
        bundle = joblib.load(path)
        return bundle, False
    # DEMO: fit a tiny synthetic model so predict runs without prior training.
    from smilesbrain.data.synthetic import synthetic_labels, synthetic_smiles
    from smilesbrain.features import featurize_many
    from smilesbrain.models.baseline import MaskedMultiTaskRidge
    smiles = synthetic_smiles(300)
    X, _kept = featurize_many(smiles)
    Y = synthetic_labels(X, rc.n_receptors, seed=1)
    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys).fit(X, Y)
    return {
        "model": model,
        "fp": FingerprintConfig(),
        "keys": rc.atlas_keys,
        "training_X": X,
    }, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--artifact", default="artifacts/affinity_baseline.joblib", type=Path)
    ap.add_argument("--atlas", default="data/atlas/density_scale100.csv", type=Path)
    ap.add_argument("--weight-mode", default="zscore", choices=["raw", "zscore", "softmax"])
    ap.add_argument("--out", default="artifacts/prediction.csv", type=Path)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(args.seed)

    rc = load_receptors()
    bundle, demo_model = _load_model(args.artifact, rc)
    model, fp, keys = bundle["model"], bundle.get("fp"), bundle["keys"]
    atlas, demo_atlas = _load_atlas(args.atlas, rc)
    if demo_model or demo_atlas:
        print("=" * 70)
        print("DEMO MODE: using synthetic", "model" if demo_model else "",
              "and" if demo_model and demo_atlas else "", "atlas" if demo_atlas else "")
        print("Outputs are NOT scientifically meaningful. Train a real model and fetch the")
        print("real atlas for genuine predictions.")
        print("=" * 70)

    if bundle.get("smiles_model"):
        pred_vec, uncertainty_vec = model.predict_with_uncertainty([args.smiles])
        pred_vec = pred_vec[0]
        uncertainty_vec = uncertainty_vec[0]
        max_similarity = np.nan
    else:
        x = morgan_fingerprint(args.smiles, fp)
        if x is None:
            raise SystemExit(f"Could not parse SMILES: {args.smiles!r}")
        if "models" in bundle:
            ensemble = np.stack([member.predict(x[None, :]) for member in bundle["models"]])
            pred_vec, uncertainty_vec = ensemble_prediction_stats(ensemble)
            pred_vec = pred_vec[0]
            uncertainty_vec = uncertainty_vec[0]
        else:
            pred_vec = model.predict(x[None, :])[0]
            uncertainty_vec = np.full(len(keys), np.nan)
        training_X = bundle.get("training_X")
        max_similarity = (
            tanimoto_to_training_set(x, training_X) if training_X is not None else np.nan
        )
    out_of_domain = (
        flag_out_of_domain(max_similarity) if np.isfinite(max_similarity) else False
    )
    affinity = {k: float(v) for k, v in zip(keys, pred_vec)}

    print(f"\nSMILES: {args.smiles}")
    print("\npredicted affinity (pChEMBL):")
    for k in keys:
        v = affinity[k]
        bar = "" if np.isnan(v) else "#" * int(max(0, min(20, (v - 4) * 4)))
        uncertainty = uncertainty_vec[keys.index(k)]
        suffix = f"+/- {uncertainty:.2f}" if np.isfinite(uncertainty) else "+/- n/a"
        print(
            f"  {k:8s} {v:6.2f}  {suffix}  {bar}"
            if np.isfinite(v)
            else f"  {k:8s}   n/a  {suffix}"
        )
    if np.isfinite(max_similarity):
        domain_label = "OUT-OF-DOMAIN" if out_of_domain else "in-domain"
        print(f"\napplicability domain: {domain_label} (max training Tanimoto {max_similarity:.3f})")
    else:
        print("\napplicability domain: unavailable (artifact has no training fingerprints)")

    result = project_affinity(affinity, atlas, weight_mode=args.weight_mode)
    scores = result.region_scores
    print(f"\nreceptors used in projection: {', '.join(result.receptors_used)}")

    order = np.argsort(scores)[::-1]
    print(f"\ntop {args.top} regions (predicted strongest effect):")
    for r in order[: args.top]:
        print(f"  region {int(r):3d}  score {scores[r]:+.3f}")

    df = pd.DataFrame({"region": np.arange(len(scores)), "score": scores})
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\nper-region scores -> {args.out}")
    manifest = write_run_manifest(
        args.out,
        config={"smiles": args.smiles, "weight_mode": args.weight_mode},
        seed=args.seed,
        input_paths=[args.artifact, args.atlas],
    )
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
