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
from smilesbrain.projection import project_affinity


def _load_atlas(path: Path, rc):
    if path.exists():
        return pd.read_csv(path, index_col=0), False
    from smilesbrain.data.synthetic import synthetic_atlas
    return synthetic_atlas(rc), True


def _load_model(path: Path, rc):
    if path.exists():
        bundle = joblib.load(path)
        return bundle["model"], bundle["fp"], bundle["keys"], False
    # DEMO: fit a tiny synthetic model so predict runs without prior training.
    from smilesbrain.data.synthetic import synthetic_smiles, synthetic_labels
    from smilesbrain.features import featurize_many
    from smilesbrain.models.baseline import MaskedMultiTaskRidge
    smiles = synthetic_smiles(300)
    X, kept = featurize_many(smiles)
    Y = synthetic_labels(X, rc.n_receptors, seed=1)
    model = MaskedMultiTaskRidge(receptor_keys=rc.atlas_keys).fit(X, Y)
    return model, FingerprintConfig(), rc.atlas_keys, True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--artifact", default="artifacts/affinity_baseline.joblib", type=Path)
    ap.add_argument("--atlas", default="data/atlas/density_scale100.csv", type=Path)
    ap.add_argument("--weight-mode", default="zscore", choices=["raw", "zscore", "softmax"])
    ap.add_argument("--out", default="artifacts/prediction.csv", type=Path)
    ap.add_argument("--top", type=int, default=10)
    args = ap.parse_args()

    rc = load_receptors()
    model, fp, keys, demo_model = _load_model(args.artifact, rc)
    atlas, demo_atlas = _load_atlas(args.atlas, rc)
    if demo_model or demo_atlas:
        print("=" * 70)
        print("DEMO MODE: using synthetic", "model" if demo_model else "",
              "and" if demo_model and demo_atlas else "", "atlas" if demo_atlas else "")
        print("Outputs are NOT scientifically meaningful. Train a real model and fetch the")
        print("real atlas for genuine predictions.")
        print("=" * 70)

    x = morgan_fingerprint(args.smiles, fp)
    if x is None:
        raise SystemExit(f"Could not parse SMILES: {args.smiles!r}")

    pred_vec = model.predict(x[None, :])[0]
    affinity = {k: float(v) for k, v in zip(keys, pred_vec)}

    print(f"\nSMILES: {args.smiles}")
    print("\npredicted affinity (pChEMBL):")
    for k in keys:
        v = affinity[k]
        bar = "" if np.isnan(v) else "#" * int(max(0, min(20, (v - 4) * 4)))
        print(f"  {k:8s} {v:6.2f}  {bar}" if np.isfinite(v) else f"  {k:8s}   n/a")

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


if __name__ == "__main__":
    main()
