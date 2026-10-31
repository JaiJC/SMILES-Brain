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

import numpy as np
import pandas as pd

from smilesbrain.config import load_receptors
from smilesbrain.projection import parse_direction_spec
from smilesbrain.service import DEMO_BANNER, load_atlas, load_model, predict_molecule
from smilesbrain.utils import set_global_seed, write_run_manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smiles", required=True)
    ap.add_argument("--artifact", default="artifacts/affinity_baseline.joblib", type=Path)
    ap.add_argument("--parcellation", default=None,
                    help="key under `parcellations` in configs/receptors.yaml")
    ap.add_argument("--atlas", default=None, type=Path,
                    help="default: the parcellation's atlas_file")
    ap.add_argument("--direction", default=None,
                    help='functional direction per receptor, e.g. "5HT2a=+1,D2=-1" '
                         "(+1 agonist, -1 antagonist); omit for magnitude-only")
    ap.add_argument("--weight-mode", default="zscore", choices=["raw", "zscore", "softmax"])
    ap.add_argument("--out", default="artifacts/prediction.csv", type=Path)
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(args.seed)

    rc = load_receptors(parcellation=args.parcellation)
    args.atlas = args.atlas or Path(rc.parcellation.atlas_file)
    direction = parse_direction_spec(args.direction) if args.direction else None
    bundle, demo_model = load_model(args.artifact, rc)
    keys = bundle["keys"]
    atlas, demo_atlas = load_atlas(args.atlas, rc)
    if demo_model or demo_atlas:
        what = " and ".join(
            w for w, on in (("model", demo_model), ("atlas", demo_atlas)) if on
        )
        print("=" * 70)
        print(DEMO_BANNER.format(what=what))
        print("=" * 70)

    try:
        prediction = predict_molecule(
            args.smiles, bundle, atlas, direction=direction, weight_mode=args.weight_mode
        )
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    affinity, uncertainty_vec = prediction.affinity, prediction.uncertainty
    max_similarity, out_of_domain = prediction.max_training_tanimoto, prediction.out_of_domain

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
        print(f"\napplicability domain: {domain_label} "
              f"(max training Tanimoto {max_similarity:.3f})")
    else:
        print("\napplicability domain: unavailable (artifact has no training fingerprints)")

    result = prediction.projection
    scores = result.region_scores
    print(f"\nreceptors used in projection: {', '.join(result.receptors_used)}")
    print(f"direction: {result.note}")

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
        config={"smiles": args.smiles, "weight_mode": args.weight_mode,
                "parcellation": rc.parcellation.key, "direction": direction},
        seed=args.seed,
        input_paths=[args.artifact, args.atlas],
    )
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
