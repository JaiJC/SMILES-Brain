#!/usr/bin/env python
"""Propose novel molecules toward a regional target profile (UNVALIDATED HYPOTHESES).

    python scripts/design_generate.py --seed-smiles-file seeds.txt \
        --high-regions 1,2,3 --low-regions 98,99,100

Runs a mutation-based genetic search from seed molecules, scoring each candidate with the
same objective as scripts/design_screen.py, and compares the result with a random-mutation
search that spends the same evaluation budget. Output is a ranked CSV; every row is labelled
as an unvalidated hypothesis. Requires RDKit, a trained fingerprint-model artifact and the
real atlas (it refuses DEMO MODE).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from smilesbrain.config import load_receptors
from smilesbrain.design.generate import (
    STATUS,
    compare_to_random,
    evolve,
    random_search,
    rdkit_canonicalize,
    rdkit_mutate,
)
from smilesbrain.design.score import fit_target_profile, score_affinity
from smilesbrain.design.screen import high_low_target, predict_bundle, region_indices
from smilesbrain.models.uncertainty import flag_out_of_domain
from smilesbrain.utils import set_global_seed, write_run_manifest


def _read_seeds(path: Path | None, values: list[str]) -> list[str]:
    seeds = list(values)
    if path:
        seeds.extend(
            line.strip() for line in path.read_text().splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    if not seeds:
        raise SystemExit("provide --seed-smiles and/or --seed-smiles-file")
    return seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed-smiles", action="append", default=[])
    ap.add_argument("--seed-smiles-file", type=Path)
    ap.add_argument("--artifact", type=Path, default=Path("artifacts/affinity_baseline.joblib"))
    ap.add_argument("--parcellation", default=None)
    ap.add_argument("--atlas", type=Path, default=None)
    ap.add_argument("--high-regions", default=None, help="zero-based region indices")
    ap.add_argument("--low-regions", default=None, help="zero-based region indices")
    ap.add_argument("--target", type=Path, help="CSV with one target value per region")
    ap.add_argument("--regularization", type=float, default=1e-6)
    ap.add_argument("--population", type=int, default=32)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--out", type=Path, default=Path("artifacts/design_generate.csv"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(args.seed)

    rc = load_receptors(parcellation=args.parcellation)
    atlas_path = args.atlas or Path(rc.parcellation.atlas_file)
    if not atlas_path.exists() or not args.artifact.exists():
        raise SystemExit(
            "real atlas and model artifact are required; refusing DEMO MODE for design."
        )
    atlas = pd.read_csv(atlas_path, index_col=0)
    if list(atlas.columns) != rc.atlas_keys or len(atlas) != rc.n_regions:
        raise SystemExit("atlas does not match the configured parcellation and receptor order")
    if args.target:
        target = pd.read_csv(args.target, header=None).to_numpy().ravel()
    else:
        try:
            target = high_low_target(
                rc.n_regions,
                region_indices(args.high_regions, rc.n_regions),
                region_indices(args.low_regions, rc.n_regions),
            )
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
    profile = fit_target_profile(atlas, target, regularization=args.regularization)
    bundle = joblib.load(args.artifact)
    if "training_X" not in bundle:
        print("WARNING: artifact has no training fingerprints; applicability domain unavailable.")

    def score_fn(smiles: str) -> float:
        pred, _unc, _sim = predict_bundle(bundle, [smiles], len(rc.atlas_keys))
        return score_affinity(pred[0], profile) if np.isfinite(pred[0]).any() else float("nan")

    seeds = _read_seeds(args.seed_smiles_file, args.seed_smiles)
    kwargs = dict(score_fn=score_fn, mutate_fn=rdkit_mutate, canonicalize=rdkit_canonicalize)
    evolved = evolve(
        seeds, rng=np.random.default_rng(args.seed), population=args.population,
        generations=args.generations, **kwargs,
    )
    baseline = random_search(
        seeds, rng=np.random.default_rng(args.seed + 1),
        n_evaluations=evolved.n_evaluations, **kwargs,
    )
    summary = compare_to_random(evolved, baseline, k=args.top)

    top = evolved.top(args.top)
    preds, uncs, sims = predict_bundle(bundle, [p.smiles for p in top], len(rc.atlas_keys))
    rows = []
    for p, pred, unc, sim in zip(top, preds, uncs, sims):
        rows.append({
            "smiles": p.smiles, "score": p.score, "generation": p.generation, "parent": p.parent,
            "max_training_tanimoto": sim,
            "out_of_domain": bool(flag_out_of_domain(float(sim))) if np.isfinite(sim) else False,
            "prediction_uncertainty_rmse": (
                float(np.sqrt(np.nanmean(unc ** 2))) if np.isfinite(unc).any() else np.nan
            ),
            "predicted_affinity": ";".join(
                f"{k}={v:.6g}" for k, v in zip(rc.atlas_keys, pred) if np.isfinite(v)
            ),
            "status": STATUS,
        })
    result = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False)

    print(f"{STATUS}: structures optimised against a *predicted-affinity* objective.")
    print("Binding affinity is not a validated predictor of brain maps (docs/VALIDATION.md).")
    print(f"target fit residual norm: {profile.residual_norm:.6g}")
    print(f"evaluations: evolved={evolved.n_evaluations} random={baseline.n_evaluations}")
    print(f"top-{args.top} mean score: evolved={summary['evolved_top_k_mean']:.4g} "
          f"random={summary['random_top_k_mean']:.4g}")
    print(result[["smiles", "score", "max_training_tanimoto", "out_of_domain"]]
          .to_string(index=False))
    manifest = write_run_manifest(
        args.out,
        config={"population": args.population, "generations": args.generations,
                "parcellation": rc.parcellation.key, "summary": summary},
        seed=args.seed, input_paths=[args.artifact, atlas_path],
    )
    print(f"results -> {args.out}\nmanifest -> {manifest}")


if __name__ == "__main__":
    main()
