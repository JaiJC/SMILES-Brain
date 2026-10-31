#!/usr/bin/env python
"""Validate predicted brain maps against observed drug-response maps.

    python scripts/evaluate.py \
        --artifact artifacts/affinity_baseline.joblib \
        --atlas data/atlas/density_scale100.csv \
        --spins data/atlas/spins_scale100.npy

`--parcellation KEY` (see configs/receptors.yaml) selects another parcellation and, unless
overridden, its atlas and spins files.

`--report` instead writes the two-tier report (artifacts/validation_report.md + .png): tier A
reproduces the result from measured affinities, tier B is zero-shot with each drug's chemotype
removed from training (asserted). Needs real inputs; it refuses to run in DEMO mode.

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
from smilesbrain.utils import set_global_seed, stable_seed, write_run_manifest


def _demo(parcellation: str | None):
    """Build a fully synthetic but self-consistent evaluation to show the harness working."""
    from smilesbrain.data.synthetic import (
        synthetic_atlas,
        synthetic_labels,
        synthetic_observed_map,
        synthetic_smiles,
    )
    from smilesbrain.features import FingerprintConfig, featurize_many, morgan_fingerprint
    from smilesbrain.models.baseline import MaskedMultiTaskRidge

    rc = load_receptors(parcellation=parcellation)
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
                                                      noise=0.6, seed=stable_seed(entry.drug))
    return rc, atlas, predict, catalog, observed, None


def _real(artifact: Path, atlas_path: Path, spins_path: Path | None, parcellation: str | None):
    from smilesbrain.features import morgan_fingerprint

    rc = load_receptors(parcellation=parcellation)
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


def _write_report(args, rc, atlas, catalog, observed, perms, demo: bool) -> None:
    import json

    from smilesbrain.validation.report import (
        build_markdown, plot_report, tier_rows_to_frame, two_tier_evaluation,
    )

    if demo:
        raise SystemExit("--report needs a real --artifact and --atlas; refusing to report on "
                         "synthetic data.")
    if not observed:
        raise SystemExit("--report: no observed maps found (see docs/VALIDATION.md).")
    if not args.labels.exists():
        raise SystemExit(f"--report: label table not found: {args.labels}")
    bundle = joblib.load(args.artifact)
    labels = pd.read_parquet(args.labels) if args.labels.suffix == ".parquet" \
        else pd.read_csv(args.labels, index_col=0)

    n_subjects = {}
    for entry in catalog:
        side = Path(entry.file).with_name(Path(entry.file).name + ".provenance.json")
        if side.exists():
            n_subjects[entry.drug] = json.loads(side.read_text()).get("n_subjects")

    from smilesbrain.validation.report import (
        build_diagnostics, dmpnn_refit, gbdt_refit, label_keys, ridge_refit,
    )

    keys = bundle["keys"]
    alpha = getattr(bundle["model"], "alpha", 1.0)
    refit = {"sklearn": lambda: ridge_refit(alpha, keys),
             "gbdt": lambda: gbdt_refit(keys),
             "dmpnn": lambda: dmpnn_refit(keys)}[args.zero_shot_backend]()
    rows = two_tier_evaluation(
        catalog, observed, atlas, labels, bundle["fp"], alpha, keys,
        perm_indices=perms, weight_mode=args.weight_mode, n_subjects=n_subjects, refit=refit,
    )
    tiers_csv = args.report_out.with_name("validation_tiers.csv")
    tiers_csv.parent.mkdir(parents=True, exist_ok=True)
    tier_rows_to_frame(rows).to_csv(tiers_csv, index=False)
    meta = {"parcellation": rc.parcellation.key, "n_regions": rc.n_regions,
            "weight_mode": args.weight_mode, "seed": args.seed}
    diag = None
    if not args.no_diagnostics:
        print("running diagnostics (target ranking, ceiling, receptor-permutation null)...")
        keys_kept = label_keys(labels.index.tolist())
        diag = build_diagnostics(catalog, observed, atlas, labels, keys_kept, labels,
                                 perm_indices=perms, weight_mode=args.weight_mode,
                                 n_perm=args.n_perm_receptor)
    md = build_markdown(rows, meta, diag)
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    fig = plot_report(rows, args.report_out.with_suffix(".png"))
    args.report_out.write_text(md + f"\n![per-drug agreement]({fig.name})\n")
    print(md)
    print(f"report -> {args.report_out}\nfigure -> {fig}\ntier table -> {tiers_csv}")
    manifest = write_run_manifest(
        args.report_out,
        config={"artifact": str(args.artifact), "atlas": str(args.atlas), "spins": str(args.spins),
                "labels": str(args.labels), "weight_mode": args.weight_mode,
                "parcellation": rc.parcellation.key, "alpha": alpha,
                "zero_shot_backend": args.zero_shot_backend},
        seed=args.seed,
        input_paths=[args.artifact, args.atlas, args.spins, args.labels]
        + [e.file for e in catalog if e.drug in observed],
    )
    print(f"manifest -> {manifest}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", default="artifacts/affinity_baseline.joblib", type=Path)
    ap.add_argument("--parcellation", default=None,
                    help="key under `parcellations` in configs/receptors.yaml")
    ap.add_argument("--atlas", default=None, type=Path,
                    help="default: the parcellation's atlas_file")
    ap.add_argument("--spins", default=None, type=Path,
                    help="default: the parcellation's spins_file")
    ap.add_argument("--report", action="store_true", help="write the two-tier validation report")
    ap.add_argument("--labels", default="data/labels.parquet", type=Path,
                    help="measured-affinity label table (used by --report)")
    ap.add_argument("--report-out", default="artifacts/validation_report.md", type=Path)
    ap.add_argument("--no-diagnostics", action="store_true",
                    help="skip the target-ranking / ceiling / permutation-null diagnostics")
    ap.add_argument("--n-perm-receptor", type=int, default=5000,
                    help="permutations for the receptor-permutation null")
    ap.add_argument("--zero-shot-backend", default="sklearn", choices=["sklearn", "gbdt", "dmpnn"],
                    help="model refit per drug for tier B (dmpnn is slow: one training run "
                         "per drug)")
    ap.add_argument("--weight-mode", default="zscore", choices=["raw", "zscore", "softmax"])
    ap.add_argument("--out", default="artifacts/validation_results.csv", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    set_global_seed(args.seed)
    parc = load_receptors(parcellation=args.parcellation).parcellation
    args.atlas = args.atlas or Path(parc.atlas_file)
    args.spins = args.spins or Path(parc.spins_file)

    demo = not (args.artifact.exists() and args.atlas.exists())
    if demo:
        print("=" * 70)
        print("DEMO MODE: synthetic model/atlas/observed-maps; naive null (NOT a spin test).")
        print("Provide --artifact and --atlas (and --spins) for a real evaluation.")
        print("=" * 70)
        rc, atlas, predict, catalog, observed, perms = _demo(args.parcellation)
    else:
        rc, atlas, predict, catalog, observed, perms = _real(args.artifact, args.atlas, args.spins, args.parcellation)

    if args.report:
        _write_report(args, rc, atlas, catalog, observed, perms, demo)
        return

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
    manifest = write_run_manifest(
        args.out,
        config={"artifact": str(args.artifact), "atlas": str(args.atlas),
                "spins": str(args.spins), "weight_mode": args.weight_mode,
                "parcellation": parc.key},
        seed=args.seed,
        input_paths=[args.artifact, args.atlas, args.spins],
    )
    print(f"manifest -> {manifest}")


if __name__ == "__main__":
    main()
