#!/usr/bin/env python
"""Evaluate whether the affinity model predicts a molecule's receptor *profile* (no brain map).

    python scripts/eval_profiles.py

Rebuilds the scaffold split used for training from the same config, checks it against the saved
artifact (so the molecules scored here are provably unseen), and reports within-molecule
Spearman rho and top-1 hit rate against a receptor-mean baseline, with scaffold-cluster
bootstrap CIs. Also scores the catalog drugs' zero-shot profiles (chemotype held out) against
their measured profiles. Writes artifacts/profile_eval.md and artifacts/profile_eval.csv.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from smilesbrain.config import load_receptors
from smilesbrain.features import FingerprintConfig, featurize_many
from smilesbrain.splits import _scaffold, scaffold_split
from smilesbrain.utils import set_global_seed, write_run_manifest
from smilesbrain.validation import load_catalog
from smilesbrain.validation.profile_eval import (
    cluster_bootstrap_mean,
    knn_profile_predict,
    paired_cluster_bootstrap,
    spearman_rows,
    top1_rows,
)
from smilesbrain.validation.profiles import measured_profiles, zero_shot_profiles
from smilesbrain.validation.report import ridge_refit

KNN_K = 25


def _fmt(mean, lo, hi) -> str:
    return f"{mean:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def _tanimoto_matrix(x_test: np.ndarray, x_train: np.ndarray) -> np.ndarray:
    """Tanimoto of every test row to every training row, via one matrix product."""
    a = x_test.astype(np.float32)
    b = x_train.astype(np.float32)
    inter = a @ b.T
    union = a.sum(1, keepdims=True) + b.sum(1)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/affinity_baseline.yaml")
    ap.add_argument("--out", default="artifacts/profile_eval.csv", type=Path)
    ap.add_argument("--n-boot", default=2000, type=int)
    ap.add_argument("--seed", default=0, type=int)
    args = ap.parse_args()
    set_global_seed(args.seed)
    cfg = yaml.safe_load(open(args.config))
    labels_path = Path(cfg["data"]["labels"])
    artifact = Path(cfg["output"]["artifact"])
    for needed in (labels_path, artifact):
        if not needed.exists():
            raise SystemExit(f"{needed} not found; refusing to evaluate on synthetic data.")

    rc = load_receptors()
    fp = FingerprintConfig(**cfg["features"])
    labels = pd.read_parquet(labels_path)[rc.atlas_keys]
    smiles = labels.index.tolist()
    X, kept = featurize_many(smiles, fp)
    Y = labels.to_numpy(dtype=float)[kept]
    smiles = [smiles[i] for i in kept]
    tr, te = scaffold_split(smiles, cfg["split"]["test_frac"], cfg["split"]["seed"])

    bundle = joblib.load(artifact)
    if bundle["training_X"].shape != X[tr].shape or not np.array_equal(bundle["training_X"], X[tr]):
        raise SystemExit("artifact was not trained on this split; retrain or fix the config")
    scaffolds_tr = {_scaffold(smiles[i]) for i in tr}
    test_scaf = np.array([_scaffold(smiles[i]) for i in te])
    assert not scaffolds_tr & set(test_scaf), "scaffold leakage between train and test"

    model = bundle["model"]
    pred = model.predict(X[te])
    baseline = np.nanmean(Y[tr], axis=0)
    true = Y[te]
    keys = rc.atlas_keys
    evaluable = np.flatnonzero(np.isfinite(true).sum(axis=1) >= 3)
    sim_full = np.full(len(te), np.nan)
    pred_knn = np.full(pred.shape, np.nan)
    sim_eval = _tanimoto_matrix(X[te][evaluable], X[tr])
    sim_full[evaluable] = sim_eval.max(axis=1)
    pred_knn[evaluable] = knn_profile_predict(sim_eval, Y[tr], KNN_K)
    del sim_eval

    rho_m, rho_b = spearman_rows(pred, true), spearman_rows(baseline, true)
    rho_k = spearman_rows(pred_knn, true)
    hit_m, chance = top1_rows(pred, true)
    hit_b, _ = top1_rows(baseline, true)
    hit_k, _ = top1_rows(pred_knn, true)
    n_eval = int(np.isfinite(rho_m).sum())
    k_measured = np.isfinite(true).sum(axis=1)

    rows = []

    def add(label, mask):
        m = mask & np.isfinite(rho_m)
        if m.sum() < 10:
            return
        c = test_scaf[m]
        r_mod = cluster_bootstrap_mean(rho_m[m], c, args.n_boot, args.seed)
        r_bas = cluster_bootstrap_mean(rho_b[m], c, args.n_boot, args.seed)
        diff = paired_cluster_bootstrap(rho_m[m], rho_b[m], c, args.n_boot, args.seed)
        r_knn = cluster_bootstrap_mean(rho_k[m], c, args.n_boot, args.seed)
        diff_k = paired_cluster_bootstrap(rho_m[m], rho_k[m], c, args.n_boot, args.seed)
        h_knn = cluster_bootstrap_mean(hit_k[m], c, args.n_boot, args.seed)
        h_mod = cluster_bootstrap_mean(hit_m[m], c, args.n_boot, args.seed)
        h_bas = cluster_bootstrap_mean(hit_b[m], c, args.n_boot, args.seed)
        h_cha = float(np.nanmean(chance[m]))
        rows.append({"subset": label, "n_molecules": int(m.sum()), "n_scaffolds": len(set(c)),
                     "rho_model": r_mod[0], "rho_model_lo": r_mod[1], "rho_model_hi": r_mod[2],
                     "rho_baseline": r_bas[0], "rho_baseline_lo": r_bas[1],
                     "rho_baseline_hi": r_bas[2],
                     "rho_diff": diff[0], "rho_diff_lo": diff[1], "rho_diff_hi": diff[2],
                     "rho_knn": r_knn[0], "rho_knn_lo": r_knn[1], "rho_knn_hi": r_knn[2],
                     "rho_diff_vs_knn": diff_k[0], "rho_diff_vs_knn_lo": diff_k[1],
                     "rho_diff_vs_knn_hi": diff_k[2], "top1_knn": h_knn[0],
                     "top1_model": h_mod[0], "top1_model_lo": h_mod[1], "top1_model_hi": h_mod[2],
                     "top1_baseline": h_bas[0], "top1_chance": h_cha})

    everything = np.ones(len(te), dtype=bool)
    add("all held-out molecules (>=3 receptors measured)", everything)
    add("3 receptors measured", k_measured == 3)
    add("4-5 receptors measured", (k_measured >= 4) & (k_measured <= 5))
    add("6+ receptors measured", k_measured >= 6)

    # Novelty: similarity of each evaluable test molecule to its nearest training molecule.
    in_sample = np.isfinite(sim_full)
    sim = sim_full
    add("novelty: nearest training Tanimoto < 0.3", in_sample & (sim < 0.3))
    add("novelty: 0.3 - 0.5", in_sample & (sim >= 0.3) & (sim < 0.5))
    add("novelty: >= 0.5", in_sample & (sim >= 0.5))
    table = pd.DataFrame(rows)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out, index=False)

    # Catalog drugs: zero-shot (chemotype removed) vs measured profile.
    catalog = [e for e in load_catalog()]
    measured, _ = measured_profiles(catalog, labels, keys)
    alpha = getattr(model, "alpha", 1.0)
    zshot, zdiag = zero_shot_profiles(catalog, labels, fp, keys, ridge_refit(alpha, keys),
                                      drugs=set(measured))
    drug_rows = []
    for d in sorted(measured):
        m_vec = np.array([measured[d].get(k, np.nan) for k in keys])
        z_vec = np.array([zshot[d].get(k, np.nan) for k in keys])
        if np.isfinite(m_vec).sum() < 3:
            continue
        rz = spearman_rows(z_vec[None, :], m_vec[None, :])[0]
        rb = spearman_rows(baseline, m_vec[None, :])[0]
        ok = np.isfinite(m_vec)
        top_true = keys[int(np.nanargmax(np.where(ok, m_vec, np.nan)))]
        order = [keys[i] for i in np.argsort(-np.where(ok, z_vec, -np.inf))]
        drug_rows.append({"drug": d, "n_measured": int(ok.sum()), "rho_zero_shot": rz,
                          "rho_baseline": rb, "measured_top": top_true,
                          "top1_hit": order[0] == top_true,
                          "top3_hit": top_true in order[:3],
                          "nearest_train_tanimoto": zdiag[d]["max_tanimoto"]})
    drugs = pd.DataFrame(drug_rows)
    drugs.to_csv(args.out.with_name("profile_eval_drugs.csv"), index=False)

    md = ["# Does the affinity model predict a molecule's receptor profile?", "",
          f"Scaffold split from `{args.config}` (seed {cfg['split']['seed']}): {len(tr)} train / "
          f"{len(te)} test molecules, **no scaffold shared** (asserted); artifact verified to have "
          f"been trained on exactly this training fold. {n_eval} test molecules have >=3 measured "
          f"receptors. Mean per-receptor held-out R^2 is in `artifacts/affinity_baseline_report.csv`.",
          "", "Within-molecule Spearman rho between predicted and measured profile across the "
          "receptors measured for that molecule; 95% CIs by resampling whole scaffolds "
          f"({args.n_boot} draws). The baseline predicts every molecule with the training mean "
          "of each receptor, so it knows nothing about the molecule. A second baseline copies the "
          "profile of the nearest training analogues (kNN).", "",
          "| subset | molecules | scaffolds | rho model | rho receptor-mean | model - receptor-mean | "
          f"rho kNN (k={KNN_K}) | model - kNN | top-1 model | top-1 receptor-mean | top-1 kNN | "
          "top-1 chance |", "|---|---:|---:|---|---|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        md.append(
            f"| {r['subset']} | {r['n_molecules']} | {r['n_scaffolds']} | "
            f"{_fmt(r['rho_model'], r['rho_model_lo'], r['rho_model_hi'])} | "
            f"{_fmt(r['rho_baseline'], r['rho_baseline_lo'], r['rho_baseline_hi'])} | "
            f"{_fmt(r['rho_diff'], r['rho_diff_lo'], r['rho_diff_hi'])} | "
            f"{_fmt(r['rho_knn'], r['rho_knn_lo'], r['rho_knn_hi'])} | "
            f"{_fmt(r['rho_diff_vs_knn'], r['rho_diff_vs_knn_lo'], r['rho_diff_vs_knn_hi'])} | "
            f"{r['top1_model']:.3f} | {r['top1_baseline']:.3f} | {r['top1_knn']:.3f} | "
            f"{r['top1_chance']:.3f} |")
    md += ["", "Novelty rows use the nearest-training Tanimoto (Morgan bits) of every evaluable test "
           f"molecule. The kNN baseline averages the {KNN_K} most similar training molecules that "
           "have each receptor measured, weighted by similarity.", "",
           "## Catalog drugs: zero-shot profile (chemotype held out) vs measured profile", "",
           "Only drugs with >=3 measured receptors. rho is across the measured receptors.", "",
           "| drug | measured receptors | rho zero-shot | rho baseline | measured top | top-1 | "
           "top-3 | nearest train Tanimoto |", "|---|---:|---:|---:|---|---|---|---:|"]
    for r in drug_rows:
        md.append(f"| {r['drug']} | {r['n_measured']} | {r['rho_zero_shot']:+.2f} | "
                  f"{r['rho_baseline']:+.2f} | {r['measured_top']} | "
                  f"{'yes' if r['top1_hit'] else 'no'} | {'yes' if r['top3_hit'] else 'no'} | "
                  f"{r['nearest_train_tanimoto']:.2f} |")
    if len(drugs):
        md += ["", f"{len(drugs)} drugs: zero-shot rho beats the baseline for "
               f"{int((drugs.rho_zero_shot > drugs.rho_baseline).sum())}; measured top receptor "
               f"is the predicted top-1 for {int(drugs.top1_hit.sum())} and in the predicted "
               f"top-3 for {int(drugs.top3_hit.sum())}."]
    report = args.out.with_name("profile_eval.md")
    report.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    manifest = write_run_manifest(report, config={"config": cfg, "n_boot": args.n_boot},
                                  seed=args.seed, input_paths=[args.config, labels_path, artifact])
    print(f"table -> {args.out}\nreport -> {report}\nmanifest -> {manifest}")


if __name__ == "__main__":
    main()
