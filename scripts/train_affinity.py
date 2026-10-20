#!/usr/bin/env python
"""Train the Stage-1 affinity model (SMILES -> 19-receptor pChEMBL).

Runs end-to-end on real ChEMBL labels when `data.labels` exists, otherwise on synthetic
learnable data so the pipeline is demonstrable offline. Evaluates with a scaffold split and
reports per-receptor R^2 on held-out chemotypes.

    python scripts/train_affinity.py --config configs/affinity_baseline.yaml
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
from smilesbrain.models.baseline import MaskedMultiTaskRidge
from smilesbrain.models.trees import MaskedMultiTaskGBDT
from smilesbrain.splits import random_split, scaffold_split
from smilesbrain.utils import set_global_seed, write_run_manifest


def _load_labels(cfg_data: dict, receptor_keys: list[str]):
    path = Path(cfg_data["labels"])
    if path.exists():
        table = (
            pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
        )
        smiles = table.index.tolist()
        cols = [k for k in receptor_keys if k in table.columns]
        missing = [k for k in receptor_keys if k not in table.columns]
        if missing:
            raise ValueError(
                "Labels are missing canonical receptor columns: " + ", ".join(missing)
            )
        return smiles, table[cols].to_numpy(dtype=float), cols, False
    # synthetic fallback
    from smilesbrain.data.synthetic import synthetic_labels, synthetic_smiles
    n = int(cfg_data.get("synthetic_n", 400))
    smiles = synthetic_smiles(n)
    X, kept = featurize_many(smiles)
    smiles = [smiles[i] for i in kept]
    Y = synthetic_labels(X, len(receptor_keys), seed=0)
    return smiles, Y, receptor_keys, True


def per_receptor_r2(y_true: np.ndarray, y_pred: np.ndarray, keys: list[str]) -> pd.DataFrame:
    rows = []
    for j, k in enumerate(keys):
        t, p = y_true[:, j], y_pred[:, j]
        m = np.isfinite(t) & np.isfinite(p)
        if m.sum() < 3:
            rows.append({"receptor": k, "n_test": int(m.sum()), "r2": np.nan})
            continue
        ss_res = np.sum((t[m] - p[m]) ** 2)
        ss_tot = np.sum((t[m] - t[m].mean()) ** 2) + 1e-12
        rows.append({"receptor": k, "n_test": int(m.sum()), "r2": 1 - ss_res / ss_tot})
    return pd.DataFrame(rows)


def _inner_split(
    smiles: list[str], train_idx: np.ndarray, cfg: dict, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Split the outer training fold without leaking scaffolds into validation."""
    validation_frac = float(cfg.get("validation_frac", 0.2))
    if validation_frac <= 0:
        return train_idx, np.array([], dtype=int)
    inner_smiles = [smiles[i] for i in train_idx]
    if cfg.get("scheme", "scaffold") == "scaffold":
        inner_train, inner_val = scaffold_split(inner_smiles, validation_frac, seed)
    else:
        inner_train, inner_val = random_split(len(inner_smiles), validation_frac, seed)
    return train_idx[inner_train], train_idx[inner_val]


def _select_alpha(
    X: np.ndarray,
    Y: np.ndarray,
    smiles: list[str],
    train_idx: np.ndarray,
    cfg: dict,
    keys: list[str],
    seed: int,
) -> tuple[float, pd.DataFrame]:
    """Select Ridge alpha on a validation fold, never on the held-out test fold."""
    inner_train, inner_val = _inner_split(smiles, train_idx, cfg, seed)
    candidates = [float(value) for value in cfg.get("alpha_grid", [cfg.get("alpha", 1.0)])]
    if not len(inner_val):
        return candidates[0], pd.DataFrame({"alpha": candidates, "mean_r2": np.nan})
    scores = []
    for alpha in candidates:
        candidate = MaskedMultiTaskRidge(receptor_keys=keys, alpha=alpha).fit(
            X[inner_train], Y[inner_train]
        )
        validation = per_receptor_r2(Y[inner_val], candidate.predict(X[inner_val]), keys)
        scores.append({"alpha": alpha, "mean_r2": validation["r2"].mean(skipna=True)})
    grid = pd.DataFrame(scores)
    best = grid.loc[grid["mean_r2"].fillna(-np.inf).idxmax(), "alpha"]
    return float(best), grid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/affinity_baseline.yaml")
    ap.add_argument("--seed", type=int, default=None)
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)
    seed = cfg["split"]["seed"] if args.seed is None else args.seed
    set_global_seed(seed)

    rc = load_receptors()
    fp = FingerprintConfig(**cfg["features"])

    smiles, Y, keys, synthetic = _load_labels(cfg["data"], rc.atlas_keys)
    print(
        f"{'SYNTHETIC ' if synthetic else ''}labels: "
        f"{len(smiles)} molecules x {len(keys)} receptors"
    )

    if synthetic:
        # labels already aligned to features via featurize_many above; re-featurise for clarity
        X, kept = featurize_many(smiles, fp)
        Y = Y[kept] if Y.shape[0] == len(smiles) else Y
    else:
        X, kept = featurize_many(smiles, fp)
        Y = Y[kept]
        smiles = [smiles[i] for i in kept]
    print(f"featurised {X.shape[0]} molecules -> {X.shape[1]} bits")
    if X.shape[0] < 3:
        raise ValueError("At least three valid molecules are required for a train/test split")

    scheme = cfg["split"]["scheme"]
    if scheme == "scaffold":
        tr, te = scaffold_split(smiles, cfg["split"]["test_frac"], cfg["split"]["seed"])
    else:
        tr, te = random_split(len(smiles), cfg["split"]["test_frac"], cfg["split"]["seed"])
    print(f"{scheme} split: {len(tr)} train / {len(te)} test")

    backend = cfg["model"]["backend"]
    selected_alpha = None
    alpha_grid = pd.DataFrame()
    if backend == "sklearn":
        selected_alpha, alpha_grid = _select_alpha(
            X, Y, smiles, tr, {**cfg["split"], **cfg["model"]}, keys, seed
        )
        model = MaskedMultiTaskRidge(receptor_keys=keys, alpha=selected_alpha)
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])
    elif backend == "gbdt":
        model = MaskedMultiTaskGBDT(
            receptor_keys=keys,
            **{
                k: v
                for k, v in cfg["model"].items()
                if k not in ("backend", "alpha", "alpha_grid")
            },
        )
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])
    elif backend == "torch":
        from smilesbrain.models.mlp import MaskedMultiTaskMLP, MLPConfig
        mcfg = MLPConfig(**{k: v for k, v in cfg["model"].items() if k not in ("backend", "alpha")})
        model = MaskedMultiTaskMLP(X.shape[1], len(keys), mcfg)
        model.fit(X[tr], Y[tr], X[te], Y[te])
        pred = model.predict(X[te])
    elif backend == "dmpnn":
        from smilesbrain.models.dmpnn import MaskedDMPNNRegressor

        dmpnn_tr, dmpnn_val = _inner_split(
            smiles, tr, cfg["split"], seed
        )
        model = MaskedDMPNNRegressor(
            receptor_keys=keys,
            **{k: v for k, v in cfg["model"].items() if k != "backend"},
        )
        model.fit(
            [smiles[i] for i in dmpnn_tr],
            Y[dmpnn_tr],
            validation_smiles=[smiles[i] for i in dmpnn_val],
            validation_Y=Y[dmpnn_val],
            test_smiles=[smiles[i] for i in te],
            test_Y=Y[te],
        )
        pred = model.predict([smiles[i] for i in te])
    else:
        raise ValueError(f"unknown backend {backend}")
    if not len(te):
        raise ValueError("The configured split produced an empty test fold")

    report = per_receptor_r2(Y[te], pred, keys)
    print("\nper-receptor R^2 (held-out):")
    print(report.to_string(index=False))
    non_sparse = {
        receptor.atlas_key for receptor in rc.receptors if not receptor.sparse
    }
    mean_r2 = report.loc[report["receptor"].isin(non_sparse), "r2"].mean(skipna=True)
    # A plain mean is dominated by receptors with a handful of test molecules (5HT4 n~25,
    # VAChT n~5), where R^2 can be large and negative. Report metrics that are not at the
    # mercy of those, so model comparisons mean something.
    well_sampled = report[report["n_test"] >= 200]
    weights = report["n_test"].where(report["r2"].notna(), 0)
    weighted_r2 = (
        float((report["r2"].fillna(0) * weights).sum() / weights.sum()) if weights.sum() else np.nan
    )
    mean_r2_well_sampled = well_sampled["r2"].mean(skipna=True)
    print(f"\nmean R^2 over non-sparse receptors: {mean_r2:.3f}")
    print(f"mean R^2 over receptors with n_test >= 200 (k={len(well_sampled)}): "
          f"{mean_r2_well_sampled:.3f}")
    print(f"n-weighted mean R^2 over all receptors: {weighted_r2:.3f}")
    if selected_alpha is not None:
        print(f"selected alpha: {selected_alpha:g}")

    out = Path(cfg["output"]["artifact"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if backend == "dmpnn":
        model.save(out)
    else:
        joblib.dump({"model": model, "fp": fp, "keys": keys, "training_X": X[tr]}, out)
    print(f"saved artifact -> {out}")
    report.to_csv(cfg["output"]["report"], index=False)
    print(f"saved report   -> {cfg['output']['report']}")
    comparison = Path(cfg["output"].get("comparison", "artifacts/model_compare.csv"))
    comparison.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "backend": backend,
                "mean_r2_non_sparse": mean_r2,
                "mean_r2_well_sampled": mean_r2_well_sampled,
                "mean_r2_n_weighted": weighted_r2,
                "mean_r2_all": report["r2"].mean(skipna=True),
                "n_test": len(te),
                "alpha": selected_alpha,
            }
        ]
    )
    if comparison.exists():
        prior = pd.read_csv(comparison)
        row = pd.concat([prior, row], ignore_index=True)
    row.to_csv(comparison, index=False)
    print(f"saved comparison -> {comparison}")
    if not alpha_grid.empty:
        alpha_grid.to_csv(out.with_name(out.stem + "_alpha_grid.csv"), index=False)
    manifest = write_run_manifest(
        out,
        config={"config_file": args.config, "config": cfg},
        seed=seed,
        input_paths=[args.config, cfg["data"]["labels"]],
    )
    print(f"saved manifest -> {manifest}")


if __name__ == "__main__":
    main()
