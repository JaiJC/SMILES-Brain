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
from smilesbrain.splits import random_split, scaffold_split


def _load_labels(cfg_data: dict, receptor_keys: list[str]):
    path = Path(cfg_data["labels"])
    if path.exists():
        table = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path, index_col=0)
        smiles = table.index.tolist()
        cols = [k for k in receptor_keys if k in table.columns]
        return smiles, table[cols].to_numpy(dtype=float), cols, False
    # synthetic fallback
    from smilesbrain.data.synthetic import synthetic_smiles, synthetic_labels
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/affinity_baseline.yaml")
    args = ap.parse_args()
    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    rc = load_receptors()
    fp = FingerprintConfig(**cfg["features"])

    smiles, Y, keys, synthetic = _load_labels(cfg["data"], rc.atlas_keys)
    print(f"{'SYNTHETIC ' if synthetic else ''}labels: {len(smiles)} molecules x {len(keys)} receptors")

    if synthetic:
        # labels already aligned to features via featurize_many above; re-featurise for clarity
        X, kept = featurize_many(smiles, fp)
        Y = Y[kept] if Y.shape[0] == len(smiles) else Y
    else:
        X, kept = featurize_many(smiles, fp)
        Y = Y[kept]
        smiles = [smiles[i] for i in kept]
    print(f"featurised {X.shape[0]} molecules -> {X.shape[1]} bits")

    scheme = cfg["split"]["scheme"]
    if scheme == "scaffold":
        tr, te = scaffold_split(smiles, cfg["split"]["test_frac"], cfg["split"]["seed"])
    else:
        tr, te = random_split(len(smiles), cfg["split"]["test_frac"], cfg["split"]["seed"])
    print(f"{scheme} split: {len(tr)} train / {len(te)} test")

    backend = cfg["model"]["backend"]
    if backend == "sklearn":
        model = MaskedMultiTaskRidge(receptor_keys=keys, alpha=cfg["model"]["alpha"])
        model.fit(X[tr], Y[tr])
        pred = model.predict(X[te])
    elif backend == "torch":
        from smilesbrain.models.mlp import MaskedMultiTaskMLP, MLPConfig
        mcfg = MLPConfig(**{k: v for k, v in cfg["model"].items() if k not in ("backend", "alpha")})
        model = MaskedMultiTaskMLP(X.shape[1], len(keys), mcfg)
        model.fit(X[tr], Y[tr], X[te], Y[te])
        pred = model.predict(X[te])
    else:
        raise ValueError(f"unknown backend {backend}")

    report = per_receptor_r2(Y[te], pred, keys)
    print("\nper-receptor R^2 (held-out):")
    print(report.to_string(index=False))
    print(f"\nmean R^2 over fitted receptors: {report['r2'].mean(skipna=True):.3f}")

    out = Path(cfg["output"]["artifact"])
    out.parent.mkdir(parents=True, exist_ok=True)
    if backend == "sklearn":
        joblib.dump({"model": model, "fp": fp, "keys": keys}, out)
        print(f"saved artifact -> {out}")
    report.to_csv(cfg["output"]["report"], index=False)
    print(f"saved report   -> {cfg['output']['report']}")


if __name__ == "__main__":
    main()
